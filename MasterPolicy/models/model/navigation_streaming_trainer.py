from huggingface_hub import hf_hub_url
import requests
import io
import torch
import json
import collections
import numpy as np
from tqdm import trange
from torch import nn


class NavigationStreamingTrainer(nn.Module):
    def __init__(self, args, vocab, model):
        super().__init__()
        self.args = args
        self.vocab = vocab
        self.model = model
        self.feat_pt = 'feat_conv.pt'
        self.max_subgoals = 10
        self.pad = 0
        self.test_mode = False

        # 将模型参数设置为当前类的参数
        for name, param in self.model.named_parameters():
            setattr(self, name, param)


    def run_train(self, splits, optimizer=None):
        '''
        流式训练循环 - 导航版本
        '''
        args = self.args

        # 创建流式数据集
        train_stream = self.create_streaming_dataset(
            splits['train'],
            augment=True  # 启用数据增强
        )
        valid_seen_stream = self.create_streaming_dataset(splits['valid_seen'])
        valid_unseen_stream = self.create_streaming_dataset(splits['valid_unseen'])

        # 调试模式处理
        if self.args.fast_epoch:
            def limited_stream(stream, limit):
                count = 0
                for item in stream:
                    if count >= limit:
                        break
                    yield item
                    count += 1

            train_stream = limited_stream(train_stream, 16)
            valid_seen_stream = limited_stream(valid_seen_stream, 1)
            valid_unseen_stream = limited_stream(valid_unseen_stream, 1)

        # 数据集分数处理
        if self.args.dataset_fraction > 0:
            def fraction_stream(stream, fraction):
                total_count = 0
                limit = int(len(splits['train']) * fraction * 0.7)
                for item in stream:
                    if total_count >= limit:
                        break
                    yield item
                    total_count += 1

            train_stream = fraction_stream(train_stream, self.args.dataset_fraction)

        # 初始化优化器
        optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.summary_writer = SummaryWriter(log_dir=args.dout)

        # 保存配置
        with open(os.path.join(args.dout, 'config.json'), 'wt') as f:
            json.dump(vars(args), f, indent=2)

        print("Saving to: %s" % args.dout)
        train_iter = 0

        for epoch in trange(0, args.epoch, desc='epoch'):
            m_train = collections.defaultdict(list)
            self.model.train()
            self.adjust_lr(optimizer, args.lr, epoch, args.decay_epoch)
            total_train_loss = []

            # 随机打乱流式数据（通过重新创建数据集）
            epoch_train_stream = self.create_streaming_dataset(
                splits['train'],
                augment=True,
                shuffle=True
            )

            batch_count = 0
            for batch in self.streaming_iterate(epoch_train_stream, args.batch):
                try:
                    feat = self.streaming_featurize(batch)

                    out = self.model.forward(feat)
                    preds = self.model.extract_preds(out, batch, feat)
                    loss = self.model.compute_loss(out, batch, feat)

                    # 记录损失
                    for k, v in loss.items():
                        ln = 'loss_' + k
                        m_train[ln].append(v.item())
                        self.summary_writer.add_scalar('train/' + ln, v.item(), train_iter)

                    # 反向传播
                    optimizer.zero_grad()
                    sum_loss = sum(loss.values())
                    sum_loss.backward()
                    optimizer.step()

                    self.summary_writer.add_scalar('train/loss', sum_loss, train_iter)
                    total_train_loss.append(float(sum_loss.detach().cpu()))
                    train_iter += len(batch)
                    batch_count += 1

                except Exception as e:
                    print(f"Error in batch processing: {e}")
                    continue

            # 保存检查点
            stats = {'epoch': epoch}
            if args.save_every_epoch:
                fsave = os.path.join(args.dout, 'net_epoch_%d.pth' % epoch)
            else:
                fsave = os.path.join(args.dout, 'latest.pth')

            self.save_checkpoint(epoch, batch_count, optimizer, args.dout)

            # 记录统计信息
            for split in stats.keys():
                if isinstance(stats[split], dict):
                    for k, v in stats[split].items():
                        self.summary_writer.add_scalar(split + '/' + k, v, train_iter)

            print(f"Epoch {epoch} completed")

    def create_streaming_dataset(self, task_list, augment=False, shuffle=False):
        '''
        创建流式数据集 - 导航版本
        '''
        if shuffle:
            import random
            task_list = random.sample(task_list, len(task_list))

        for task_info in task_list:
            if isinstance(task_info, dict):
                task_path = task_info['task']
                repeat_idx = task_info.get('repeat_idx', 0)
            else:
                task_path = task_info
                repeat_idx = 0

            if augment:
                # 数据增强：7种swapColor变体
                for swapColor in range(7):
                    task_data = self.load_streaming_task(task_path, repeat_idx, swapColor)
                    if task_data is not None:
                        yield task_data
            else:
                # 无数据增强
                task_data = self.load_streaming_task(task_path, repeat_idx, False)
                if task_data is not None:
                    yield task_data

    def load_streaming_task(self, task_path, repeat_idx, swapColor):
        '''
        流式加载单个任务数据
        '''
        try:
            # 加载JSON数据
            json_filename = f"{task_path}/pp/ann_{repeat_idx}.json"
            json_url = hf_hub_url(
                repo_id=self.args.huggingface_id,
                filename=json_filename,
                repo_type="dataset"
            )

            json_response = requests.get(json_url, timeout=30)
            if json_response.status_code != 200:
                return None

            ex = json.loads(json_response.content.decode('utf-8'))

            # 加载图像特征
            if swapColor == 0:
                pt_filename = f"train/{task_path}/{self.feat_pt}"
            elif swapColor in [1, 2]:
                pt_filename = f"train/{task_path}/feat_conv_colorSwap{swapColor}_panoramic.pt"
            else:
                pt_filename = f"train/{task_path}/feat_conv_onlyAutoAug{swapColor - 2}_panoramic.pt"

            pt_url = hf_hub_url(
                repo_id=self.args.huggingface_id,
                filename=pt_filename,
                repo_type="dataset"
            )

            pt_response = requests.get(pt_url, timeout=30)
            if pt_response.status_code != 200:
                return None

            with io.BytesIO(pt_response.content) as buffer:
                im = torch.load(buffer, map_location='cpu')

            return {
                'ex': ex,
                'im': im,
                'task_path': task_path,
                'repeat_idx': repeat_idx,
                'swapColor': swapColor
            }

        except Exception as e:
            print(f"Error loading task {task_path}: {e}")
            return None

    def streaming_iterate(self, data_stream, batch_size):
        '''
        流式批处理生成器
        '''
        current_batch = []
        for data_item in data_stream:
            if data_item is None:
                continue

            current_batch.append(data_item)

            if len(current_batch) >= batch_size:
                yield current_batch
                current_batch = []

        if current_batch:
            yield current_batch

    def streaming_featurize(self, batch_data):
        '''
        流式特征处理 - 导航版本
        '''
        device = torch.device('cuda') if self.args.gpu else torch.device('cpu')
        feat = collections.defaultdict(list)

        for data_item in batch_data:
            if data_item is None:
                continue

            ex = data_item['ex']
            swapColor = data_item['swapColor']
            im_data = data_item['im']

            try:
                # 辅助特征提取
                action_high_order = np.array([ah['action'] for ah in ex['num']['action_high']])
                low_to_high_idx = ex['num']['low_to_high_idx']
                action_high = action_high_order[low_to_high_idx]

                feat['action_high'].append(action_high)
                feat['action_high_order'].append(action_high_order)

                # GotoLocation 验证
                val_action_high = (
                            action_high == self.vocab['action_high'].word2index('GotoLocation', train=False)).astype(
                    np.int64)

                v = 0
                while v < (len(val_action_high) - 1):
                    if (val_action_high[v] - val_action_high[v + 1]) == 1:
                        val_action_high[v + 1] = 1
                        v += 1
                    v += 1
                val_action_high[-1] = 1

                # 序列化语言动作
                self.serialize_lang_action(ex, action_high_order)

                # 语言处理
                lang_goal, lang_instr = ex['num']['lang_goal'], ex['num']['lang_instr']
                lang_goal = self.zero_input(lang_goal) if self.args.zero_goal else lang_goal
                lang_instr = self.zero_input(lang_instr) if self.args.zero_instr else lang_instr

                feat['lang_goal'].append(lang_goal)
                feat['lang_instr'].append(lang_instr)

                # 动作处理
                alow = []
                alow_manip = []
                obj_high_indices = []

                for ia, a in enumerate(ex['num']['action_low']):
                    if val_action_high[ia] == 1 and a['action'] in self.vocab['action_low'].word2index(
                            ['<<pad>>', '<<seg>>', '<<stop>>', 'LookDown_15', 'LookUp_15', 'RotateLeft_90',
                             'RotateRight_90', 'MoveAhead_25'], train=False):
                        alow.append(a['action'])
                    elif val_action_high[ia] == 1:
                        alow.append(self.vocab['action_low'].word2index('Manipulate', train=False))

                    if not (a['action'] in self.vocab['action_low'].word2index(
                            ['<<pad>>', '<<seg>>', '<<stop>>', 'LookDown_15', 'LookUp_15', 'RotateLeft_90',
                             'RotateRight_90', 'MoveAhead_25'], train=False)):
                        alow_manip.append(a['action'])
                        obj_high_indices.append(low_to_high_idx[ia])

                feat['action_low'].append(alow)
                feat['action_low_manip'].append(alow_manip)
                feat['obj_high_indices'].append(obj_high_indices)

                # 辅助损失
                if self.args.subgoal_aux_loss_wt > 0:
                    feat['subgoals_completed'].append(
                        np.array(ex['num']['low_to_high_idx'])[
                            val_action_high.nonzero()[0].astype(int)] / self.max_subgoals
                    )

                if self.args.pm_aux_loss_wt > 0:
                    num_actions = len(alow)
                    subgoal_progress = [(i + 1) / float(num_actions) for i in range(num_actions)]
                    feat['subgoal_progress'].append(subgoal_progress)

                # 对象导航和掩码
                obj_list = [self.vocab['objnav'].word2index('<<nav>>')]
                high_idx = 0
                indices = []

                for a in ex['plan']['low_actions']:
                    if a['api_action']['action'] in ['MoveAhead', 'LookUp', 'LookDown', 'RotateRight', 'RotateLeft']:
                        if a['high_idx'] == (high_idx + 1):
                            obj_list.append(self.vocab['objnav'].word2index('<<nav>>', train=False))
                            high_idx += 1
                        continue

                    if a['api_action']['action'] == 'PutObject':
                        label = a['api_action']['receptacleObjectId'].split('|')
                    else:
                        label = a['api_action']['objectId'].split('|')

                    try:
                        class_name = label[4].split('_')[0] if len(label) >= 5 else label[0]
                        indices.append(classes.index(class_name))
                    except (IndexError, ValueError):
                        indices.append(0)

                    if a['high_idx'] == (high_idx + 1):
                        try:
                            class_name = (label[4].split('_')[0] if len(label) >= 5 else label[0]).lower()
                            obj_list.append(self.vocab['objnav'].word2index(class_name, train=False))
                        except:
                            obj_list.append(self.vocab['objnav'].word2index('<<nav>>', train=False))
                        high_idx += 1

                new_obj_list = [obj_list[o + 1] for o, obj in enumerate(obj_list) if
                                (obj == self.vocab['objnav'].word2index('<<nav>>'))]
                feat['objnav'].append(new_obj_list)
                feat['action_low_mask_label'].append(indices)

                # 图像特征处理
                if len(im_data) >= 5:
                    # 使用 val_action_high 作为掩码
                    val_indices = val_action_high.nonzero()[0]
                    feat['frames'].append(im_data[2][val_indices])
                    feat['frames_left'].append(im_data[0][val_indices])
                    feat['frames_up'].append(im_data[1][val_indices])
                    feat['frames_down'].append(im_data[3][val_indices])
                    feat['frames_right'].append(im_data[4][val_indices])
                else:
                    # 添加空张量
                    empty_tensor = torch.tensor([], device=device, dtype=torch.float)
                    for view in ['frames', 'frames_left', 'frames_up', 'frames_down', 'frames_right']:
                        feat[view].append(empty_tensor)

            except Exception as e:
                print(f"Error processing task {data_item.get('task_path', 'unknown')}: {e}")
                continue

        # 张量化和填充
        feat = self._tensorize_and_pad(feat, device)

        # 添加方向特征
        if self.orientation:
            feat = self._add_orientation_features(feat, device)

        return feat

    def _tensorize_and_pad(self, feat, device):
        '''
        张量化和填充逻辑
        '''
        # 实现与之前类似的张量化逻辑
        for k, v in feat.items():
            try:
                if k in {'lang_goal'}:
                    # 语言特征处理
                    seqs = [torch.tensor(vv, device=device) for vv in v if len(vv) > 0]
                    if seqs:
                        pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
                        seq_lengths = np.array([len(vv) for vv in v if len(vv) > 0])
                        embed_seq = self.emb_word(pad_seq)
                        packed_input = pack_padded_sequence(embed_seq, seq_lengths, batch_first=True,
                                                            enforce_sorted=False)
                        feat[k] = packed_input
                    else:
                        feat[k] = None

                elif k in {'lang_instr'}:
                    # 指令处理
                    num_instr = np.array([len(vv) for vv in v])
                    seqs = [torch.tensor(vvv, device=device) for vv in v for vvv in vv if len(vv) > 0]
                    if seqs:
                        pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
                        embed_seq = self.emb_word(pad_seq)
                        fin_seq = []
                        in_idx = 0
                        for l in num_instr:
                            if l > 0:
                                fin_seq.append(embed_seq[in_idx:in_idx + l])
                                in_idx += l
                            else:
                                fin_seq.append(torch.tensor([], device=device))
                        feat[k] = {'seq': fin_seq}
                    else:
                        feat[k] = {'seq': []}

                # 其他特征类型的处理...

            except Exception as e:
                print(f"Error processing feature {k}: {e}")
                feat[k] = torch.tensor([], device=device)

        return feat

    def _add_orientation_features(self, feat, device):
        '''
        添加方向特征
        '''
        import math

        def get_orientation(d, device):
            if d == 'left':
                h, v = -math.pi / 2, 0.0
            elif d == 'up':
                h, v = 0.0, -math.pi / 12
            elif d == 'down':
                h, v = 0.0, math.pi / 12
            elif d == 'right':
                h, v = math.pi / 2, 0.0
            else:  # 'front'
                h, v = 0.0, 0.0

            orientation = torch.cat([
                torch.cos(torch.ones(1, device=device) * h),
                torch.sin(torch.ones(1, device=device) * h),
                torch.cos(torch.ones(1, device=device) * v),
                torch.sin(torch.ones(1, device=device) * v),
            ]).unsqueeze(-1).unsqueeze(-1).repeat(1, 7, 7)

            return orientation

        # 为每个视角添加方向信息
        orientation_mapping = {
            'frames': 'front',
            'frames_left': 'left',
            'frames_up': 'up',
            'frames_down': 'down',
            'frames_right': 'right'
        }

        for view_key, direction in orientation_mapping.items():
            if view_key in feat and feat[view_key].numel() > 0:
                view_tensor = feat[view_key]
                orientation_tensor = get_orientation(direction, device)
                batch_size, channels, height, width = view_tensor.shape
                orientation_expanded = orientation_tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1)
                feat[view_key] = torch.cat([view_tensor, orientation_expanded], dim=1)

        return feat

    def save_checkpoint(self, epoch, batch_count, optimizer, dout_path):
        '''
        保存检查点
        '''
        if self.args.save_every_epoch:
            filename = f'net_epoch_{epoch}.pth'
        else:
            filename = 'latest.pth'

        checkpoint = {
            'metric': {'epoch': epoch, 'batch_count': batch_count},
            'model': self.model.state_dict(),
            'optim': optimizer.state_dict(),
            'args': self.args,
            'vocab': self.vocab,
        }

        torch.save(checkpoint, os.path.join(dout_path, filename))