import os
import random
import json
import torch
import pprint
import collections
import numpy as np
from torch import nn
from tensorboardX import SummaryWriter
from tqdm import trange
from MasterPolicy.models.model import constants
from huggingface_hub import hf_hub_url
classes = [0] + constants.OBJECTS + ['AppleSliced', 'ShowerCurtain', 'TomatoSliced', 'LettuceSliced', 'Lamp', 'ShowerHead', 'EggCracked', 'BreadSliced', 'PotatoSliced', 'Faucet']


class Module(nn.Module):

    def __init__(self, args, vocab):
        '''
        Base Seq2Seq agent with common train and val loops
        '''
        super().__init__()

        # sentinel tokens
        self.pad = 0
        self.seg = 1

        # args and vocab
        self.args = args
        self.vocab = vocab

        # emb modules
        self.emb_word = nn.Embedding(len(vocab['word']), args.demb)
        # self.emb_word = nn.Embedding(len(vocab['word']), args.demb)
        self.emb_action_low = nn.Embedding(len(vocab['action_low']), args.demb)
        self.emb_objnav = nn.Embedding(len(vocab['objnav']), args.demb)

        # end tokens
        self.stop_token = self.vocab['action_low'].word2index("<<stop>>", train=False)
        self.seg_token = self.vocab['action_low'].word2index("<<seg>>", train=False)

        # set random seed (Note: this is not the seed used to initialize THOR object locations)
        random.seed(a=args.seed)

        # summary self.writer
        self.summary_writer = None

    def run_train(self, splits, args=None, optimizer=None):
        '''
        training loop
        '''

        # args
        args = args or self.args

        # splits
        train_list = splits['train']
        valid_seen = splits['valid_seen']
        valid_unseen = splits['valid_unseen']

        train = [(s, False) for s in train_list]
        train = train + [(s, 1) for s in train_list] + [(s, 2) for s in train_list]
        train = train + [(s, 3) for s in train_list] + [(s, 4) for s in train_list] + [(s, 5) for s in train_list] + [(s, 6) for s in train_list]
        valid_seen = [(s, False) for s in valid_seen]
        valid_unseen = [(s, False) for s in valid_unseen]

        # debugging: chose a small fraction of the dataset
        if self.args.dataset_fraction > 0:
            small_train_size = int(self.args.dataset_fraction * 0.7)
            small_valid_size = int((self.args.dataset_fraction * 0.3) / 2)
            train = train[:small_train_size]
            valid_seen = valid_seen[:small_valid_size]
            valid_unseen = valid_unseen[:small_valid_size]

        # debugging: use to check if training loop works without waiting for full epoch
        if self.args.fast_epoch:
            train = train[-16:]
            valid_seen = valid_seen[:1]
            valid_unseen = valid_unseen[:1]

        # initialize summary writer for tensorboardX
        self.summary_writer = SummaryWriter(log_dir=args.dout)

        # dump config
        fconfig = os.path.join(args.dout, 'config.json')
        with open(fconfig, 'wt') as f:
            json.dump(vars(args), f, indent=2)

        # optimizer
        optimizer = optimizer or torch.optim.Adam(self.parameters(), lr=args.lr)

        # display dout
        print("Saving to: %s" % self.args.dout)
        best_loss = {'train': 1e10, 'valid_seen': 1e10, 'valid_unseen': 1e10}
        train_iter, valid_seen_iter, valid_unseen_iter = 0, 0, 0
        for epoch in trange(0, args.epoch, desc='epoch'):
            m_train = collections.defaultdict(list)
            self.train()
            self.adjust_lr(optimizer, args.lr, epoch, decay_epoch=args.decay_epoch)
            # p_train = {}
            total_train_loss = list()
            random.shuffle(train) # shuffle every epoch
            # obj_classes = []
            
            c_st = 0
            for batch, feat in self.iterate(train, args.batch):
                c_st += 1
                
                out = self.forward(feat)
                preds = self.extract_preds(out, batch, feat)
                # p_train.update(preds)
                loss = self.compute_loss(out, batch, feat)
                # print(loss.keys())
                for k, v in loss.items():
                    ln = 'loss_' + k
                    m_train[ln].append(v.item())
                    self.summary_writer.add_scalar('train/' + ln, v.item(), train_iter)

                # optimizer backward pass
                optimizer.zero_grad()
                sum_loss = sum(loss.values())
                sum_loss.backward()
                optimizer.step()

                self.summary_writer.add_scalar('train/loss', sum_loss, train_iter)
                sum_loss = sum_loss.detach().cpu()
                total_train_loss.append(float(sum_loss))
                train_iter += self.args.batch
               
            stats = {'epoch': epoch,}
           

            if args.save_every_epoch:
                fsave = os.path.join(args.dout, 'net_epoch_%d.pth' % epoch)
            else:
                fsave = os.path.join(args.dout, 'latest.pth')
            torch.save({
                'metric': {'epoch': epoch, 'c_st':c_st}, #stats,
                'model': self.state_dict(),
                'optim': optimizer.state_dict(),
                'args': self.args,
                'vocab': self.vocab,
            }, fsave)

            ## debug action output json for train
          
            # write stats
            for split in stats.keys():
                if isinstance(stats[split], dict):
                    for k, v in stats[split].items():
                        self.summary_writer.add_scalar(split + '/' + k, v, train_iter)
            pprint.pprint(stats)

    def run_train_stream(self, splits, optimizer=None):
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
        optimizer = optimizer or torch.optim.Adam(self.parameters(), lr=args.lr)
        self.summary_writer = SummaryWriter(log_dir=args.dout)

        # 保存配置
        with open(os.path.join(args.dout, 'config.json'), 'wt') as f:
            json.dump(vars(args), f, indent=2)

        print("Saving to: %s" % args.dout)
        train_iter = 0

        for epoch in trange(0, args.epoch, desc='epoch'):
            m_train = collections.defaultdict(list)
            self.train()
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

                    out = self.forward(feat)
                    preds = self.extract_preds(out, batch, feat)
                    loss = self.compute_loss(out, batch, feat)

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
            'model': self.state_dict(),
            'optim': optimizer.state_dict(),
            'args': self.args,
            'vocab': self.vocab,
        }

        torch.save(checkpoint, os.path.join(dout_path, filename))
    def run_pred(self, dev, args=None, name='dev', iter=0):
        '''
        validation loop
        '''
        args = args or self.args
        m_dev = collections.defaultdict(list)
        p_dev = {}
        self.eval()
        total_loss = list()
        dev_iter = iter
        for batch, feat in self.iterate(dev, args.batch):
            out = self.forward(feat)
            preds = self.extract_preds(out, batch, feat)
            p_dev.update(preds)
            loss = self.compute_loss(out, batch, feat)
            for k, v in loss.items():
                ln = 'loss_' + k
                m_dev[ln].append(v.item())
                self.summary_writer.add_scalar("%s/%s" % (name, ln), v.item(), dev_iter)
            sum_loss = sum(loss.values())
            self.summary_writer.add_scalar("%s/loss" % (name), sum_loss, dev_iter)
            total_loss.append(float(sum_loss.detach().cpu()))
            dev_iter += len(batch)

        m_dev = {k: sum(v) / len(v) for k, v in m_dev.items()}
        total_loss = sum(total_loss) / len(total_loss)
        return p_dev, dev_iter, total_loss, m_dev

    def featurize(self, batch):
        raise NotImplementedError()

    def forward(self, feat, max_decode=100):
        raise NotImplementedError()

    def extract_preds(self, out, batch, feat):
        raise NotImplementedError()

    def compute_loss(self, out, batch, feat):
        raise NotImplementedError()

    def compute_metric(self, preds, data):
        raise NotImplementedError()

    def get_task_and_ann_id(self, ex):
        '''
        single string for task_id and annotation repeat idx
        '''
        return "%s_%s" % (ex['task_id'], str(ex['ann']['repeat_idx']))

    def make_debug(self, preds, data):
        '''
        readable output generator for debugging
        '''
        debug = {}
        for task in data:
            ex = self.load_task_json(task)
            i = self.get_task_and_ann_id(ex)
            debug[i] = {
                'lang_goal': ex['turk_annotations']['anns'][ex['ann']['repeat_idx']]['task_desc'],
                'action_low': [a['discrete_action']['action'] for a in ex['plan']['low_actions']],
                'p_action_low': preds[i]['action_low'].split(),
            }
        return debug

    def load_task_json(self, task):
        '''
        load preprocessed json from disk
        '''
        json_path = os.path.join(self.args.data, task['task'], '%s' % self.args.pp_folder, 'ann_%d.json' % task['repeat_idx'])
        with open(json_path) as f:
            data = json.load(f)
        return data

    def get_task_root(self, ex):
        '''
        returns the folder path of a trajectory
        '''
        return os.path.join(self.args.data, ex['split'], *(ex['root'].split('/')[-2:]))

    def iterate(self, data, batch_size):
        '''
        breaks dataset into batch_size chunks for training
        '''
        error_no=0
        for i in trange(0, len(data), batch_size, desc='batch'):
            try:
                tasks = data[i:i+batch_size]
                batch = [(self.load_task_json(task), swapColor) for task, swapColor in tasks]
                feat = self.featurize(batch)
                yield batch, feat
            except Exception as e:
                error_no += 1
                print(f"no. {error_no} of wrong trajs, {e}")
                continue

    def zero_input(self, x, keep_end_token=True):
        '''
        pad input with zeros (used for ablations)
        '''
        end_token = [x[-1]] if keep_end_token else [self.pad]
        return list(np.full_like(x[:-1], self.pad)) + end_token

    def zero_input_list(self, x, keep_end_token=True):
        '''
        pad a list of input with zeros (used for ablations)
        '''
        end_token = [x[-1]] if keep_end_token else [self.pad]
        lz = [list(np.full_like(i, self.pad)) for i in x[:-1]] + end_token
        return lz

    @staticmethod
    def adjust_lr(optimizer, init_lr, epoch, decay_epoch=5):
        '''
        decay learning rate every decay_epoch
        '''
        lr = init_lr * (0.1 ** (epoch // decay_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    @classmethod
    def load(cls, fsave, fargs=None):
        save = torch.load(fsave)
        vocab = save['vocab']
        args_to_use = fargs or save['args']
        if args_to_use is None:
            raise RuntimeError("Need fargs or checkpoint['args']")
        model = cls(args_to_use, vocab)
        model.load_state_dict(save['model'])
        optimizer = torch.optim.Adam(model.parameters(), lr=getattr(args_to_use, "lr", 1e-3))
        return model, optimizer

    @classmethod
    def has_interaction(cls, action):
        '''
        check if low-level action is interactive
        '''
        non_interact_actions = ['MoveAhead', 'Rotate', 'Look', '<<stop>>', '<<pad>>', '<<seg>>']
        if any(a in action for a in non_interact_actions):
            return False
        else:
            return True
