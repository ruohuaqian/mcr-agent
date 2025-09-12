from huggingface_hub import hf_hub_url
import requests
import io
import torch
import json
import collections
from itertools import groupby
from operator import itemgetter


class SubgoalStreamingTrainer:
    def __init__(self, split, optimizer, args, vocab):
        self.args = args
        self.vocab = vocab
        self.feat_pt = 'feat_conv.pt'
        self.max_subgoals = 10
        self.stop_token = self.vocab['action_low'].word2index('<<stop>>')
        self.pad = 0
        self.test_mode = False
        self.optimizer = optimizer
        self.split = split

    def run_train(self, splits):
        '''
        流式训练循环
        '''
        args = self.args

        # 创建流式数据集
        train_stream = self.create_streaming_dataset(
            splits['train'],
            augment=True  # 启用数据增强（7种swapColor）
        )
        valid_seen_stream = self.create_streaming_dataset(splits['valid_seen'])
        valid_unseen_stream = self.create_streaming_dataset(splits['valid_unseen'])

        # 调试模式处理
        if args.fast_epoch:
            # 对于流式数据，我们需要在迭代时限制数量
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

        # 初始化优化器和记录器
        optimizer = self.optimizer or torch.optim.Adam(self.parameters(), lr=args.lr)
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
            batch_count = 0

            # 流式训练循环
            for batch in self.streaming_iterate(train_stream, args.batch):
                try:
                    feat = self.streaming_featurize(batch)

                    # 跳过没有子目标数据的批次
                    if len(feat['sub_objs']) == 0:
                        continue

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

                    # 定期保存检查点
                    if batch_count % 2628 == 0:
                        self.save_checkpoint(epoch, batch_count, optimizer, args.dout)

                except Exception as e:
                    print(f"Error in batch processing: {e}")
                    continue

            # 保存epoch检查点
            self.save_checkpoint(epoch, 0, optimizer, args.dout, is_epoch_end=True)

    def create_streaming_dataset(self, task_list, augment=False):
        '''
        创建流式数据集
        '''
        for task in task_list:
            if augment:
                # 数据增强：7种swapColor变体
                for swapColor in range(7):
                    yield self.load_streaming_task(task, swapColor)
            else:
                # 无数据增强
                yield self.load_streaming_task(task, False)

    def load_streaming_task(self, task_path, swapColor):
        '''
        流式加载单个任务数据
        '''
        try:
            # 加载JSON数据
            json_filename = f"{task_path}/pp/ann_0.json"  # 假设repeat_idx=0
            json_url = hf_hub_url(
                repo_id=self.args.huggingface_id,
                filename=json_filename,
                repo_type="dataset"
            )

            json_response = requests.get(json_url, timeout=30)
            ex = json.loads(json_response.content.decode('utf-8'))

            # 加载图像特征
            if not swapColor:
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
            with io.BytesIO(pt_response.content) as buffer:
                im = torch.load(buffer, map_location='cpu')

            return {
                'ex': ex,
                'im': im,
                'task_path': task_path,
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
        流式特征处理
        '''
        device = torch.device('cuda') if self.args.gpu else torch.device('cpu')
        feat = collections.defaultdict(list)

        for data_item in batch_data:
            ex = data_item['ex']
            im_data = data_item['im']

            # 辅助特征提取
            action_high_order = np.array([ah['action'] for ah in ex['num']['action_high']])
            low_to_high_idx = ex['num']['low_to_high_idx']
            action_high = action_high_order[low_to_high_idx]

            feat['action_high'].append(action_high)
            feat['action_high_order'].append(action_high_order)

            # GotoLocation 验证
            val_action_high = (action_high == self.vocab['action_high'].word2index('GotoLocation', train=False)).astype(
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

            # 子目标分析
            subgoal_analysis = self.args.subgoal_analysis
            alow_list = [a['action'] for a in ex['num']['action_low']]
            sub_action_high = (
                        action_high == self.vocab['action_high'].word2index(subgoal_analysis, train=False)).astype(
                np.int64)
            sub_actions = np.array(alow_list)[sub_action_high.nonzero()[0].astype(int)]

            # 语言处理
            lang_goal, lang_instr = ex['num']['lang_goal'], ex['num']['lang_instr']
            lang_instr_sep = ex['num']['lang_instr_sep']

            lang_goal = self.zero_input(lang_goal) if self.args.zero_goal else lang_goal
            lang_instr = self.zero_input(lang_instr) if self.args.zero_instr else lang_instr
            lang_instr_sep = self.zero_input(lang_instr_sep) if self.args.zero_instr else lang_instr_sep

            feat['lang_goal'].append(lang_goal)
            feat['lang_instr'].append(lang_instr)

            sub_indices = (
            np.array(action_high_order == self.vocab['action_high'].word2index(subgoal_analysis)).astype(int).nonzero()[
                0])
            subgoal_lang = np.array(lang_instr_sep)[sub_indices]
            feat['sub_indices'].append(list(sub_indices))

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
                    np.array(ex['num']['low_to_high_idx'])[val_action_high.nonzero()[0].astype(int)] / self.max_subgoals
                )

            if self.args.pm_aux_loss_wt > 0:
                num_actions = len(alow)
                subgoal_progress = [(i + 1) / float(num_actions) for i in range(num_actions)]
                feat['subgoal_progress'].append(subgoal_progress)

            # 对象导航
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

                indices.append(classes.index(label[4].split('_')[0] if len(label) >= 5 else label[0]))

                if a['high_idx'] == (high_idx + 1):
                    obj_list.append(self.vocab['objnav'].word2index(
                        (label[4].split('_')[0] if len(label) >= 5 else label[0]).lower(), train=False))
                    high_idx += 1

            new_obj_list = [obj_list[o + 1] for o, obj in enumerate(obj_list) if
                            (obj == self.vocab['objnav'].word2index('<<nav>>'))]
            feat['objnav'].append(new_obj_list)
            feat['action_low_mask_label'].append(indices)

            # 子目标帧处理
            if len(sub_actions) > 0:
                sah = []
                for k, g in groupby(enumerate(sub_action_high.nonzero()[0]), lambda ix: ix[0] - ix[1]):
                    sah.append(np.array(list(map(itemgetter(1), g))))

                # 使用流式加载的图像数据
                sub_frames_high = np.copy(sub_action_high)
                for sfh in range(1, len(sub_frames_high)):
                    if sub_action_high[sfh] < sub_action_high[sfh - 1]:
                        sub_frames_high[sfh] = 1

                sub_frames = im_data[2][sub_frames_high.nonzero()[0]]

                sac_ind = 0
                sfr_ind = 0
                for sii, s in enumerate(sah):
                    so = np.array(indices)[
                        (np.array(obj_high_indices) == np.array(low_to_high_idx)[s][0]).astype(int).nonzero()[0]]

                    feat['sub_objs'].append(so)
                    feat['sub_actions'].append(list(sub_actions[sac_ind:sac_ind + len(s)]) + [self.stop_token])
                    feat['sub_frames'].append(sub_frames[sfr_ind:sfr_ind + len(s) + 1])
                    feat['sub_lang'].append(subgoal_lang[sii])

                    sac_ind += len(s)
                    sfr_ind += (len(s) + 1)

        # 张量化和填充（保持原有逻辑）
        return self._tensorize_and_pad(feat, device)

    def _tensorize_and_pad(self, feat, device):
        '''
        张量化和填充逻辑
        '''
        # 这里保持原有的 tensorize_and_pad 逻辑
        for k, v in feat.items():
            if k in {'lang_goal', 'sub_lang'}:
                seqs = [torch.tensor(vv, device=device) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
                seq_lengths = np.array(list(map(len, v)))
                embed_seq = self.emb_word(pad_seq)
                packed_input = pack_padded_sequence(embed_seq, seq_lengths, batch_first=True, enforce_sorted=False)
                feat[k] = {'seqs': seqs, 'emb': embed_seq, 'pi': packed_input, 'seq_len': seq_lengths}


            elif k in {'lang_instr'}:
                # language embedding and padding
                num_instr = np.array(list(map(len, v)))
                seqs = [torch.tensor(vvv, device=device) for vv in v for vvv in vv]

                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)

                # lang_pad = (pad_seq==0)
                embed_seq = self.emb_word(pad_seq)
                fin_seq = []

                in_idx = 0
                for l in num_instr:
                    fin_seq.append(embed_seq[in_idx:in_idx + l])

                    in_idx += l
                feat[k] = {'seq': fin_seq}

            elif k in {'action_low_mask'}:
                # mask padding
                seqs = [torch.tensor(vv, device=device, dtype=torch.float) for vv in v]
                feat[k] = seqs
            elif k in {'action_low_mask_label'}:
                # label
                seqs = torch.tensor([vvv for vv in v for vvv in vv], device=device, dtype=torch.long)
                feat[k] = seqs
            elif k in {'subgoal_progress', 'subgoals_completed'}:
                # auxillary padding
                seqs = [torch.tensor(vv, device=device, dtype=torch.float) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
                feat[k] = pad_seq
            elif k in {'action_high', 'action_high_order'}:
                seqs = [torch.tensor(vv, device=device, dtype=torch.long) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True,
                                       padding_value=self.vocab['action_high'].word2index('<<pad>>'))
                feat[k] = pad_seq
            elif k in {'objnav'}:
                seqs = [torch.tensor(vv, device=device, dtype=torch.long) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.vocab['objnav'].word2index('<<pad>>'))
                embed_seq = self.emb_objnav(pad_seq)
                feat[k] = embed_seq
            else:
                # default: tensorize and pad sequence
                seqs = [torch.tensor(vv, device=device, dtype=torch.float if (
                        'sub_frames' in k or 'frames' in k or 'orientation' in k) else torch.long) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
                feat[k] = pad_seq

        return feat

    def save_checkpoint(self, epoch, batch_count, optimizer, dout_path, is_epoch_end=False):
        '''
        保存检查点
        '''
        if is_epoch_end:
            filename = f'net_epoch_{epoch}.pth'
        else:
            filename = f'net_epoch_{epoch}_{batch_count}.pth'

        checkpoint = {
            'metric': {'epoch': epoch, 'batch_count': batch_count},
            'model': self.state_dict(),
            'optim': optimizer.state_dict(),
            'args': self.args,
            'vocab': self.vocab,
        }

        torch.save(checkpoint, os.path.join(dout_path, filename))