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
from PCC.models.model import constants
classes = [0] + constants.OBJECTS + ['AppleSliced', 'ShowerCurtain', 'TomatoSliced', 'LettuceSliced', 'Lamp', 'ShowerHead', 'EggCracked', 'BreadSliced', 'PotatoSliced', 'Faucet']
import torch.nn.functional as F

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
        self.emb_action_low = nn.Embedding(len(vocab['action_low']), args.demb)
        self.emb_action_high = nn.Embedding(len(vocab['action_high']), args.demb)
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
            total_train_loss = list()
            random.shuffle(train) # shuffle every epoch
            
            for batch, feat in self.iterate(train, args.batch):

                out = self.forward(feat)
                loss = self.compute_loss(out, batch, feat)
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
            
            # save the latest checkpoint
            if args.save_every_epoch:
                fsave = os.path.join(args.dout, 'net_epoch_%d.pth' % epoch)
            else:
                fsave = os.path.join(args.dout, 'latest.pth')
            torch.save({
                'metric': stats, #stats,
                'model': self.state_dict(),
                'optim': optimizer.state_dict(),
                'args': self.args,
                'vocab': self.vocab,
            }, fsave)

            # write stats
            for split in stats.keys():
                if isinstance(stats[split], dict):
                    for k, v in stats[split].items():
                        self.summary_writer.add_scalar(split + '/' + k, v, train_iter)
            pprint.pprint(stats)

    def run_train_stream(self, splits, args=None, optimizer=None):
        '''
        training loop with streaming data loading
        '''
        # args
        args = args or self.args

        # splits
        train_list = splits['train']
        valid_seen = splits['valid_seen']
        valid_unseen = splits['valid_unseen']

        # 创建流式数据集
        train_stream = ALFREDStreamingDataset(
            repo_id=self.args.huggingface_id,
            task_list=[{'task': s, 'repeat_idx': 0} for s in train_list],
            feat_pt=self.feat_pt,
            args=self.args
        )

        valid_seen_stream = ALFREDStreamingDataset(
            repo_id=self.args.huggingface_id,
            task_list=[{'task': s, 'repeat_idx': 0} for s in valid_seen],
            feat_pt=self.feat_pt,
            args=self.args
        )

        valid_unseen_stream = ALFREDStreamingDataset(
            repo_id=self.args.huggingface_id,
            task_list=[{'task': s, 'repeat_idx': 0} for s in valid_unseen],
            feat_pt=self.feat_pt,
            args=self.args
        )

        # 调试模式：使用小数据集
        if self.args.dataset_fraction > 0:
            # 对于流式数据，我们需要在迭代时进行限制
            train_size = int(len(train_list) * self.args.dataset_fraction * 0.7)
            valid_size = int(len(valid_seen) * (self.args.dataset_fraction * 0.3) / 2)

            def limited_stream(stream, limit):
                count = 0
                for item in stream:
                    if count >= limit:
                        break
                    yield item
                    count += 1

            train_stream = limited_stream(train_stream, train_size)
            valid_seen_stream = limited_stream(valid_seen_stream, valid_size)
            valid_unseen_stream = limited_stream(valid_unseen_stream, valid_size)

        # 初始化 tensorboard
        self.summary_writer = SummaryWriter(log_dir=args.dout)

        # 保存配置
        fconfig = os.path.join(args.dout, 'config.json')
        with open(fconfig, 'wt') as f:
            json.dump(vars(args), f, indent=2)

        # 优化器
        optimizer = optimizer or torch.optim.Adam(self.parameters(), lr=args.lr)

        print("Saving to: %s" % self.args.dout)
        best_loss = {'train': 1e10, 'valid_seen': 1e10, 'valid_unseen': 1e10}
        train_iter, valid_seen_iter, valid_unseen_iter = 0, 0, 0

        for epoch in trange(0, args.epoch, desc='epoch'):
            m_train = collections.defaultdict(list)
            self.train()
            self.adjust_lr(optimizer, args.lr, epoch, decay_epoch=args.decay_epoch)
            total_train_loss = list()

            # 使用流式迭代器
            for batch_idx, batch in enumerate(self.streaming_iterate(train_stream, args.batch)):
                feat = self.streaming_featurize(batch)

                out = self.forward(feat)
                loss = self.compute_loss(out, batch, feat)

                for k, v in loss.items():
                    ln = 'loss_' + k
                    m_train[ln].append(v.item())
                    self.summary_writer.add_scalar('train/' + ln, v.item(), train_iter)

                # 优化器反向传播
                optimizer.zero_grad()
                sum_loss = sum(loss.values())
                sum_loss.backward()
                optimizer.step()

                self.summary_writer.add_scalar('train/loss', sum_loss, train_iter)
                sum_loss = sum_loss.detach().cpu()
                total_train_loss.append(float(sum_loss))
                train_iter += len(batch)

                # 每 N 个batch进行验证
                if batch_idx % args.validation_interval == 0:
                    self.validate(valid_seen_stream, valid_unseen_stream, train_iter)

            # 保存检查点
            stats = {'epoch': epoch}
            if args.save_every_epoch:
                fsave = os.path.join(args.dout, 'net_epoch_%d.pth' % epoch)
            else:
                fsave = os.path.join(args.dout, 'latest.pth')

            torch.save({
                'metric': stats,
                'model': self.state_dict(),
                'optim': optimizer.state_dict(),
                'args': self.args,
                'vocab': self.vocab,
            }, fsave)

            pprint.pprint(stats)
    def run_pred(self, dev, args=None, name='dev', iter=0):
        '''
        validation loop
        '''
        args = args or self.args
        self.eval()
        total_correct = 0
        total_num = 0
        total_loss = 0
        dev_iter = iter
        for batch, feat in self.iterate(dev, args.batch):
            out = self.forward(feat)
            preds = F.softmax(out['out_obj'])
            preds = torch.argmax(preds, dim=1)

            total_num += len(preds)

            correct = (preds == feat['objnav'].cuda()).sum()
            total_correct+=correct

            loss = self.compute_loss(out, batch, feat)
            total_loss+=loss['objnav']
            del(batch)
            del(feat)

        accuracy = total_correct / total_num
        final_loss = total_loss / total_num
        return accuracy, final_loss

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

    def make_debug_streaming(self, preds, data):
        '''
        Generates a readable debug output by streaming and processing data in parallel.

        :param preds: Dictionary of predictions keyed by task ID.
        :param data: The list of task dictionaries from your splits file.
        :return: A dictionary with debug information.
        '''
        print(f"Starting debug data generation for {len(data)} tasks using {num_proc} processes...")

        # 1. Convert your list of tasks into a Hugging Face Dataset in memory
        #    We need to transpose the list of dicts into a dict of lists
        task_dict = {
            'task': [t['task'] for t in data],
            'repeat_idx': [t['repeat_idx'] for t in data]
        }
        task_dataset = Dataset.from_dict(task_dict)

        def _fetch_and_extract_info(task):
            # Fetch the full JSON from the Hub
            ex = load_task_json_from_hub(self.args.huggingface_id, task, self.args.pp_folder)

            i = get_task_and_ann_id(ex)
            # ----------------------------------------------------------------

            # Extract the ground truth information
            lang_goal = ex['turk_annotations']['anns'][ex['ann']['repeat_idx']]['task_desc']
            action_low = [a['discrete_action']['action'] for a in ex['plan']['low_actions']]

            return {
                'id': i,
                'lang_goal': lang_goal,
                'action_low': action_low,
            }

        # 3. Apply the function in parallel using .map()
        #    This is the "batch reading" step. It will fetch and process 'num_proc' tasks at a time.
        processed_dataset = task_dataset.map(
            _fetch_and_extract_info,
            num_proc=num_proc
        )

        # 4. Assemble the final debug dictionary
        #    Now we iterate through the PRE-PROCESSED data, which is very fast.
        debug = {}
        for item in tqdm(processed_dataset, desc="Assembling debug output"):
            task_id = item['id']
            if task_id in preds:
                debug[task_id] = {
                    'lang_goal': item['lang_goal'],
                    'action_low': item['action_low'],
                    'p_action_low': preds[task_id]['action_low'].split(),
                }

        return debug

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
        json_path = os.path.join(self.args.data, task['task'], '%s' % self.args.pp_folder,
                                 'ann_%d.json' % task['repeat_idx'])
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"ERROR loading JSON from {json_path}")
            raise


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
            # try:
            tasks = data[i:i+batch_size]
            batch = [(self.load_task_json(task), swapColor) for task, swapColor in tasks]
            feat = self.featurize(batch)
            yield batch, feat
            # except:
            #     print("no. of wrong trajs", error_no+1)
            #     error_no+=1
            #     continue

    def streaming_iterate(self, data_stream, batch_size):
        '''
        流式批处理生成器
        '''
        current_batch = []
        for data_item in data_stream:
            current_batch.append(data_item)

            if len(current_batch) >= batch_size:
                yield current_batch
                current_batch = []

        # 处理最后不足一个batch的数据
        if current_batch:
            yield current_batch

    def validate(self, valid_seen_stream, valid_unseen_stream, global_step):
        '''
        流式验证函数
        '''
        self.eval()

        with torch.no_grad():
            # 验证 seen
            valid_seen_loss = []
            for batch in self.streaming_iterate(valid_seen_stream, self.args.batch):
                feat = self.streaming_featurize(batch)
                out = self.forward(feat)
                loss = self.compute_loss(out, batch, feat)
                valid_seen_loss.append(sum(loss.values()).item())

            # 验证 unseen
            valid_unseen_loss = []
            for batch in self.streaming_iterate(valid_unseen_stream, self.args.batch):
                feat = self.streaming_featurize(batch)
                out = self.forward(feat)
                loss = self.compute_loss(out, batch, feat)
                valid_unseen_loss.append(sum(loss.values()).item())

        # 记录验证结果
        avg_seen_loss = sum(valid_seen_loss) / len(valid_seen_loss) if valid_seen_loss else 0
        avg_unseen_loss = sum(valid_unseen_loss) / len(valid_unseen_loss) if valid_unseen_loss else 0

        self.summary_writer.add_scalar('valid/seen_loss', avg_seen_loss, global_step)
        self.summary_writer.add_scalar('valid/unseen_loss', avg_unseen_loss, global_step)

        self.train()

    def get_task_list_from_hf(self):
        '''
        从 Hugging Face 数据集获取任务列表
        '''
        from datasets import load_dataset

        # 加载数据集元数据
        dataset = load_dataset(
            self.args.huggingface_id,
            streaming=True,
            repo_type="dataset"
        )

        # 获取所有任务路径
        task_list = []
        for split in ['train', 'valid_seen', 'valid_unseen']:
            if split in dataset:
                for example in dataset[split]:
                    task_list.append({
                        'task': example['task'],
                        'repeat_idx': example.get('repeat_idx', 0)
                    })

        return task_list

    def cleanup_streaming_resources(self):
        '''
        清理流式加载的临时资源
        '''
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
    def load(cls, fsave):
        '''
        load pth model from disk
        '''
        save = torch.load(fsave)
        model = cls(save['args'], save['vocab'])
        model.load_state_dict(save['model'])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.load_state_dict(save['optim'])
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
