import os
import random
import requests
import io
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
from huggingface_hub import login
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
    def setup_hf_auth(self):
        # use environ path
        if os.environ.get('HF_TOKEN'):
            login(token=os.environ.get('HF_TOKEN'))

    def run_train_stream(self, splits, optimizer=None):
        '''
        流式训练循环 - 导航版本
        '''
        self.setup_hf_auth()
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

            json_response = requests.get(
                json_url,
                timeout=120,
                stream=False
            )
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

            pt_response = requests.get(
                pt_url,
                timeout=300,
                stream=True
            )
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
    def streaming_featurize(self, batch_data):
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

    def load_task_json_from_hub(self, task):
        '''
        Downloads and loads a single task JSON from the Hugging Face Hub.
        '''
        try:
            filename = f"{task['task']}/{self.args.pp_folder}/ann_{task['repeat_idx']}.json"
            json_url = hf_hub_url(
                repo_id=self.args.huggingface_id,
                filename=filename,
                repo_type="dataset"
            )
            response = requests.get(json_url, timeout=30)
            if response.status_code != 200:
                return None

            with io.BytesIO(pt_response.content) as buffer:
                im = torch.load(buffer, map_location='cpu')

            return im
        except Exception as e:
            print(f"Error loading task from Hub for task='{task.get('task')}': {e}")
            raise e
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
