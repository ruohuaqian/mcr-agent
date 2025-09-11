import os
import random
import json
import torch
import pprint
import collections
import numpy as np
from torch import nn
from tensorboardX import SummaryWriter
from tqdm import trange,tqdm
from Interactions.models.model import constants
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download
from functools import partial
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

    def _training_step(self, feat, batch, optimizer, train_iter):
        '''
        Helper function to perform a single training step.
        This avoids code duplication in the main loop.
        '''
        # optimizer backward pass
        optimizer.zero_grad()
        out = self.forward(feat)
        loss = self.compute_loss(out, batch, feat)
        sum_loss = sum(loss.values())
        sum_loss.backward()
        optimizer.step()

        # logging
        for k, v in loss.items():
            ln = 'loss_' + k
            self.summary_writer.add_scalar('train/' + ln, v.item(), train_iter)
        self.summary_writer.add_scalar('train/loss', sum_loss.item(), train_iter)

        return sum_loss.detach().cpu().item(), len(batch)

    def run_train(self, splits, args=None, optimizer=None):
        '''
        training loop
        '''

        # args
        args = args or self.args
        device = torch.device('cuda') if args.gpu else torch.device('cpu')
        self.to(device)

        # optimizer
        optimizer = optimizer or torch.optim.Adam(self.parameters(), lr=args.lr)

        # initialize summary writer for tensorboardX
        self.summary_writer = SummaryWriter(log_dir=args.dout)

        # dump config
        fconfig = os.path.join(args.dout, 'config.json')
        with open(fconfig, 'wt') as f:
            json.dump(vars(args), f, indent=2)

        # display dout
        print("Saving to: %s" % self.args.dout)

        # -------------------------------------------------------------------------
        # Data Preparation Stage
        # -------------------------------------------------------------------------

        if args.use_streaming:
            print("Using streaming data pipeline...")

            train_list_original = splits['train']
            augmented_train_list = []
            augmented_train_list.extend([dict(s, swapColor=0) for s in train_list_original])
            for i in range(1, 7):
                augmented_train_list.extend([dict(s, swapColor=i) for s in train_list_original])

            raw_datasets = DatasetDict()
            raw_datasets['train'] = Dataset.from_list(augmented_train_list)
            # Note: You would create your validation sets here too if you plan to use them
            # raw_datasets['valid_seen'] = Dataset.from_list(...)

            train_stream = raw_datasets['train'].to_iterable_dataset()

            # debugging: use to check if training loop works without waiting for full epoch
            if self.args.fast_epoch:
                train_stream = train_stream.take(args.batch * 10)

            p_preprocess_function = partial(self.preprocess_function)
            p_collate_fn = partial(self.collate_fn, device=device)
            # Apply processing and shuffling

            processed_train_stream = train_stream.map(p_preprocess_function)
            processed_train_stream = processed_train_stream.shuffle(buffer_size=1000, seed=args.seed)

            # ...
            train_loader = DataLoader(
                processed_train_stream,
                batch_size=args.batch,
                collate_fn=p_collate_fn,
                prefetch_factor = 2,  # <-- 推荐值：让数据加载更流畅
                num_workers = 0
            )
        else:
            print("Using local file pipeline...")

            # splits
            train_list = splits['train']
            valid_seen_list = splits['valid_seen']
            valid_unseen_list = splits['valid_unseen']

            train = [(s, 0) for s in train_list]
            train += [(s, i) for i in range(1, 7) for s in train_list]
            valid_seen = [(s, False) for s in valid_seen_list]
            valid_unseen = [(s, False) for s in valid_unseen_list]

            # debugging: chose a small fraction of the dataset
            if self.args.dataset_fraction > 0:
                small_train_size = int(self.args.dataset_fraction * 0.7)
                small_valid_size = int((self.args.dataset_fraction * 0.3) / 2)
                train = train[:small_train_size]
                valid_seen = valid_seen[:small_valid_size]
                valid_unseen = valid_unseen[:small_valid_size]

            # debugging: use to check if training loop works without waiting for full epoch
            if self.args.fast_epoch:
                train = train[-112:]
                valid_seen = valid_seen[:1]
                valid_unseen = valid_unseen[:1]

        # -------------------------------------------------------------------------
        # Main Epoch Loop
        # -------------------------------------------------------------------------

        train_iter = 0
        for epoch in trange(0, args.epoch, desc='epoch'):
            m_train = collections.defaultdict(list)
            self.train()
            self.adjust_lr(optimizer, args.lr, epoch, decay_epoch=args.decay_epoch)
            total_train_loss = []

            # Choose the correct iterator based on the mode
            if args.use_streaming:
                data_iterator = train_loader
            else:
                random.shuffle(train)  # shuffle every epoch
                data_iterator = self.iterate(train, args.batch)

            c_st = 0
            for data_item in tqdm(data_iterator, desc="batch"):
                c_st += 1

                # Unpack the data item correctly for each mode
                if args.use_streaming:
                    feat, batch = data_item
                else:
                    batch, feat = data_item

                if len(feat.get('sub_objs', [])) == 0:
                    continue

                # Perform a single training step
                sum_loss, batch_len = self._training_step(feat, batch, optimizer, train_iter)
                total_train_loss.append(sum_loss)
                train_iter += batch_len

                # Conditional checkpoint saving (original logic)
                if not (c_st % 2628):
                    fsave = os.path.join(args.dout, f'net_epoch_{epoch}_{c_st}.pth')
                    torch.save({
                        'metric': {'epoch': epoch, 'c_st': c_st},
                        'model': self.state_dict(),
                        'optim': optimizer.state_dict(),
                        'args': self.args,
                        'vocab': self.vocab,
                    }, fsave)

            # -------------------------------------------------------------------------
            # Post-Epoch Actions
            # -------------------------------------------------------------------------

            # NOTE: Your validation loop would go here. It would also need an if/else
            # to choose between the streaming validator and the list-based one.
            stats = {'epoch': epoch}

            # save the latest checkpoint
            if args.save_every_epoch:
                fsave = os.path.join(args.dout, 'net_epoch_%d.pth' % epoch)
            else:
                fsave = os.path.join(args.dout, 'latest.pth')
            torch.save({
                'metric': {'epoch': epoch},
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
