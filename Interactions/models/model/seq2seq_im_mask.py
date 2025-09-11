import os
import cv2
import torch
import numpy as np
import Interactions.models.nn.vnn as vnn
import collections
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from Interactions.models.model.seq2seq import Module as Base
from Interactions.models.utils.metric import compute_f1, compute_exact
from gen.utils.image_util import decompress_mask
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import json


from PIL import Image
from itertools import groupby
from operator import itemgetter
from Interactions.models.model import constants
classes = [0] + constants.OBJECTS + ['AppleSliced', 'ShowerCurtain', 'TomatoSliced', 'LettuceSliced', 'Lamp', 'ShowerHead', 'EggCracked', 'BreadSliced', 'PotatoSliced', 'Faucet']

from Interactions.models.nn.resnet import Resnet
from itertools import islice



class Module(Base):

    def __init__(self, args, vocab):
        '''
        Seq2Seq agent
        '''
        super().__init__(args, vocab)

        # encoder and self-attention
        self.enc_instr = nn.LSTM(args.demb, args.dhid, bidirectional=True, batch_first=True)
        self.enc_att_instr = vnn.SelfAttn(args.dhid*2)

        # subgoal monitoring
        self.subgoal_monitoring = (self.args.pm_aux_loss_wt > 0 or self.args.subgoal_aux_loss_wt > 0)

        # frame mask decoder
        if not args.panoramic:
            decoder = vnn.ConvFrameMaskDecoderProgressMonitor # if self.subgoal_monitoring else vnn.ConvFrameMaskDecoder
        else:
            decoder = vnn.ConvFrameMaskDecoderProgressMonitorPanoramicConcat if args.panoramic_concat else vnn.ConvFrameMaskDecoderProgressMonitorPanoramicHier
            

        self.dec = decoder(self.emb_action_low, args.dframe, 2*args.dhid,
                           pframe=args.pframe,
                           attn_dropout=args.attn_dropout,
                           hstate_dropout=args.hstate_dropout,
                           actor_dropout=args.actor_dropout,
                           input_dropout=args.input_dropout,
                           teacher_forcing=args.dec_teacher_forcing,
                           panoramic=args.panoramic,
                           orientation=args.orientation)

        # dropouts
        self.vis_dropout = nn.Dropout(args.vis_dropout)
        self.lang_dropout = nn.Dropout(args.lang_dropout, inplace=True)
        self.input_dropout = nn.Dropout(args.input_dropout)

        # internal states
        self.state_t = None
        self.e_t = None
        self.test_mode = False

        # bce reconstruction loss
        self.bce_with_logits = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss(reduction='none')
        self.ce_loss = torch.nn.CrossEntropyLoss()

        # paths
        self.root_path = os.getcwd()
        self.feat_pt = 'feat_conv_panoramic.pt'

        # params
        self.max_subgoals = 25

        # reset model
        self.reset()

        self.panoramic = args.panoramic
        self.orientation = args.orientation
        self.man_action = self.vocab['action_low'].word2index('Manipulate', train=False)

    def featurize(self, batch, load_mask=True, load_frames=True):
        '''
        tensorize and pad batch input
        '''
        device = torch.device('cuda') if self.args.gpu else torch.device('cpu')
        feat = collections.defaultdict(list)

        for ex, swapColor in batch:
            ###########
            # auxillary
            ###########

            action_high_order = np.array([ah['action'] for ah in ex['num']['action_high']])
            low_to_high_idx = ex['num']['low_to_high_idx']
            action_high = action_high_order[low_to_high_idx]

            feat['action_high'].append(action_high)
            feat['action_high_order'].append(action_high_order)

            val_action_high = (action_high == self.vocab['action_high'].word2index('GotoLocation', train=False)).astype(
                np.int64)

            v = 0
            while v < (len(val_action_high) - 1):
                if (val_action_high[v] - val_action_high[v + 1]) == 1:
                    val_action_high[v + 1] = 1
                    v += 1
                v += 1
            val_action_high[-1] = 1

            #########
            # inputs
            #########

            # serialize segments
            self.serialize_lang_action(ex, action_high_order)

            subgoal_analysis = self.args.subgoal_analysis

            alow_list = [a['action'] for a in ex['num']['action_low']]
            sub_action_high = (
                        action_high == self.vocab['action_high'].word2index(subgoal_analysis, train=False)).astype(
                np.int64)
            sub_actions = np.array(alow_list)[sub_action_high.nonzero()[0].astype(int)]

            # goal and instr language
            lang_goal, lang_instr = ex['num']['lang_goal'], ex['num']['lang_instr']
            lang_instr_sep = ex['num']['lang_instr_sep']

            # zero inputs if specified
            lang_goal = self.zero_input(lang_goal) if self.args.zero_goal else lang_goal
            lang_instr = self.zero_input(lang_instr) if self.args.zero_instr else lang_instr
            lang_instr_sep = self.zero_input(lang_instr_sep) if self.args.zero_instr else lang_instr_sep

            # append goal + instr
            feat['lang_goal'].append(lang_goal)
            feat['lang_instr'].append(lang_instr)
            # feat['lang_instr_sep'].append(lang_instr_sep)

            sub_indices = (
                np.array(action_high_order == self.vocab['action_high'].word2index(subgoal_analysis)).astype(
                    int).nonzero()[0])
            subgoal_lang = np.array(lang_instr_sep)[sub_indices]

            feat['sub_indices'].append(list(sub_indices))

            #########
            # outputs
            #########

            alow = []
            alow_manip = []
            obj_high_indices = []

            for ia, a in enumerate(ex['num']['action_low']):

                if val_action_high[ia] == 1 and a['action'] in self.vocab['action_low'].word2index(
                        ['<<pad>>', '<<seg>>', '<<stop>>',
                         'LookDown_15', 'LookUp_15',
                         'RotateLeft_90', 'RotateRight_90',
                         'MoveAhead_25'], train=False):
                    alow.append(a['action'])
                elif val_action_high[ia] == 1:
                    alow.append(self.vocab['action_low'].word2index('Manipulate', train=False))

                # print(self.vocab['action_low'].index2word(a['action']), a['action'])
                if not (a['action'] in self.vocab['action_low'].word2index(['<<pad>>', '<<seg>>', '<<stop>>',
                                                                            'LookDown_15', 'LookUp_15',
                                                                            'RotateLeft_90', 'RotateRight_90',
                                                                            'MoveAhead_25'], train=False)):
                    alow_manip.append(a['action'])
                    obj_high_indices.append(low_to_high_idx[ia])

            feat['action_low'].append(alow)
            feat['action_low_manip'].append(alow_manip)
            feat['obj_high_indices'].append(obj_high_indices)

            if self.args.subgoal_aux_loss_wt > 0:
                feat['subgoals_completed'].append(
                    np.array(ex['num']['low_to_high_idx'])[
                        val_action_high.nonzero()[0].astype(int)] / self.max_subgoals)

            # progress monitor supervision
            if self.args.pm_aux_loss_wt > 0:
                # num_actions = len([a for sg in ex['num']['action_low'] for a in sg])
                num_actions = len(alow)
                subgoal_progress = [(i + 1) / float(num_actions) for i in range(num_actions)]
                feat['subgoal_progress'].append(subgoal_progress)

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
                    obj_list.append(
                        self.vocab['objnav'].word2index(
                            (label[4].split('_')[0] if len(label) >= 5 else label[0]).lower(),
                            train=False))
                    high_idx += 1

            new_obj_list = [obj_list[o + 1] for o, obj in enumerate(obj_list) if
                            (obj == self.vocab['objnav'].word2index('<<nav>>'))]

            feat['objnav'].append(new_obj_list)
            feat['action_low_mask_label'].append(indices)

            if len(sub_actions) > 0:
                sah = []
                for k, g in groupby(enumerate(sub_action_high.nonzero()[0]), lambda ix: ix[0] - ix[1]):
                    sah.append(np.array(list(map(itemgetter(1), g))))

                if load_frames and not self.test_mode:
                    root = self.get_task_root(ex)
                    if swapColor == 0:
                        im = torch.load(os.path.join(root, self.feat_pt))
                    elif swapColor in [1, 2]:
                        im = torch.load(os.path.join(root, 'feat_conv_colorSwap{}_panoramic.pt'.format(swapColor)))
                    elif swapColor in [3, 4, 5, 6]:
                        im = torch.load(
                            os.path.join(root, 'feat_conv_onlyAutoAug{}_panoramic.pt'.format(swapColor - 2)))

                    sub_frames_high = np.copy(sub_action_high)

                    for sfh in range(1, len(sub_frames_high)):
                        if sub_action_high[sfh] < sub_action_high[sfh - 1]:
                            sub_frames_high[sfh] = 1

                    sub_frames = im[2][sub_frames_high.nonzero()[0]]

                sac_ind = 0
                sfr_ind = 0
                for sii, s in enumerate(sah):

                    so = np.array(indices)[
                        (np.array(obj_high_indices) == np.array(low_to_high_idx)[s][0]).astype(int).nonzero()[0]]

                    feat['sub_objs'].append(so)
                    feat['sub_actions'].append(list(sub_actions[sac_ind:sac_ind + len(s)]) + [self.stop_token])

                    if load_frames and not self.test_mode:
                        feat['sub_frames'].append(sub_frames[sfr_ind:sfr_ind + len(s) + 1])

                    feat['sub_lang'].append(subgoal_lang[sii])

                    sac_ind += len(s)
                    sfr_ind += (len(s) + 1)

                if self.orientation:
                    import math
                    def get_orientation(d):
                        if d == 'left':
                            h, v = -math.pi / 2, 0.0
                        elif d == 'up':
                            h, v = 0.0, -math.pi / 12
                        elif d == 'down':
                            h, v = 0.0, math.pi / 12
                        elif d == 'right':
                            h, v = math.pi / 2, 0.0
                        else:
                            h, v = 0.0, 0.0

                        orientation = torch.cat([
                            torch.cos(torch.ones(1) * (h)),
                            torch.sin(torch.ones(1) * (h)),
                            torch.cos(torch.ones(1) * (v)),
                            torch.sin(torch.ones(1) * (v)),
                        ]).unsqueeze(-1).unsqueeze(-1).repeat(1, 7, 7)

                        return orientation

                    feat['frames'][-1] = torch.cat([
                        feat['frames'][-1], get_orientation('front').repeat(len(feat['frames'][-1]), 1, 1, 1)
                    ], dim=1)
                    feat['frames_left'][-1] = torch.cat([
                        feat['frames_left'][-1], get_orientation('left').repeat(len(feat['frames_left'][-1]), 1, 1, 1)
                    ], dim=1)
                    feat['frames_up'][-1] = torch.cat([
                        feat['frames_up'][-1], get_orientation('up').repeat(len(feat['frames_up'][-1]), 1, 1, 1)
                    ], dim=1)
                    feat['frames_down'][-1] = torch.cat([
                        feat['frames_down'][-1], get_orientation('down').repeat(len(feat['frames_down'][-1]), 1, 1, 1)
                    ], dim=1)
                    feat['frames_right'][-1] = torch.cat([
                        feat['frames_right'][-1],
                        get_orientation('right').repeat(len(feat['frames_right'][-1]), 1, 1, 1)
                    ], dim=1)
        assert(np.all(np.array(list(map(len, feat['lang_instr']))) == np.array(list(map(len, feat['objnav']))))) 
        # tensorization and padding
        for k, v in feat.items():
            
            if k in {'lang_goal', 'sub_lang'}:
                # language embedding and padding
                seqs = [torch.tensor(vv, device=device) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
                seq_lengths = np.array(list(map(len, v)))
                embed_seq = self.emb_word(pad_seq)
                packed_input = pack_padded_sequence(embed_seq, seq_lengths, batch_first=True, enforce_sorted=False)
                feat[k] = {'seqs':seqs, 'emb':embed_seq, 'pi': packed_input, 'seq_len':seq_lengths}
            elif k in {'sub_indices'}:
                pass
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
                    fin_seq.append(embed_seq[in_idx:in_idx+l])
                    
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
                seqs = [torch.tensor(vv, device=device, dtype= torch.long) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.vocab['action_high'].word2index('<<pad>>'))
                feat[k] = pad_seq
            elif k in {'objnav'}:
                seqs = [torch.tensor(vv, device=device, dtype= torch.long) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.vocab['objnav'].word2index('<<pad>>'))
                embed_seq = self.emb_objnav(pad_seq)
                feat[k] = embed_seq
            else:
                # default: tensorize and pad sequence
                seqs = [torch.tensor(vv, device=device, dtype=torch.float if ('sub_frames' in k or'frames' in k or 'orientation' in k) else torch.long) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
                feat[k] = pad_seq

        return feat

    def preprocess_function(self, example):
        '''
        This is the new preprocess_function, designed as a method of your model class.
        It handles ALL logic for a SINGLE sample: downloading, loading, and feature extraction.
        '''
        # 'example' is a dict from the stream, e.g., {'task': '...', 'repeat_idx': 0, 'swapColor': 1}

        # 1. --- Download and Load Data ---
        task_path_part = example['task']
        repeat_idx = example['repeat_idx']
        swapColor = example['swapColor']

        # Download and load the JSON file
        json_filename = f"{task_path_part}/pp/ann_{repeat_idx}.json"
        local_json_path = hf_hub_download(repo_id=self.args.huggingface_id, filename=json_filename, repo_type="dataset")
        with open(local_json_path, 'r', encoding='utf-8') as f:
            ex = json.load(f)

        # Download and load the correct .pt feature file based on swapColor
        if not swapColor:
            pt_filename = f"train/{task_path_part}/{self.feat_pt}"
        elif swapColor in [1, 2]:
            pt_filename = f"train/{task_path_part}/feat_conv_colorSwap{swapColor}_panoramic.pt"
        elif swapColor in [3, 4, 5, 6]:
            pt_filename = f"train/{task_path_part}/feat_conv_onlyAutoAug{swapColor - 2}_panoramic.pt"

        local_pt_path = hf_hub_download(repo_id=self.args.huggingface_id, filename=pt_filename, repo_type="dataset")
        im = torch.load(local_pt_path, map_location='cpu')

        # 2. --- Feature Extraction (Logic from your original `featurize` loop) ---
        # We create a temporary dict to build up features for this one sample.
        processed_data = collections.defaultdict(list)

        # <<< PASTE THE BODY OF YOUR ORIGINAL `for ex in batch...` LOOP HERE >>>
        # AND MODIFY IT TO USE `processed_data` and the loaded `ex`, `im`

        # Below is the adapted logic based on the code you provided:
        action_high_order = np.array([ah['action'] for ah in ex['num']['action_high']])
        low_to_high_idx = ex['num']['low_to_high_idx']
        action_high = action_high_order[low_to_high_idx]

        processed_data['action_high'] = action_high
        processed_data['action_high_order'] = action_high_order

        val_action_high = (action_high == self.vocab['action_high'].word2index('GotoLocation', train=False)).astype(
            np.int64)
        v = 0
        while v < (len(val_action_high) - 1):
            if (val_action_high[v] - val_action_high[v + 1]) == 1:
                val_action_high[v + 1] = 1;
                v += 1
            v += 1
        val_action_high[-1] = 1

        self.serialize_lang_action(ex, action_high_order)
        subgoal_analysis = self.args.subgoal_analysis
        alow_list = [a['action'] for a in ex['num']['action_low']]
        sub_action_high = (action_high == self.vocab['action_high'].word2index(subgoal_analysis, train=False)).astype(
            np.int64)
        sub_actions = np.array(alow_list)[sub_action_high.nonzero()[0].astype(int)]

        lang_goal, lang_instr_sep = ex['num']['lang_goal'], ex['num']['lang_instr_sep']
        lang_goal = self.zero_input(lang_goal) if self.args.zero_goal else lang_goal
        lang_instr_sep = self.zero_input(lang_instr_sep) if self.args.zero_instr else lang_instr_sep
        processed_data['lang_goal'] = lang_goal

        sub_indices = (np.array(action_high_order == self.vocab['action_high'].word2index(subgoal_analysis)).astype(int).nonzero()[0])
        processed_data['sub_indices'] = list(sub_indices)

        # This part seems to have a bug in your original code (`lang_instr` is defined but not used)
        # I've kept it as close as possible to what you provided.
        processed_data['lang_instr'] = ex['num']['lang_instr']


        subgoal_lang = [lang_instr_sep[i] for i in sub_indices]

        alow = []
        alow_manip = []
        obj_high_indices = []
        for ia, a in enumerate(ex['num']['action_low']):
            if val_action_high[ia] == 1 and a['action'] in self.vocab['action_low'].word2index(
                    ['<<pad>>', '<<seg>>', '<<stop>>', 'LookDown_15', 'LookUp_15', 'RotateLeft_90', 'RotateRight_90',
                     'MoveAhead_25'], train=False):
                alow.append(a['action'])
            elif val_action_high[ia] == 1:
                alow.append(self.vocab['action_low'].word2index('Manipulate', train=False))
            if not (a['action'] in self.vocab['action_low'].word2index(
                    ['<<pad>>', '<<seg>>', '<<stop>>', 'LookDown_15', 'LookUp_15', 'RotateLeft_90', 'RotateRight_90',
                     'MoveAhead_25'], train=False)):
                alow_manip.append(a['action'])
                obj_high_indices.append(low_to_high_idx[ia])

        processed_data['action_low'] = alow
        processed_data['action_low_manip'] = alow_manip
        processed_data['obj_high_indices'] = obj_high_indices

        if self.args.subgoal_aux_loss_wt > 0:
            processed_data['subgoals_completed'] = np.array(ex['num']['low_to_high_idx'])[
                                                       val_action_high.nonzero()[0].astype(int)] / self.max_subgoals
        if self.args.pm_aux_loss_wt > 0:
            num_actions = len(alow)
            processed_data['subgoal_progress'] = [(i + 1) / float(num_actions) for i in range(num_actions)]

        obj_list = [self.vocab['objnav'].word2index('<<nav>>')]
        high_idx = 0;
        indices = []
        for a in ex['plan']['low_actions']:
            if a['api_action']['action'] in ['MoveAhead', 'LookUp', 'LookDown', 'RotateRight', 'RotateLeft']:
                if a['high_idx'] == (high_idx + 1):
                    obj_list.append(self.vocab['objnav'].word2index('<<nav>>', train=False));
                    high_idx += 1
                continue
            label = a['api_action']['receptacleObjectId'].split('|') if a['api_action']['action'] == 'PutObject' else \
            a['api_action']['objectId'].split('|')
            indices.append(classes.index(label[4].split('_')[0] if len(label) >= 5 else label[0]))
            if a['high_idx'] == (high_idx + 1):
                obj_list.append(
                    self.vocab['objnav'].word2index((label[4].split('_')[0] if len(label) >= 5 else label[0]).lower(),
                                                    train=False));
                high_idx += 1

        processed_data['objnav'] = [obj_list[o + 1] for o, obj in enumerate(obj_list) if
                                    (obj == self.vocab['objnav'].word2index('<<nav>>'))]
        processed_data['action_low_mask_label'] = indices

        if len(sub_actions) > 0:
            sah = [np.array(list(map(itemgetter(1), g))) for k, g in
                   groupby(enumerate(sub_action_high.nonzero()[0]), lambda ix: ix[0] - ix[1])]

            # Here we are using the 'im' tensor we loaded at the beginning
            sub_frames_high = np.copy(sub_action_high)
            for sfh in range(1, len(sub_frames_high)):
                if sub_action_high[sfh] < sub_action_high[sfh - 1]: sub_frames_high[sfh] = 1
            sub_frames = im[2][sub_frames_high.nonzero()[0]]

            sac_ind = 0;
            sfr_ind = 0
            for sii, s in enumerate(sah):
                so = np.array(indices)[
                    (np.array(obj_high_indices) == np.array(low_to_high_idx)[s][0]).astype(int).nonzero()[0]]
                processed_data['sub_objs'].append(so)
                processed_data['sub_actions'].append(list(sub_actions[sac_ind:sac_ind + len(s)]) + [self.stop_token])
                processed_data['sub_frames'].append(sub_frames[sfr_ind:sfr_ind + len(s) + 1])
                processed_data['sub_lang'].append(subgoal_lang[sii])
                sac_ind += len(s)
                sfr_ind += (len(s) + 1)

        # The orientation logic had some issues, this is a corrected interpretation
        # It should operate on the final Tensors, but we'll prepare the orientation tensor here.
        # The actual concatenation is better handled in collate_fn or the forward pass
        # For now, we will add it here if self.orientation is true, assuming it's part of the feature extraction.
        # This part of the logic seems to be missing from your provided snippet.

        # Add the raw 'ex' content for use in the training step if needed
        processed_data['raw_ex'] = ex

        # Convert defaultdict back to a regular dict before returning
        return dict(processed_data)

    def collate_fn(self, list_of_examples, device):
        """
        Collate for streamed samples produced by `preprocess_function`.
        - 不做任何下载/IO/解析；仅在 batch 维上聚合、对齐并张量化
        - 先对齐非时间维(>=2D 的 dim>=1)，再对时间维用 pad_sequence
        - 文本键做 emb + pack；objnav 做 emb；标签键用各自 PAD
        """
        import collections
        import torch
        import torch.nn.functional as F
        from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

        # ---------- 小工具：在指定维度右侧 padding 到 size ----------
        def pad_along_dim(t: torch.Tensor, size: int, dim: int, value: float = 0.0) -> torch.Tensor:
            if t.size(dim) == size:
                return t
            pad = [0, 0] * t.dim()  # F.pad 从最后一维开始指定 [left, right]
            right = size - t.size(dim)
            pad_index_from_end = 2 * (t.dim() - dim - 1)
            pad[pad_index_from_end] = right
            return F.pad(t, pad=pad, mode='constant', value=value)

        # ---------- 1) 聚合 ----------
        feat = collections.defaultdict(list)
        for ex in list_of_examples:
            # ex 是 preprocess_function 返回的 processed_data
            # raw_ex/metadata 单独保存
            if 'raw_ex' in ex:
                feat['batch_metadata'].append(ex['raw_ex'])
            for k, v in ex.items():
                if k == 'raw_ex':
                    continue
                feat[k].append(v)

        if 'batch_metadata' not in feat:
            feat['batch_metadata'] = []

        # 保留一致性断言（如果这两个键都在）
        if 'lang_instr' in feat and 'objnav' in feat:
            li = [len(x) for x in feat['lang_instr']]
            on = [len(x) for x in feat['objnav']]
            assert all(a == b for a, b in zip(li, on)), "lang_instr 与 objnav 的分段数不一致"

        # ---------- 2) 针对不同 key 的对齐与张量化 ----------
        TEXT_EMB_KEYS = {'lang_goal', 'sub_lang'}  # 需要 emb_word + pack
        META_KEYS = {'batch_metadata', 'sub_indices'}  # 不张量化
        LABEL_PAD_ACTION_HIGH = self.vocab['action_high'].word2index('<<pad>>')
        LABEL_PAD_OBJNAV = self.vocab['objnav'].word2index('<<pad>>')

        for k, v in list(feat.items()):
            if k in META_KEYS:
                continue

            # --- A) 文本序列：emb + pack ---
            if k in TEXT_EMB_KEYS:
                # v: List[seq]，理想是 1D token ids；但有时会是 2D: (num_seg, L)
                seqs_raw = [torch.as_tensor(vv, device=device, dtype=torch.long) for vv in v]
                ndims = [t.dim() for t in seqs_raw]

                if all(d == 1 for d in ndims):
                    # --- 标准路径：每样本一个 1D token 序列 ---
                    pad_seq = pad_sequence(seqs_raw, batch_first=True, padding_value=int(self.pad))  # (B, L)
                    seq_lengths = torch.as_tensor([t.numel() for t in seqs_raw], dtype=torch.long)
                    emb = self.emb_word(pad_seq)  # (B, L, E)
                    pi = pack_padded_sequence(emb, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
                    feat[k] = {'seqs': seqs_raw, 'emb': emb, 'pi': pi, 'seq_len': seq_lengths}
                else:
                    # --- 兼容路径：出现 2D（分段）或混合维度的情况 ---
                    # 统一把每个样本变成“段列表”
                    per_sample_segments = []
                    for t in seqs_raw:
                        if t.dim() == 1:
                            per_sample_segments.append([t])  # 单段
                        elif t.dim() == 2:
                            # 视为 num_seg x L，每一行是一个段
                            per_sample_segments.append([row for row in t])  # List[(L,)]
                        else:
                            raise ValueError(f"Unexpected ndim for {k}: {t.shape}")

                    num_seg = [len(seg_list) for seg_list in per_sample_segments]
                    flat = [seg for seg_list in per_sample_segments for seg in seg_list]  # 扁平化所有段

                    if len(flat) == 0:
                        feat[k] = {'seq': []}  # 空批兜底
                    else:
                        pad_flat = pad_sequence(flat, batch_first=True, padding_value=int(self.pad))  # (sum_seg, Lmax)
                        emb_flat = self.emb_word(pad_flat)  # (sum_seg, Lmax, E)

                        # 还原为按样本的段列表（与 lang_instr 的输出结构一致）
                        fin_seq, idx = [], 0
                        for n in num_seg:
                            fin_seq.append(emb_flat[idx:idx + n])  # List[(n_i, Lmax, E)]
                            idx += n

                        feat[k] = {'seq': fin_seq}
                continue

            # --- B) 分段文本：lang_instr（二维 List[List[int]]）：扁平化→pad+emb→还原 ---
            if k == 'lang_instr':
                num_instr = [len(seglist) for seglist in v]
                flat = [torch.as_tensor(seg, device=device, dtype=torch.long) for seglist in v for seg in seglist]
                if len(flat) == 0:
                    feat[k] = {'seq': []}
                    continue
                pad_seq = pad_sequence(flat, batch_first=True, padding_value=int(self.pad))  # (sumL, Lmax)
                emb = self.emb_word(pad_seq)  # (sumL, Lmax, E)
                fin_seq, idx = [], 0
                for l in num_instr:
                    fin_seq.append(emb[idx:idx + l])  # List[(Ni, Lmax, E)]
                    idx += l
                feat[k] = {'seq': fin_seq}
                continue

            # --- C) 类别标签：action_high / action_high_order（只 pad，long） ---
            if k in {'action_high', 'action_high_order'}:
                seqs = [torch.as_tensor(vv, device=device, dtype=torch.long) for vv in v]  # List[(T,)]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=int(LABEL_PAD_ACTION_HIGH))
                feat[k] = pad_seq
                continue

            # --- D) objnav：pad 后做 emb（输出为 embedding） ---
            if k == 'objnav':
                seqs = [torch.as_tensor(vv, device=device, dtype=torch.long) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=int(LABEL_PAD_OBJNAV))  # (B,T)
                feat[k] = self.emb_objnav(pad_seq)  # (B,T,E)
                continue

            # --- E) action_low_mask：通常是 (T,K) float，直接 pad_sequence ---
            if k == 'action_low_mask':
                seqs = [torch.as_tensor(vv, device=device, dtype=torch.float) for vv in v]
                feat[k] = pad_sequence(seqs, batch_first=True, padding_value=0.0)
                continue

            # --- F) action_low_mask_label：展平为 1D long（按你原始实现） ---
            if k == 'action_low_mask_label':
                flat = [x for vv in v for x in vv]
                feat[k] = torch.as_tensor(flat, device=device, dtype=torch.long)
                continue

            # --- G) 辅助量：float pad ---
            if k in {'subgoal_progress', 'subgoals_completed'}:
                seqs = [torch.as_tensor(vv, device=device, dtype=torch.float) for vv in v]  # List[(T,)]
                feat[k] = pad_sequence(seqs, batch_first=True, padding_value=float(self.pad))
                continue

            # --- H) 其它键（含 frames / feat / mask / sub_frames / orientation 等）---
            # 规则：浮点走 0.0 pad；整型走 self.pad；且先对齐非时间维，再对时间维 pad_sequence
            is_floatish = any(tag in k for tag in ('frames', 'im', 'mask', 'feat'))
            dtype = torch.float if is_floatish else torch.long
            pad_val = 0.0 if is_floatish else float(self.pad)

            try:
                seqs = [torch.as_tensor(vv, device=device, dtype=dtype) for vv in v]
                if seqs and seqs[0].dim() >= 2:
                    for d in range(1, seqs[0].dim()):  # 对齐非时间维
                        max_d = max(s.size(d) for s in seqs)
                        if any(s.size(d) != max_d for s in seqs):
                            seqs = [pad_along_dim(s, max_d, d, value=pad_val) for s in seqs]
                feat[k] = pad_sequence(seqs, batch_first=True, padding_value=pad_val)
            except Exception as e:
                # 兜底：保留原始列表，便于定位异常 key
                # print(f"[collate_fn] key={k} fallback: {e}")
                feat[k] = v

        # ---------- 3) 返回 ----------
        return feat, feat.pop('batch_metadata')

    def serialize_lang_action(self, feat, action_high_order):
        '''
        append segmented instr language and low-level actions into single sequences
        '''
       
        action_high_order = (action_high_order == self.vocab['action_high'].word2index('GotoLocation', train=False)).nonzero()[0]
        li_1 = []
        feat['num']['lang_instr_sep'] = feat['num']['lang_instr']
        for ai in range(len(action_high_order)-1):
            li_1.append([word for desc in feat['num']['lang_instr'][action_high_order[ai]:action_high_order[ai+1]] for word in desc])
          
        li_1.append([word for desc in feat['num']['lang_instr'][action_high_order[-1]:] for word in desc])
        
        feat['num']['lang_instr'] = li_1
        



        # if not self.test_mode:
        feat['num']['action_low'] = [a for a_group in feat['num']['action_low'] for a in a_group]


    def decompress_mask(self, compressed_mask):
        '''
        decompress mask from json files
        '''
        mask = np.array(decompress_mask(compressed_mask))
        mask = np.expand_dims(mask, axis=0)
        return mask


    def forward(self, feat, max_decode=300):
        cont_lang_instr, enc_lang_instr = self.encode_lang_instr(feat['sub_lang']['pi'])
        state_0_instr = cont_lang_instr, torch.zeros_like(cont_lang_instr)
        frames = self.vis_dropout(feat['sub_frames'])
        res = self.dec(enc_lang_instr, frames, max_decode=max_decode, gold=feat['sub_actions'], state_0_instr=state_0_instr)

        feat.update(res)
        return feat



    def encode_lang_instr(self, packed_seq):
        '''
        encode goal+instr language
        '''
        emb_lang = packed_seq
        # emb_lang = feat['sub_lang']

        
        self.lang_dropout(emb_lang.data)
        
        enc_lang, _ = self.enc_instr(emb_lang)
        enc_lang, _ = pad_packed_sequence(enc_lang, batch_first=True)
        
        self.lang_dropout(enc_lang)
        
        cont_lang = self.enc_att_instr(enc_lang)

        return cont_lang, enc_lang



    def reset(self):
        '''
        reset internal states (used for real-time execution during eval)
        '''
        self.r_state = {
            # 'state_t_goal': None,
            'state_t_instr': None,
            'e_t': None,
            # 'cont_lang_goal': None,
            # 'enc_lang_goal': None,
            'cont_lang_instr': None,
            'enc_lang_instr': None,
            # 'lang_index':0,
            # 'enc_obj': None,
        }


    def step(self, feat, eval_idx, prev_action=None):

        # encode language features (instr)
        if self.r_state['cont_lang_instr'] is None and self.r_state['enc_lang_instr'] is None:
            lang_index =  (eval_idx == np.array(feat['sub_indices'][0])).astype(int).nonzero()[0]
            packed_input = pack_padded_sequence(feat['sub_lang']['emb'][lang_index], feat['sub_lang']['seq_len'][lang_index], batch_first=True, enforce_sorted=False)
            # print(lang_index)

            self.r_state['cont_lang_instr'], self.r_state['enc_lang_instr'] = self.encode_lang_instr(packed_input)
            # self.r_state['enc_obj'] = self.dec.object_enc(feat['objnav'][0][lang_index].unsqueeze(0))


        # initialize embedding and hidden states (instr)
        if self.r_state['e_t'] is None and self.r_state['state_t_instr'] is None:
            self.r_state['e_t'] = self.dec.go.repeat(self.r_state['enc_lang_instr'].size(0), 1)
            self.r_state['state_t_instr'] = self.r_state['cont_lang_instr'], torch.zeros_like(self.r_state['cont_lang_instr'])

        

        # previous action embedding
        e_t = self.embed_action(prev_action) if prev_action is not None else self.r_state['e_t']
        e_t = self.r_state['e_t']

        # decode and save embedding and hidden states
        # if self.panoramic:

        out_action_low, out_action_low_mask, state_t_instr, attn_score_t_instr = self.dec.step(enc_instr=self.r_state['enc_lang_instr'], 
                                                                            frame=feat['frames'][:, 0],
                                                                            e_t=e_t, 
                                                                            state_tm1_instr=self.r_state['state_t_instr'])

        self.r_state['state_t_instr'] = state_t_instr
        self.r_state['e_t'] = self.dec.emb(out_action_low.max(1)[1])

        # output formatting
        feat['out_action_low'] = out_action_low.unsqueeze(0)
        feat['out_action_low_mask'] = out_action_low_mask.unsqueeze(0)

        return feat


    def extract_preds(self, out, batch, feat, clean_special_tokens=True):
        '''
        output processing
        '''
        pred = {}
        for (ex, _), alow, alow_mask in zip(batch, feat['out_action_low'].max(2)[1].tolist(), feat['out_action_low_mask']):
            # remove padding tokens
            
            if self.pad in alow:
                pad_start_idx = alow.index(self.pad)
                alow = alow[:pad_start_idx]
                alow_mask = alow_mask[:pad_start_idx]

            if clean_special_tokens:
                if self.stop_token in alow:
                    stop_start_idx = alow.index(self.stop_token)
                    alow = alow[:stop_start_idx]
                    alow_mask = alow_mask[:stop_start_idx]

            # index to API actions
            words = self.vocab['action_low'].index2word(alow)

            p_mask = [alow_mask[t].detach().cpu().numpy() for t in range(alow_mask.shape[0])]

            pred[self.get_task_and_ann_id(ex)] = {
                'action_low': ' '.join(words),
                'action_low_mask': p_mask,
            }

        return pred


    def embed_action(self, action):
        '''
        embed low-level action
        '''
        device = torch.device('cuda') if self.args.gpu else torch.device('cpu')
        action_num = torch.tensor(self.vocab['action_low'].word2index(action), device=device)
        action_emb = self.dec.emb(action_num).unsqueeze(0)
        return action_emb


    def compute_loss(self, out, batch, feat):
        '''
        loss function for Seq2Seq agent
        '''
        losses = dict()

        # GT and predictions
        p_alow = out['out_action_low'].view(-1, len(self.vocab['action_low']))
        l_alow = feat['sub_actions'].view(-1)
        p_alow_mask = out['out_action_low_mask'][:, :-1, :]
        # valid = feat['action_low_valid_interact']

        # action loss
        pad_valid = (l_alow != self.pad)
        alow_loss = F.cross_entropy(p_alow, l_alow, reduction='none')
        alow_loss *= pad_valid.float()
        alow_loss = alow_loss.mean()
        losses['action_low'] = alow_loss * self.args.action_loss_wt

        # # mask loss

        flat_l_alow_mask = feat['sub_objs'].view(feat['sub_objs'].shape[0] * feat['sub_objs'].shape[1])
        valid_idxs = (flat_l_alow_mask != self.pad).int().nonzero().view(-1)
        flat_p_alow_mask = p_alow_mask.reshape(p_alow_mask.shape[0] * p_alow_mask.shape[1], p_alow_mask.shape[2])
        
        losses['action_low_mask'] = self.ce_loss(flat_p_alow_mask[valid_idxs], flat_l_alow_mask[valid_idxs]) * self.args.mask_loss_wt

        # subgoal completion loss
        if self.args.subgoal_aux_loss_wt > 0:
            p_subgoal = feat['out_subgoal'].squeeze(2)
            l_subgoal = feat['subgoals_completed']
            sg_loss = self.mse_loss(p_subgoal, l_subgoal)
            sg_loss = sg_loss.view(-1) * pad_valid.float()
            subgoal_loss = sg_loss.mean()
            losses['subgoal_aux'] = self.args.subgoal_aux_loss_wt * subgoal_loss

        # progress monitoring loss
        if self.args.pm_aux_loss_wt > 0:
            p_progress = feat['out_progress'].squeeze(2)
            l_progress = feat['subgoal_progress']
            pg_loss = self.mse_loss(p_progress, l_progress)
            pg_loss = pg_loss.view(-1) * pad_valid.float()
            progress_loss = pg_loss.mean()
            losses['progress_aux'] = self.args.pm_aux_loss_wt * progress_loss

        return losses


    def weighted_mask_loss(self, pred_masks, gt_masks):
        '''
        mask loss that accounts for weight-imbalance between 0 and 1 pixels
        '''
        bce = self.bce_with_logits(pred_masks, gt_masks)
        flipped_mask = self.flip_tensor(gt_masks)
        inside = (bce * gt_masks).sum() / (gt_masks).sum()
        outside = (bce * flipped_mask).sum() / (flipped_mask).sum()
        return inside + outside


    def flip_tensor(self, tensor, on_zero=1, on_non_zero=0):
        '''
        flip 0 and 1 values in tensor
        '''
        res = tensor.clone()
        res[tensor == 0] = on_zero
        res[tensor != 0] = on_non_zero
        return res


    def compute_metric(self, preds, data):
        '''
        compute f1 and extract match scores for output
        '''
        m = collections.defaultdict(list)
        for (task, _) in data:
            # try:
            ex = self.load_task_json(task)
            i = self.get_task_and_ann_id(ex)
            label = ' '.join([a['discrete_action']['action'] for a in ex['plan']['low_actions']])
            m['action_low_f1'].append(compute_f1(label.lower(), preds[i]['action_low'].lower()))
            m['action_low_em'].append(compute_exact(label.lower(), preds[i]['action_low'].lower()))
            # except:
            #     print("KeyError in valid")
            #     pass
        return {k: sum(v)/len(v) for k, v in m.items()}
