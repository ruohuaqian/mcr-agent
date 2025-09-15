import os
import torch
import pprint
import json
from vocab import Vocab
from importlib import import_module
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from OEM.models.utils.helper_utils import optimizer_to
from huggingface_hub import hf_hub_download

torch.backends.cudnn.enabled = False

if __name__ == '__main__':
    # parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # settings
    parser.add_argument('--seed', help='random seed', default=123, type=int)
    parser.add_argument('--huggingface_id', help='dataset huggingface id', default='byeonghwikim/abp_dataset')
    parser.add_argument('--splits', help='json file containing train/dev/test splits', default='data/splits/oct21.json')
    parser.add_argument('--preprocess', help='store preprocessed data to json files', action='store_true')
    parser.add_argument('--pp_folder', help='folder name for preprocessed data', default='pp')
    parser.add_argument('--save_every_epoch', help='save model after every epoch (warning: consumes a lot of space)',
                        action='store_true')
    parser.add_argument('--model', help='model to use', default='seq2seq_im_mask')
    parser.add_argument('--gpu', help='use gpu', action='store_true')
    parser.add_argument('--dout', help='where to save model', default='exp/model:{model}')

    # ===== 新增：微调相关参数 =====
    parser.add_argument('--resume', help='resume training from a checkpoint')  # 续训：加载模型+优化器等
    parser.add_argument('--finetune_from', help='initialize weights from a checkpoint (fine-tune, reset optimizer)')
    parser.add_argument('--freeze', help='comma-separated prefixes to freeze (e.g., "encoder,visual_backbone,embeddings")',
                        default='')
    parser.add_argument('--reset_optim', help='when used with --resume, ignore optimizer/scheduler states',
                        action='store_true')
    # ============================

    parser.add_argument('--use_templated_goals',
                        help='use templated goals instead of human-annotated goal descriptions (only available for train set)',
                        action='store_true')
    parser.add_argument('--use_streaming', help='use huggingface stream dataset', action='store_true')

    # hyper parameters
    parser.add_argument('--batch', help='batch size', default=4, type=int)
    parser.add_argument('--epoch', help='number of epochs', default=30, type=int)
    parser.add_argument('--lr', help='optimizer learning rate', default=1e-4, type=float)
    parser.add_argument('--decay_epoch', help='num epoch to adjust learning rate', default=10, type=int)
    parser.add_argument('--dhid', help='hidden layer size', default=512, type=int)
    parser.add_argument('--dframe', help='image feature vec size', default=3 * 7 * 7, type=int)
    parser.add_argument('--demb', help='language embedding size', default=100, type=int)
    parser.add_argument('--pframe', help='image pixel size (assuming square shape eg: 300x300)', default=300, type=int)
    parser.add_argument('--mask_loss_wt', help='weight of mask loss', default=1., type=float)
    parser.add_argument('--action_loss_wt', help='weight of action loss', default=1., type=float)
    parser.add_argument('--subgoal_aux_loss_wt', help='weight of subgoal completion predictor', default=0.2, type=float)
    parser.add_argument('--pm_aux_loss_wt', help='weight of progress monitor', default=0.2, type=float)

    # dropouts
    parser.add_argument('--zero_goal', help='zero out goal language', action='store_true')
    parser.add_argument('--zero_instr', help='zero out step-by-step instr language', action='store_true')
    parser.add_argument('--lang_dropout', help='dropout rate for language (goal + instr)', default=0., type=float)
    parser.add_argument('--input_dropout', help='dropout rate for concatted input feats', default=0., type=float)
    parser.add_argument('--vis_dropout', help='dropout rate for Resnet feats', default=0.3, type=float)
    parser.add_argument('--hstate_dropout', help='dropout rate for LSTM hidden states during unrolling', default=0.3,
                        type=float)
    parser.add_argument('--attn_dropout', help='dropout rate for attention', default=0., type=float)
    parser.add_argument('--actor_dropout', help='dropout rate for actor fc', default=0., type=float)

    # other settings
    parser.add_argument('--dec_teacher_forcing', help='use gpu', action='store_true')
    parser.add_argument('--temp_no_history', help='use gpu', action='store_true')
    parser.add_argument('--panoramic', help='use panoramic', action='store_true')
    parser.add_argument('--orientation', help='use orientation features', action='store_true')
    parser.add_argument('--panoramic_concat', help='use panoramic', action='store_true')

    # debugging
    parser.add_argument('--fast_epoch', help='fast epoch during debugging', action='store_true')
    parser.add_argument('--dataset_fraction', help='use fraction of the dataset for debugging (0 indicates full size)',
                        default=0, type=int)

    # args and init
    args = parser.parse_args()
    args.dout = args.dout.format(**vars(args))
    torch.manual_seed(args.seed)

    # make output dir
    pprint.pprint(args)
    if not os.path.isdir(args.dout):
        os.makedirs(args.dout)

    # load train/valid/tests splits
    with open(args.splits) as f:
        splits = json.load(f)
        pprint.pprint({k: len(v) for k, v in splits.items()})

    # preprocess and save
    vocab_path = hf_hub_download(
        repo_id="byeonghwikim/abp_dataset",
        filename="%s.vocab" % args.pp_folder,
        repo_type="dataset"
    )
    print(f"Vocab successfully downloaded to: {vocab_path}")
    vocab = torch.load(vocab_path)
    with open('objnav_cls.txt', 'r') as f:
        obj_list = f.readlines()
    vocab['objnav'] = Vocab([w.strip().lower() for w in obj_list] + ['<<nav>>', '<<pad>>'])
    vocab['action_low'].word2index(['Manipulate'], train=True)
    print("Vocab loaded successfully.")

    # load model
    M = import_module(args.model)

    model, optimizer = None, None

    # ====== 加载逻辑：区分 续训 vs 微调 vs 全新训练 ======
    if args.resume:
        print("Resuming from checkpoint (model + optimizer):", args.resume)
        model, optimizer = M.Module.load(args.resume)
        if args.reset_optim:
            print("[fine-tune mode on resume] resetting optimizer/scheduler states.")
            optimizer = None  # 重新创建优化器（由 run_train_stream 内部或后续逻辑完成）

    elif args.finetune_from:
        print("Finetune from weights:", args.finetune_from)
        # 1) 构建新模型（使用当前超参/配置）
        model = M.Module(args, vocab)

        # 2) 仅加载权重（忽略旧优化器/调度器等状态），允许部分层尺寸不匹配
        ckpt = torch.load(args.finetune_from, map_location='cpu')
        # 兼容多种 ckpt 结构
        state = ckpt.get('state_dict', ckpt.get('model', ckpt))
        # 去除可能的 'module.' 前缀（DataParallel 保存）
        cleaned = {}
        for k, v in state.items():
            if k.startswith('module.'):
                cleaned[k[len('module.'):]] = v
            else:
                cleaned[k] = v
        # 丢弃形状不匹配的键
        cur = model.state_dict()
        drop = [k for k, v in cleaned.items() if k in cur and cur[k].shape != v.shape]
        for k in drop:
            cleaned.pop(k)
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        print(f"[finetune] loaded keys: {len(cleaned)} | dropped (mismatch): {len(drop)} | "
              f"missing: {len(missing)} | unexpected: {len(unexpected)}")

        # 3) 可选：冻结部分模块（按前缀匹配）
        if args.freeze.strip():
            prefixes = [p.strip() for p in args.freeze.split(',') if p.strip()]
            n_all, n_frozen = 0, 0
            for name, p in model.named_parameters():
                n_all += 1
                if any(name.startswith(pref) for pref in prefixes):
                    p.requires_grad = False
                    n_frozen += 1
            print(f"[finetune] frozen params: {n_frozen}/{n_all} (prefixes={prefixes})")

        optimizer = None  # 微调：重置优化器/调度器

    else:
        print("Training from scratch.")
        model = M.Module(args, vocab)
        optimizer = None
    # =====================================================

    # to gpu
    if args.gpu:
        device = torch.device('cuda')
        model = model.to(device)
        if optimizer is not None:
            optimizer_to(optimizer, device)

    # start train loop
    # - 若 optimizer=None，由模型内部/训练函数创建新的优化器（推荐微调时如此）
    # - 若使用 --resume 且未 --reset_optim，则沿用旧优化器继续训练
    model.run_train_stream(splits, args, optimizer=optimizer)
