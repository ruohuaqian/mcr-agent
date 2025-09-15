import os
import torch
import pprint
import json
from vocab import Vocab
from data.preprocess import Dataset
from importlib import import_module
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from MasterPolicy.models.utils.helper_utils import optimizer_to
from huggingface_hub import hf_hub_download


torch.backends.cudnn.enabled = False

if __name__ == '__main__':
    # parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # settings
    parser.add_argument('--seed', help='random seed', default=123, type=int)
    parser.add_argument('--huggingface_id', help='dataset huggingface id', default='byeonghwikim/abp_dataset')
    parser.add_argument('--splits', help='json file containing train/dev/test splits', default='data/splits/oct21.json')
    parser.add_argument('--pp_folder', help='folder name for preprocessed data', default='pp')
    parser.add_argument('--save_every_epoch', help='save model after every epoch (warning: consumes a lot of space)',
                        action='store_true')
    parser.add_argument('--model', help='model to use', default='seq2seq_im_mask')
    parser.add_argument('--gpu', help='use gpu', action='store_true')
    parser.add_argument('--dout', help='where to save model', default='exp/model:{model}')
    parser.add_argument('--resume', help='load a checkpoint')
    parser.add_argument('--use_templated_goals',
                        help='use templated goals instead of human-annotated goal descriptions (only available for train set)',
                        action='store_true')
    parser.add_argument('--use_streaming', help='use huggingface stream dataset', action='store_true')

    # hyper parameters
    parser.add_argument('--batch', help='batch size', default=4, type=int)
    parser.add_argument('--epoch', help='number of epochs', default=20, type=int)
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

    # Load vocab from huggingface.
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
    if args.resume:
        print("Loading: " + args.resume)
        model, optimizer = M.Module.load(args.resume)
    else:
        model = M.Module(args, vocab)
        optimizer = None

    # to gpu
    if args.gpu:
        model = model.to(torch.device('cuda'))
        if not optimizer is None:
            optimizer_to(optimizer, torch.device('cuda'))

    # 开始训练
    model.run_train_stream(splits, args)
