import argparse
import torch.multiprocessing as mp
from eval_task import EvalTask
from eval_subgoals import EvalSubgoals


if __name__ == '__main__':
    # multiprocessing settings
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()

    # parser
    parser = argparse.ArgumentParser()

    # settings
    parser.add_argument('--splits', type=str, default="data/splits/oct21.json")
    parser.add_argument('--huggingface_id', help='dataset huggingface id', default='byeonghwikim/abp_dataset')
    parser.add_argument('--reward_config', default='models/config/rewards.json')
    parser.add_argument('--eval_split', type=str, choices=['train', 'valid_seen', 'valid_unseen'])
    parser.add_argument('--nav_model_path', type=str, default="exp/pretrained/pretrained.pth")
    parser.add_argument('--pickup_model_path', type=str, default="exp/pretrained/pretrained.pth")
    parser.add_argument('--put_model_path', type=str, default="exp/pretrained/pretrained.pth")
    parser.add_argument('--cool_model_path', type=str, default="exp/pretrained/pretrained.pth")
    parser.add_argument('--heat_model_path', type=str, default="exp/pretrained/pretrained.pth")
    parser.add_argument('--toggle_model_path', type=str, default="exp/pretrained/pretrained.pth")
    parser.add_argument('--slice_model_path', type=str, default="exp/pretrained/pretrained.pth")
    parser.add_argument('--clean_model_path', type=str, default="exp/pretrained/pretrained.pth")
    parser.add_argument('--object_model_path', type=str, default="exp/pretrained/pretrained.pth")
    parser.add_argument('--subgoal_model_path', type=str, default="exp/pretrained/pretrained.pth")
    # parser.add_argument('--use_streaming_eval', action='store_true', help='Use streaming evaluation')

    parser.add_argument('--nav_model', type=str, default='MasterPolicy.models.model.seq2seq_im_mask')
    parser.add_argument('--sub_model', type=str, default='Interactions.models.model.seq2seq_im_mask_sub')
    parser.add_argument('--object_model', type=str, default='OEM.models.model.seq2seq_im_mask_obj')
    parser.add_argument('--subgoal_pred_model', type=str, default='PCC.models.model.seq2seq_im_mask')
    parser.add_argument('--preprocess', dest='preprocess', action='store_true')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true')
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--use_virtual_display', dest='use_virtual_display', action='store_true')

    # eval params
    parser.add_argument('--max_steps', type=int, default=400, help='max steps before episode termination')
    parser.add_argument('--max_fails', type=int, default=10, help='max API execution failures before episode termination')

    # eval settings
    parser.add_argument('--subgoals', type=str, help="subgoals to evaluate independently, eg:all or GotoLocation,PickupObject...", default="")
    parser.add_argument('--smooth_nav', dest='smooth_nav', action='store_true', help='smooth nav actions (might be required based on training data)')
    parser.add_argument('--skip_model_unroll_with_expert', action='store_true', help='forward model with expert actions')
    parser.add_argument('--no_teacher_force_unroll_with_expert', action='store_true', help='no teacher forcing with expert')

    # debug
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--fast_epoch', dest='fast_epoch', action='store_true')

    # parse arguments
    args = parser.parse_args()
        # eval = EvalTask(args, manager)
    from models.eval.streaming_eval_task import StreamingEvalTask
    # eval mode
    if args.subgoals:
        eval = EvalSubgoals(args, manager)
    else:
    # replace the eval task
        eval = StreamingEvalTask(args, manager)
    # start threads
    eval.spawn_threads_streaming()
