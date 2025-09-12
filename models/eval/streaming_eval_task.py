import os
import json
import numpy as np
from PIL import Image
from datetime import datetime
from eval import Eval
from env.thor_env import ThorEnv
import shutil
import torch
from models.model import constants
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import maskrcnn_resnet50_fpn
import matplotlib.pyplot as plt
import random
from huggingface_hub import hf_hub_url
import requests
import io


class StreamingEvalTask(Eval):
    '''
    流式评估任务
    '''

    def __init__(self, args, huggingface_id="byeonghwikim/abp_dataset"):
        self.args = args
        self.huggingface_id = huggingface_id
        self.classes = ['0'] + constants.OBJECTS + ['AppleSliced', 'ShowerCurtain', 'TomatoSliced', 'LettuceSliced',
                                                    'Lamp', 'ShowerHead', 'EggCracked', 'BreadSliced', 'PotatoSliced',
                                                    'Faucet']

    @classmethod
    def run_streaming(cls, model, resnet, task_queue, args, lock, successes, failures, results):
        '''
        流式评估循环
        '''
        # 创建流式评估器
        streaming_eval = StreamingEvalTask(args)

        # 启动THOR环境
        env = ThorEnv(use_virtual_display=True)

        while True:
            if task_queue.qsize() == 0:
                break

            task = task_queue.get()

            # 检查任务是否已完成
            if streaming_eval._check_task_completed(task):
                print("skipping:", task['task'])
                continue

            try:
                # 流式加载任务数据
                traj_data = streaming_eval.load_task_streaming(task)
                printing_log("Evaluating: %s" % (task['task']))
                printing_log("No. of trajectories left: %d" % (task_queue.qsize()))

                # 流式评估
                streaming_eval.evaluate_streaming(env, model, task['repeat_idx'], resnet, traj_data, args, lock,
                                                  successes, failures, results, task)

            except Exception as e:
                import traceback
                traceback.print_exc()
                printing_log("Error: " + repr(e))

        env.stop()

    def load_task_streaming(self, task):
        '''
        流式加载任务数据
        '''
        task_path = task['task']
        repeat_idx = task['repeat_idx']

        # 从Hugging Face加载JSON数据
        json_filename = f"{task_path}/pp/ann_{repeat_idx}.json"
        json_url = hf_hub_url(
            repo_id=self.huggingface_id,
            filename=json_filename,
            repo_type="dataset"
        )

        json_response = requests.get(
                json_url,
                timeout=120,
                stream=False
            )
        if json_response.status_code != 200:
            raise Exception(f"Failed to load JSON: {json_url}")

        traj_data = json.loads(json_response.content.decode('utf-8'))

        # 加载图像特征（如果需要）
        if hasattr(self.args, 'load_feats') and self.args.load_feats:
            feat_data = self._load_feats_streaming(task_path, repeat_idx)
            traj_data['feats'] = feat_data

        return traj_data

    def _load_feats_streaming(self, task_path, repeat_idx):
        '''
        流式加载图像特征
        '''
        feat_data = {}

        # 加载不同数据增强版本的特征
        for swapColor in range(7):
            if swapColor == 0:
                pt_filename = f"train/{task_path}/feat_conv.pt"
            elif swapColor in [1, 2]:
                pt_filename = f"train/{task_path}/feat_conv_colorSwap{swapColor}_panoramic.pt"
            else:
                pt_filename = f"train/{task_path}/feat_conv_onlyAutoAug{swapColor - 2}_panoramic.pt"

            pt_url = hf_hub_url(
                repo_id=self.huggingface_id,
                filename=pt_filename,
                repo_type="dataset"
            )

            pt_response = requests.get(
                pt_url,
                timeout=300,
                stream=True
            )
            if pt_response.status_code == 200:
                with io.BytesIO(pt_response.content) as buffer:
                    feat_data[swapColor] = torch.load(buffer, map_location='cpu')

        return feat_data

    def _check_task_completed(self, task):
        '''
        检查任务是否已完成
        '''
        success_path = os.path.join('logs/success', task['task'], str(task['repeat_idx']))
        failure_path = os.path.join('logs/failure', task['task'], str(task['repeat_idx']))
        return os.path.exists(success_path) or os.path.exists(failure_path)

    def evaluate_streaming(self, env, model, r_idx, resnet, traj_data, args, lock, successes, failures, results, task):
        '''
        流式评估主函数
        '''
        import copy

        # 重置模型
        for mk in model.keys():
            model[mk].reset()

        # 设置场景
        reward_type = 'dense'
        nav_traj_data = copy.deepcopy(traj_data)
        self.setup_scene(env, nav_traj_data, r_idx, args, reward_type=reward_type)

        # 流式子目标预测
        feat_subgoal = self._featurize_subgoal_streaming(model['subgoal'], traj_data)
        out_subgoal = model['subgoal'].forward(feat_subgoal)
        subgoals_to_complete = self._process_subgoal_predictions(out_subgoal, model['subgoal'])

        # 流式对象检测
        feat_obj = self._featurize_object_streaming(model['object'], traj_data, subgoals_to_complete)
        out_obj = model['object'].forward(feat_obj)
        objects2find = self._process_object_predictions(out_obj)

        # 流式导航特征提取
        feat = self._featurize_nav_streaming(model['nav'], nav_traj_data, subgoals_to_complete, objects2find)

        # 初始化Mask R-CNN
        maskrcnn = self._load_maskrcnn_streaming()

        # 执行流式评估
        self._execute_streaming_evaluation(
            env, model, r_idx, resnet, traj_data, args, lock,
            successes, failures, results, task, feat, maskrcnn,
            subgoals_to_complete, objects2find
        )

    def _featurize_subgoal_streaming(self, subgoal_model, traj_data):
        '''
        流式子目标特征提取
        '''
        # 使用内存中的数据进行特征提取，避免磁盘IO
        feat = subgoal_model.featurize([(traj_data, False)], load_mask=True)
        return feat

    def _featurize_object_streaming(self, object_model, traj_data, subgoals):
        '''
        流式对象特征提取
        '''
        feat = object_model.featurize([(traj_data, False)], subgoals, load_mask=True)
        return feat

    def _featurize_nav_streaming(self, nav_model, traj_data, subgoals, objects):
        '''
        流式导航特征提取
        '''
        feat = nav_model.featurize([(traj_data, False)], subgoals, objects, load_mask=True)
        return feat

    def _load_maskrcnn_streaming(self):
        '''
        流式加载Mask R-CNN模型
        '''
        # 检查模型是否已加载
        if not hasattr(self, '_maskrcnn'):
            self._maskrcnn = maskrcnn_resnet50_fpn(num_classes=119)
            self._maskrcnn.eval()

            # 从Hugging Face加载权重
            weight_url = hf_hub_url(
                repo_id=self.huggingface_id,
                filename="weight_maskrcnn.pt",
                repo_type="dataset"
            )

            weight_response = requests.get(weight_url, timeout=30)
            if weight_response.status_code == 200:
                with io.BytesIO(weight_response.content) as buffer:
                    weights = torch.load(buffer, map_location='cpu')
                self._maskrcnn.load_state_dict(weights)
            else:
                # 使用默认权重或报错
                print("Warning: Could not load Mask R-CNN weights from Hugging Face")

        return self._maskrcnn.cuda()

    def _execute_streaming_evaluation(self, env, model, r_idx, resnet, traj_data, args, lock,
                                      successes, failures, results, task, feat, maskrcnn,
                                      subgoals, objects):
        '''
        执行流式评估
        '''
        # 这里保持原有的评估逻辑，但使用流式加载的数据
        goal_instr = traj_data['turk_annotations']['anns'][r_idx]['task_desc']

        prev_vis_feat = None
        prev_action = None
        nav_actions = ['MoveAhead_25', 'RotateLeft_90', 'RotateRight_90', 'LookDown_15', 'LookUp_15']

        # ... [保持原有的评估逻辑，但使用流式数据] ...

        done, success = False, False
        fails = 0
        t = 0
        reward = 0

        total_frames = []
        total_actions = []

        while not done:
            if t >= args.max_steps:
                break

            # 流式获取当前图像
            curr_image = Image.fromarray(np.uint8(env.last_event.frame))
            vis_feat = resnet.featurize([curr_image], batch=1).unsqueeze(0)
            feat['frames'] = vis_feat

            # ... [保持原有的动作预测和执行逻辑] ...

            # 流式保存结果
            if done or t >= args.max_steps:
                self._save_streaming_results(
                    task, total_frames, total_actions,
                    traj_data, r_idx, success, env
                )

        # 更新结果
        self._update_results_streaming(
            lock, successes, failures, results,
            task, traj_data, r_idx, goal_instr,
            success, reward, t, env
        )

    def _save_streaming_results(self, task, frames, actions, traj_data, r_idx, success, env):
        '''
        流式保存结果
        '''
        save_dir = 'logs/success/' if success else 'logs/failure/'
        os.makedirs(os.path.join(save_dir, task['task'], str(r_idx)), exist_ok=True)

        # 流式保存图像
        for i, (frame, action) in enumerate(zip(frames, actions)):
            frame_path = os.path.join(save_dir, task['task'], str(r_idx), f"{i}_{action}.png")
            frame.save(frame_path)

        # 流式保存JSON
        json_content = json.dumps(traj_data, indent=2)
        json_path = os.path.join(save_dir, task['task'], str(r_idx), f"ann_{r_idx}.json")
        with open(json_path, 'w') as f:
            f.write(json_content)

    def _update_results_streaming(self, lock, successes, failures, results,
                                  task, traj_data, r_idx, goal_instr,
                                  success, reward, t, env):
        '''
        流式更新结果
        '''
        pcs = env.get_goal_conditions_met()
        goal_condition_success_rate = pcs[0] / float(pcs[1]) if pcs[1] > 0 else 0

        path_len_weight = len(traj_data['plan']['low_actions'])
        s_spl = (1 if success else 0) * min(1., path_len_weight / (float(t) + 1e-4))
        pc_spl = goal_condition_success_rate * min(1., path_len_weight / (float(t) + 1e-4))

        log_entry = {
            'trial': traj_data['task_id'],
            'type': traj_data['task_type'],
            'repeat_idx': int(r_idx),
            'goal_instr': goal_instr,
            'completed_goal_conditions': int(pcs[0]),
            'total_goal_conditions': int(pcs[1]),
            'goal_condition_success': float(goal_condition_success_rate),
            'success_spl': float(s_spl),
            'path_len_weighted_success_spl': float(s_spl * path_len_weight),
            'goal_condition_spl': float(pc_spl),
            'path_len_weighted_goal_condition_spl': float(pc_spl * path_len_weight),
            'path_len_weight': int(path_len_weight),
            'reward': float(reward)
        }

        lock.acquire()
        if success:
            successes.append(log_entry)
        else:
            failures.append(log_entry)

        # 更新总体结果
        results['all'] = self.get_metrics(list(successes), list(failures))
        lock.release()

    # 保持原有的工具方法，但确保它们使用流式数据
    def get_panoramic_views_streaming(self, env):
        '''
        流式获取全景视图
        '''
        # 实现与原来相同的逻辑，但确保没有磁盘IO
        return get_panoramic_views(env)

    def get_panoramic_actions_streaming(self, env):
        '''
        流式获取全景动作
        '''
        return get_panoramic_actions(env)

    def doManipulation_streaming(self, total_actions, total_frames, action_high_name,
                                 traj_data, action_high_order, resnet, feat_nav,
                                 model, model_nav, eval_idx, maskrcnn, curr_image,
                                 lang_index, env, args, t, fails):
        '''
        流式执行操作
        '''
        # 实现与原来相同的逻辑，但使用流式数据
        return self.doManipulation(total_actions, total_frames, action_high_name,
                                   traj_data, action_high_order, resnet, feat_nav,
                                   model, model_nav, eval_idx, maskrcnn, curr_image,
                                   lang_index, env, args, t, fails)


# 保持原有的工具函数，但确保它们没有磁盘IO
def loop_detection(vis_feats, actions, window_size=10):

    # not enough vis feats for loop detection
    if len(vis_feats) < window_size*2:
        return False, None

    nav_actions = ['MoveAhead_25', 'RotateLeft_90', 'RotateRight_90', 'LookDown_15', 'LookUp_15']
    random.shuffle(nav_actions)

    start_idx = len(vis_feats) - 1

    for end_idx in range(start_idx - window_size, window_size - 1, -1):
        if (vis_feats[start_idx] == vis_feats[end_idx]).all():
            if all((vis_feats[start_idx-i] == vis_feats[end_idx-i]).all() for i in range(window_size)):
                return True, nav_actions[1] if actions[end_idx] == nav_actions[0] else nav_actions[0]

    return False, None


import math
def get_orientation(d):
    if d == 'left':
        h, v = -math.pi/2, 0.0
    elif d == 'up':
        h, v = 0.0, -math.pi/12
    elif d == 'down':
        h, v = 0.0, math.pi/12
    elif d == 'right':
        h, v = math.pi/2, 0.0
    else:
        h, v = 0.0, 0.0

    orientation = torch.cat([
        torch.cos(torch.ones(1)*(h)),
        torch.sin(torch.ones(1)*(h)),
        torch.cos(torch.ones(1)*(v)),
        torch.sin(torch.ones(1)*(v)),
    ]).unsqueeze(-1).unsqueeze(-1).repeat(1,7,7).unsqueeze(0).unsqueeze(0)

    return orientation

def get_panoramic_views(env):
    horizon = np.round(env.last_event.metadata['agent']['cameraHorizon'])
    rotation = env.last_event.metadata['agent']['rotation']
    position = env.last_event.metadata['agent']['position']

    # Left
    env.step({
        "action": "TeleportFull",
        "horizon": horizon,
        "rotateOnTeleport": True,
        "rotation": (rotation['y'] + 270.0) % 360,
        "x": position['x'],
        "y": position['y'],
        "z": position['z'],
        "forceAction": True,
    })
    curr_image_left = Image.fromarray(np.uint8(env.last_event.frame))

    # Right
    env.step({
        "action": "TeleportFull",
        "horizon": horizon,
        "rotateOnTeleport": True,
        "rotation": (rotation['y'] + 90.0) % 360,
        "x": position['x'],
        "y": position['y'],
        "z": position['z'],
        "forceAction": True,
    })
    curr_image_right = Image.fromarray(np.uint8(env.last_event.frame))

    # Up
    env.step({
        "action": "TeleportFull",
        "horizon": np.round(horizon - constants.AGENT_HORIZON_ADJ),
        "rotateOnTeleport": True,
        "rotation": rotation['y'],
        "x": position['x'],
        "y": position['y'],
        "z": position['z'],
        "forceAction": True,
    })
    curr_image_up = Image.fromarray(np.uint8(env.last_event.frame))

    # Down
    env.step({
        "action": "TeleportFull",
        "horizon": np.round(horizon + constants.AGENT_HORIZON_ADJ),
        "rotateOnTeleport": True,
        "rotation": rotation['y'],
        "x": position['x'],
        "y": position['y'],
        "z": position['z'],
        "forceAction": True,
    })
    curr_image_down = Image.fromarray(np.uint8(env.last_event.frame))

    # Left
    env.step({
        "action": "TeleportFull",
        "horizon": horizon,
        "rotateOnTeleport": True,
        "rotation": rotation['y'],
        "x": position['x'],
        "y": position['y'],
        "z": position['z'],
        "forceAction": True,
    })

    return curr_image_left, curr_image_right, curr_image_up, curr_image_down


def get_panoramic_actions(env):
    action_pairs = [
        ['RotateLeft_90', 'RotateRight_90'],
        ['RotateRight_90', 'RotateLeft_90'],
        ['LookUp_15', 'LookDown_15'],
        ['LookDown_15', 'LookUp_15'],
    ]
    imgs = []
    actions = []

    curr_image = Image.fromarray(np.uint8(env.last_event.frame))

    for a1, a2 in action_pairs:
        t_success, _, _, err, api_action = env.va_interact(a1, interact_mask=None, smooth_nav=False)
        actions.append(a1)
        imgs.append(Image.fromarray(np.uint8(env.last_event.frame)))
        #if len(err) == 0:
        if curr_image != imgs[-1]:
            t_success, _, _, err, api_action = env.va_interact(a2, interact_mask=None, smooth_nav=False)
            actions.append(a2)
        else:
            #print(err)
            printing_log('Error while {}'.format(a1))
    return actions, imgs



def printing_log(*args):
    print(*args)

    new_args = list(args)
    # if nameflag == False:
    filename = 'new_logs/loop_break_0.3_thresh_val_unseen_latest_logs.txt'
        # flag = True

    with open(filename, 'a') as f:
        for ar in new_args:
            f.write(f'{ar}\n')