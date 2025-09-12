from huggingface_hub import hf_hub_url
from huggingface_hub import login
import requests
import io
import torch
import json
import os


class ALFREDStreamingDataset:
    def __init__(self, repo_id, task_list, feat_pt, args):
        self.repo_id = repo_id
        self.task_list = task_list
        self.feat_pt = feat_pt
        self.args = args
        self.setup_hf_auth()

    def setup_hf_auth(self):
        # 方法1: 使用环境变量
        if os.environ.get('HF_TOKEN'):
            login(token=os.environ.get('HF_TOKEN'))

    def __iter__(self):

        for task_info in self.task_list:
            task_path = task_info['task']
            repeat_idx = task_info['repeat_idx']

            for swapColor in range(7):
                try:
                    # 使用正确的 hf_hub_url 函数
                    json_filename = f"{task_path}/pp/ann_{repeat_idx}.json"
                    json_url = hf_hub_url(
                        repo_id=self.repo_id,
                        filename=json_filename,
                        repo_type="dataset"
                    )

                    # 下载 JSON
                    json_response = requests.get(
                        json_url,
                        timeout=120,
                        stream=False
                        )
                    if json_response.status_code != 200:
                        continue
                    ex = json.loads(json_response.content.decode('utf-8'))

                    # 下载 PT 文件
                    if swapColor == 0:
                        pt_filename = f"train/{task_path}/{self.feat_pt}"
                    elif swapColor in [1, 2]:
                        pt_filename = f"train/{task_path}/feat_conv_colorSwap{swapColor}_panoramic.pt"
                    else:
                        pt_filename = f"train/{task_path}/feat_conv_onlyAutoAug{swapColor - 2}_panoramic.pt"

                    pt_url = hf_hub_url(
                        repo_id=self.repo_id,
                        filename=pt_filename,
                        repo_type="dataset"
                    )

                    pt_response = requests.get(
                                    pt_url,
                                    timeout=300,
                                    stream=True
                                )
                    if pt_response.status_code != 200:
                        continue

                    with io.BytesIO(pt_response.content) as buffer:
                        im = torch.load(buffer, map_location='cpu')

                    yield {
                        'ex': ex,
                        'im': im,
                        'task_path': task_path,
                        'repeat_idx': repeat_idx,
                        'swapColor': swapColor
                    }

                except Exception as e:
                    print(f"Skipping {task_path}, repeat {repeat_idx}, swap {swapColor}: {e}")
                    continue