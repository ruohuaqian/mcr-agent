from huggingface_hub import HfFileSystem
import requests
import io
import torch
import json


class ALFREDStreamingDataset:
    def __init__(self, repo_id, task_list, feat_pt, args):
        self.repo_id = repo_id
        self.task_list = task_list
        self.feat_pt = feat_pt
        self.args = args
        self.fs = HfFileSystem()

    def __iter__(self):
        for task_info in self.task_list:
            task_path = task_info['task']
            repeat_idx = task_info['repeat_idx']

            for swapColor in range(7):
                try:
                    # 方法1: 使用 HfFileSystem
                    json_path = f"{self.repo_id}/dataset/{task_path}/pp/ann_{repeat_idx}.json"
                    with self.fs.open(json_path, 'r') as f:
                        ex = json.load(f)

                    # 方法2: 或者使用直接的URL构造
                    if swapColor == 0:
                        pt_filename = f"train/{task_path}/{self.feat_pt}"
                    elif swapColor in [1, 2]:
                        pt_filename = f"train/{task_path}/feat_conv_colorSwap{swapColor}_panoramic.pt"
                    else:
                        pt_filename = f"train/{task_path}/feat_conv_onlyAutoAug{swapColor - 2}_panoramic.pt"

                    pt_url = f"https://huggingface.co/datasets/{self.repo_id}/resolve/main/{pt_filename}"

                    response = requests.get(pt_url, stream=True)
                    if response.status_code == 200:
                        with io.BytesIO() as buffer:
                            for chunk in response.iter_content(chunk_size=8192):
                                buffer.write(chunk)
                            buffer.seek(0)
                            im = torch.load(buffer, map_location='cpu')

                    yield {
                        'ex': ex,
                        'im': im,
                        'task_path': task_path,
                        'repeat_idx': repeat_idx,
                        'swapColor': swapColor
                    }

                except Exception as e:
                    print(f"Error loading {task_path}, repeat {repeat_idx}, swap {swapColor}: {e}")
                    continue