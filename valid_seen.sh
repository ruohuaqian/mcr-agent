export PYTHONPATH="${MCR_ROOT}:${MCR_ROOT}/models:${PYTHONPATH}"

cd "${MCR_ROOT}/models"
python eval/eval_seq2seq.py   \
		--nav_model_path /content/drive/MyDrive/mcr-agent/exp/MasterPolicy/latest.pth  \
		--pickup_model_path /content/drive/MyDrive/mcr-agent/exp/Interactions/PickupObject/latest.pth  \
		--put_model_path /content/drive/MyDrive/mcr-agent/exp/Interactions/PutObject/latest.pth  \
		--heat_model_path /content/drive/MyDrive/mcr-agent/exp/Interactions/HeatObject/latest.pth  \
		--cool_model_path /content/drive/MyDrive/mcr-agent/exp/Interactions/CoolObject/latest.pth  \
		--clean_model_path /content/drive/MyDrive/mcr-agent/exp/Interactions/CleanObject/latest.pth  \
		--toggle_model_path /content/drive/MyDrive/mcr-agent/exp/Interactions/ToggleObject/latest.pth  \
		--slice_model_path /content/drive/MyDrive/mcr-agent/exp/Interactions/SliceObject/latest.pth  \
		--object_model_path /content/drive/MyDrive/mcr-agent/exp/OEM/latest.pth  \
		--subgoal_model_path /content/drive/MyDrive/mcr-agent/exp/PCC/latest.pth  \
		--splits /content/drive/MyDrive/mcr-agent/splits/rest_train.json \
		--eval_split	valid_seen                          \
		--huggingface_id		byeonghwikim/abp_dataset       \
		--gpu                                               \
		--max_step	80                                 \
		--max_fail	10                                  \
		--reward_config ${MCR_ROOT}/models/config/rewards.json \
		--num_threads 4;