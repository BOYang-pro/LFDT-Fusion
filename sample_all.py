import os

os.system('python -m torch.distributed.launch  --nproc_per_node=8 train.py --sample_selected ddp-solver++ --model_selected DFT --batch_size 4 --fusion_task MEF --strategy MEAN')

os.system('python -m torch.distributed.launch  --nproc_per_node=8 train.py --sample_selected ddp-solver++ --model_selected DFT --batch_size 4 --fusion_task VI-IR --strategy MAX')

os.system('python -m torch.distributed.launch  --nproc_per_node=8 train.py --sample_selected ddp-solver++ --model_selected DFT --batch_size 4 --fusion_task VI-NIR --strategy MEAN')

os.system('python -m torch.distributed.launch  --nproc_per_node=8 train.py --sample_selected ddp-solver++ --model_selected DFT --batch_size 4 --fusion_task PIF --strategy MAX')

os.system('python -m torch.distributed.launch  --nproc_per_node=8 train.py --sample_selected ddp-solver++ --model_selected DFT --batch_size 4 --fusion_task MFF --strategy MAX')

os.system('python -m torch.distributed.launch  --nproc_per_node=8 train.py --sample_selected ddp-solver++ --model_selected DFT --batch_size 4 --fusion_task SPECT-MRI --strategy MAX')

os.system('python -m torch.distributed.launch  --nproc_per_node=8 train.py --sample_selected ddp-solver++ --model_selected DFT --batch_size 4 --fusion_task PET-MRI --strategy MAX')