python3 -m torch.distributed.launch --nproc_per_node=8 train.py --model_type ViT-S_16 \
                 --mixup \
                 --output_dir ./exp/EXPERIMENT_NAME