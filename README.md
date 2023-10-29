# BigVision-PyTorch
PyTorch Implementation of Google Research "Big Vision" ViT. Lightweight implementation that replicates results from **"Better plain ViT baselines for ImageNet-1k"** - https://arxiv.org/abs/2205.01580.

Start a training job using ```train.sh```, or the following command:
```
python3 -m torch.distributed.launch --nproc_per_node=8 train.py --model_type ViT-S_16 \
   --mixup \
   --output_dir ./exp/EXPERIMENT_NAME
```

Make sure to download [ImageNet2012](https://www.image-net.org/challenges/LSVRC/2012/) and extract the non-TFDS version. Here's a [reference script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4). Set the dataset directories in ```data_utils.py```.

**Note:** There have known to be some discrepencies with weight decay in PyTorch vs. JAX/TensorFlow. If you are unable to replicate results, feel free to open an issue.
