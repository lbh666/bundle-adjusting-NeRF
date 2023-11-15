#!/bin/bash

OUTDIR='/mnt/share/lbh/'
DATADIR='/mnt/datasets'

mkdir -p /run/determined/workdir/.cache/torch/hub/checkpoints
cp ${OUTDIR}/ckpts/alexnet-owt-7be5be79.pth /run/determined/workdir/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth
python train.py --output_root=${OUTDIR}/Ev-NeRF_wkdir --data.root=${OUTDIR}/event_nope_nerf/data/blender \
 --group=NeRF --name=no_downsample --visdom=false \
 --model=nerf --yaml=nerf_blender_repr --data.image_size=[800,800]
