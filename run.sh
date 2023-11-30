#!/bin/bash

OUTDIR='/mnt/share/lbh/'
DATADIR='/mnt/datasets'

mkdir -p /run/determined/workdir/.cache/torch/hub/checkpoints
cp ${OUTDIR}/ckpts/alexnet-owt-7be5be79.pth /run/determined/workdir/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth
python train.py --output_root=${OUTDIR}/Ev-NeRF_wkdir --data.root=${OUTDIR}/event_nope_nerf/data/event_blender \
 --group=BaRF --name=event_blender_from_scarch_sparse --visdom=false \
 --model=barf --yaml=barf_blender --barf_c2f=[0.1,0.5] --camera.from_scarch=true --data.scene=sparse_lego

