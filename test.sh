#!/bin/bash

OUTDIR='/mnt/share/lbh/'
DATADIR='/mnt/datasets'

mkdir -p /run/determined/workdir/.cache/torch/hub/checkpoints
cp ${OUTDIR}/ckpts/alexnet-owt-7be5be79.pth /run/determined/workdir/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth
# python evaluate.py --output_root=${OUTDIR}/Ev-NeRF_wkdir --data.root=${OUTDIR}/event_nope_nerf/data/event_blender \
#  --group=BaRF --name=event_blender_from_scarch \
#  --model=barf --yaml=barf_blender --barf_c2f=[0.1,0.5] --resume --data.val_sub=

python evaluate.py --output_root=${OUTDIR}/Ev-NeRF_wkdir --data.root=${OUTDIR}/event_nope_nerf/data/blender \
 --group=BaRF --name=from_scarch \
 --model=barf --yaml=barf_blender --barf_c2f=[0.1,0.5] --resume --data.val_sub= --camera.from_scarch=true
