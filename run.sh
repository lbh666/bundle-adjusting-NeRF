#!/bin/bash

OUTDIR='/mnt/share/lbh/'
DATADIR='/mnt/datasets'

python train.py --output_root=${OUTDIR}/Ec-NeRF_wkdir --data.root=${OUTDIR}/event_nope_nerf/data/blender \
 --