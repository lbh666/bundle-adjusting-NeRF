import numpy as np
import os,sys,time
import torch
import importlib

import options
from util import log
import tqdm
from easydict import EasyDict as edict
import torchvision.transforms.functional as torchvision_F



novel_path = 'results'

def main():
    if not os.path.exists(novel_path):
        os.makedirs(novel_path)

    eps = 1e-10
    log.process(os.getpid())
    log.title("[{}] (PyTorch code for evaluating NeRF/BARF)".format(sys.argv[0]))

    opt_cmd = options.parse_arguments(sys.argv[1:])
    opt = options.set(opt_cmd=opt_cmd)

    with torch.cuda.device(opt.device):

        model = importlib.import_module("model.{}".format(opt.model))
        m = model.Model(opt)

        m.load_dataset(opt,eval_split="test")
        m.build_networks(opt)

        m.restore_checkpoint(opt)
        m.graph.eval()

        pose_pred,pose_GT = m.get_all_training_poses(opt)
        poses = pose_pred if opt.model=="barf" else pose_GT
        pose_novel_tqdm = tqdm.tqdm(poses,desc="rendering novel views",leave=False)
        intr = edict(next(iter(m.test_loader))).intr[:1].to(opt.device)
        for i,pose in enumerate(pose_novel_tqdm):
            with torch.no_grad():
                ret = m.graph.render_by_slices(opt,pose[None],intr=intr) if opt.nerf.rand_rays else \
                        m.graph.render(opt,pose[None],intr=intr)
            invdepth = (1-ret.depth)/ret.opacity if opt.camera.ndc else 1/(ret.depth/ret.opacity+eps)
            rgb_map = ret.rgb.view(-1,opt.H,opt.W,3).permute(0,3,1,2) # [B,3,H,W]
            invdepth_map = invdepth.view(-1,opt.H,opt.W,1).permute(0,3,1,2) # [B,1,H,W]
            torchvision_F.to_pil_image(rgb_map.cpu()[0]).save("{}/rgb_{}.png".format(novel_path,i))
            torchvision_F.to_pil_image(invdepth_map.cpu()[0]).save("{}/depth_{}.png".format(novel_path,i))

if __name__=="__main__":
    main()
