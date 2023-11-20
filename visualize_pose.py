import numpy as np
import os,sys,time
import torch
import importlib

import options
from util import log
import util_vis
import camera

def main():

    log.process(os.getpid())
    log.title("[{}] (PyTorch code for training NeRF/BARF)".format(sys.argv[0]))

    opt_cmd = options.parse_arguments(sys.argv[1:])
    opt = options.set(opt_cmd=opt_cmd)
    options.save_options_file(opt)

    with torch.cuda.device(opt.device):

        model = importlib.import_module("model.{}".format(opt.model))
        m = model.Model(opt)

        m.load_dataset(opt)
        m.build_networks(opt)
        m.setup_optimizer(opt)
        m.restore_checkpoint(opt)
        m.setup_visualizer(opt)
        align = True
        if align:
            pose_pred,pose_GT = m.get_all_training_poses(opt)
            poses = pose_pred if opt.model=="barf" else pose_GT
            pose, sim3 = m.prealign_cameras(opt,pose_pred,pose_GT)

        else:
            pose,pose_GT = m.get_all_training_poses(opt)
        util_vis.vis_cameras(opt,m.vis,step=0,poses=[pose,pose_GT])

if __name__=="__main__":
    main()
