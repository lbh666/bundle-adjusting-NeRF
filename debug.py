import numpy as np
import os,sys,time
import torch
import importlib

import options
from util import log
import random
import cv2 as cv
def main():

    log.process(os.getpid())
    log.title("[{}] (PyTorch code for training NeRF/BARF)".format(sys.argv[0]))

    opt_cmd = options.parse_arguments(sys.argv[1:])
    opt = options.set(opt_cmd=opt_cmd)
    options.save_options_file(opt)

    with torch.cuda.device(opt.device):

        model = importlib.import_module("model.{}".format(opt.model))
        m = model.Model(opt)
        m.setup_visualizer(opt)
        m.load_dataset(opt)
        # pose, pose_GT = m.get_all_training_poses(opt)
        # util_vis.vis_cameras(opt,m.vis,step=0,poses=[pose_GT,pose_GT])
        while True:
            i = random.randint(0, len(m.train_data)-1)
            data = m.train_data[i]['voxel'].sum(dim=1).squeeze()
            data = (data - data.min()) / (data.max() - data.min()) * 255.
            data = data.cpu().detach().numpy().astype(np.uint8)
            cv.imshow('1', data)
            cv.waitKey(0)
            

if __name__=="__main__":
    main()
