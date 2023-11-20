import torch


x = torch.load(r'output\BARF\from_scarch\model.ckpt')

print(x['graph']['se3_refine.weight'].shape)