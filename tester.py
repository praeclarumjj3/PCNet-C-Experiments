import os
import cv2
import time
import numpy as np

import torch
import torch.optim
import torch.distributed as dist
import torchvision.utils as vutils
from torch.utils.data import DataLoader

import models
import utils
import datasets
from utils.visualize_utils import visualize_demo

class Tester(object):

    def __init__(self, args):

        # get rank
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        if self.rank == 0:
            # mkdir path
            if not os.path.exists('visualizations/demos'):
                os.makedirs('visualizations/demos')

        # create model
        self.model = models.__dict__[args.model['algo']](
            args.model, load_pretrain=args.load_pretrain, dist_model=True)

        self.model.load_state(self.args.load_model)
        self.model.switch_to('eval')

        # dataset
        trainval_class = datasets.__dict__[args.data['trainval_dataset']]

        val_dataset = trainval_class(args.data, 'val')
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=args.data['batch_size_val'],
            shuffle=False,
            num_workers=0,)

        self.args = args

    def run(self):

        # test
        self.train()

    def demo(self):

        for i, inputs in enumerate(self.val_loader):
            
            if i==self.args.demo:
                break

            self.model.set_input(*inputs)
            tensor_dict = self.forward_only(ret_loss = False)
            
            images = tensor_dict['common_tensors'][2]
            comp_masks = tensor_dict['mask_tensors'][0]
            incomp_masks = tensor_dict['mask_tensors'][1]
            inputs = tensor_dict['common_tensors'][0]
            comp_img = tensor_dict['common_tensors'][2]
            pred_obj = tensor_dict['common_tensors'][3] * comp_masks

            if torch.mean(comp_masks).item() != 0:
                visualize_demo(i, 0, images, comp_masks, incomp_masks, inputs, pred_obj, comp_img.detach(), self.args.data)

            if not os.path.exists('visualizations/demos/image_{:04d}/'.format(i)):
                os.makedirs('visualizations/demos/image_{:04d}/'.format(i))
            
