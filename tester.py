import os
import cv2
import time
import numpy as np

import torch
import torch.optim
from torch.utils.data import DataLoader
from colorama import init
from colorama import Fore, Style
init(autoreset=True)
import warnings
warnings.filterwarnings("ignore")
import models
import utils
import datasets
from utils.visualize_utils import visualize_demo
from icecream import ic

class Tester(object):

    def __init__(self, args):

        if not os.path.exists('visualizations/demos'):
            os.makedirs('visualizations/demos')

        # create model
        self.model = models.__dict__[args.model['algo']](
            args.model, load_pretrain=args.load_pretrain)

        self.model.load_state("{}/saved_checkpoints".format(args.exp_path),
                                  args.load_iter)
        self.model.switch_to('eval')

        # dataset
        trainval_class = datasets.__dict__[args.demo['trainval_dataset']]

        val_dataset = trainval_class(args.data, 'val')
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=args.demo['batch_size'],
            shuffle=False,
            num_workers=0,)

        self.args = args

    def run(self):

        # test
        self.demo()
    

    def demo(self):
        
        for i, inputs in enumerate(self.val_loader):
            
            if i==self.args.demo['num']:
                break
            print(Style.BRIGHT + Fore.GREEN + "Starting Visualization...")
            start_time = time.time()
            
            rgbs, visible_masks, modals, rgb_gt = inputs

            if not os.path.exists('visualizations/demos/image_{:04d}/'.format(i)):
                os.makedirs('visualizations/demos/image_{:04d}/'.format(i))

            for j in range(rgbs.shape[1]):

                in_tup = (rgbs[:, j], visible_masks[:, j], modals[:, j], rgb_gt)

                self.model.set_input(*in_tup)
                tensor_dict = self.model.forward_only(ret_loss = False)

                incomp_masks = tensor_dict['mask_tensors'][0]
                comp_masks = 1 - tensor_dict['mask_tensors'][1]

                images = tensor_dict['common_tensors'][2]
                inputs = tensor_dict['common_tensors'][0]
                comp_img = tensor_dict['common_tensors'][1]
                pred_obj = tensor_dict['common_tensors'][3] * comp_masks

                if torch.mean(comp_masks).item() != 0:
                    visualize_demo(i, j, images, comp_masks, incomp_masks, inputs, pred_obj, comp_img.detach(), self.args.data)

            end_time = time.time()
            print(Fore.YELLOW + "Duration: {}".format(end_time-start_time))