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
from utils.visualize_utils import visualize

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
            with torch.no_grad():
                tensor_dict, _ = self.model.forward_only()

            if self.rank == 0:
               
                all_together.append(
                    utils.visualize_tensor(tensor_dict,
                    self.args.data.get('data_mean', [0,0,0]),
                    self.args.data.get('data_std', [1,1,1])))
                
                all_together = torch.cat(all_together, dim=2)
                grid = vutils.make_grid(all_together,
                                        nrow=1,
                                        normalize=True,
                                        range=(0, 255),
                                        scale_each=False)
                cv2.imwrite("visualizations/demos/{}{}.png".format(
                    'demo', i), grid.permute(1, 2, 0).numpy())
            # results = self.forward_only(ret_loss = False)

            # rgb_erased, comp_img, images = results['common_tensors']
            # modals, vsb_masks = results['mask_tensors']

            # if torch.mean(vsb_masks).item() != 0:
                    # visualize_demo(i, j, images, comp_masks[:, j]+incomp_masks[:, j], incomp_masks[:, j], inputs, pred_obj, comp_img.detach(), self.args.data)

            if not os.path.exists('visualizations/demos/image_{:04d}/'.format(i)):
                os.makedirs('visualizations/demos/image_{:04d}/'.format(i))
            
