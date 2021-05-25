import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim
import torch.distributed as dist
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from etaprogress.progress import ProgressBar
from colorama import init
from colorama import Fore, Style
from utils.visualize_utils import visualize_run
from icecream import ic

import sys
import models
import utils
import datasets
#from dataset import ImageRawDataset, PartialCompEvalDataset, PartialCompDataset
import inference as infer
import pdb

class Trainer(object):

    def __init__(self, args):

        # get rank
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        if self.rank == 0:
            # mkdir path
            if not os.path.exists('visualizations/runs/'):
                os.makedirs('visualizations/runs/')
            if not os.path.exists('{}/saved_checkpoints'.format(args.exp_path)):
                os.makedirs('{}/saved_checkpoints'.format(args.exp_path))

        # create model
        self.model = models.__dict__[args.model['algo']](
            args.model, load_pretrain=args.load_pretrain, dist_model=True)

        # optionally resume from a checkpoint
        assert not (args.load_iter is not None and args.load_pretrain is not None), \
            "load_iter and load_pretrain are exclusive."

        if args.load_iter is not None:
            self.model.load_state("{}/saved_checkpoints".format(args.exp_path),
                                  args.load_iter, args.resume)
            self.start_iter = args.load_iter
        else:
            self.start_iter = 0

        self.curr_step = self.start_iter

        # lr scheduler & datasets
        trainval_class = datasets.__dict__[args.data['trainval_dataset']]

        if not args.validate:  # train
            self.lr_scheduler = utils.StepLRScheduler(
                self.model.optim,
                args.model['lr_steps'],
                args.model['lr_mults'],
                args.model['lr'],
                args.model['warmup_lr'],
                args.model['warmup_steps'],
                last_iter=self.start_iter - 1)

            train_dataset = trainval_class(args.data, 'train')
            train_sampler = utils.DistributedGivenIterationSampler(
                train_dataset,
                args.model['total_iter'],
                args.data['batch_size'],
                last_iter=self.start_iter - 1)
            self.train_loader = DataLoader(train_dataset,
                                           batch_size=args.data['batch_size'],
                                           shuffle=False,
                                           num_workers=args.data['workers'],
                                           pin_memory=False,
                                           sampler=train_sampler)

        val_dataset = trainval_class(args.data, 'val')
        val_sampler = utils.DistributedSequentialSampler(val_dataset)
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=args.data['batch_size_val'],
            shuffle=False,
            num_workers=args.data['workers'],
            pin_memory=False,
            sampler=val_sampler)

        self.args = args

    def run(self):

        # offline validate
        if self.args.validate:
            self.validate('off_val')
            return

        if self.args.trainer['initial_val']:
            self.validate('on_val')

        # train
        self.train()
    
    def plot_iter_loss(self, k, iter_l1_loss, iter_adv_g_loss, iter_adv_d_loss):
        plt.close()
        plt.plot(list(range(1, len(iter_l1_loss) + 1)), iter_l1_loss, label = "Rec Loss")
        plt.plot(list(range(1, len(iter_adv_g_loss) + 1)), iter_adv_g_loss, label = "Adv G Loss")
        plt.plot(list(range(1, len(iter_adv_d_loss) + 1)), iter_adv_d_loss, label = "Adv D Loss")
        if not os.path.exists('visualizations/losses/{}'.format(k)):
                os.makedirs('visualizations/losses/{}'.format(k))
        plt.legend()
        plt.title("Training Loss")
        plt.savefig('visualizations/losses/{}/train_loss.png'.format(k))
        plt.close()
        plt.plot(list(range(1, len(iter_l1_loss) + 1)), iter_l1_loss, label = "Rec Loss")
        plt.legend()
        plt.title("Training Loss")
        plt.savefig('visualizations/losses/{}/train_rec_loss.png'.format(k))
        plt.close()
        plt.plot(list(range(1, len(iter_adv_g_loss) + 1)), iter_adv_g_loss, label = "Adv G Loss",  color = "green")
        plt.legend()
        plt.title("Training Loss")
        plt.savefig('visualizations/losses/{}/train_adv_g_loss.png'.format(k))
        plt.close()
        plt.plot(list(range(1, len(iter_adv_d_loss) + 1)), iter_adv_d_loss, label = "Adv D Loss",  color = "red")
        plt.legend()
        plt.title("Training Loss")
        plt.savefig('visualizations/losses/{}/train_adv_d_loss.png'.format(k))

    def train(self):

        btime_rec = utils.AverageMeter(10)
        dtime_rec = utils.AverageMeter(10)
        recorder = {}
        for rec in self.args.trainer['loss_record']:
            recorder[rec] = utils.AverageMeter(10)

        self.model.switch_to('train')

        end = time.time()

        if self.rank == 0:
            total = self.args.model['total_iter']
            bar = ProgressBar(total, max_width=80)
        
        if self.rank == 0:
            iter_l1_loss = []
            iter_adv_g_loss = []
            iter_adv_d_loss = []

        for i, inputs in enumerate(self.train_loader):
            self.curr_step = self.start_iter + i
            self.lr_scheduler.step(self.curr_step)
            curr_lr = self.lr_scheduler.get_lr()[0]

            # measure data loading time
            dtime_rec.update(time.time() - end)

            self.model.set_input(*inputs)
            loss_dict = self.model.step()

            for k in loss_dict.keys():
                recorder[k].update(utils.reduce_tensors(loss_dict[k]).item())

            btime_rec.update(time.time() - end)
            end = time.time()

            self.curr_step += 1

            if self.rank == 0:
                bar.numerator = self.curr_step 
                print(bar, end='\r')

                    
            if self.rank == 0:
                for key, coef in self.args.model['lambda_dict'].items():
                    loss_dict[key] = coef * loss_dict[key]
                l1_loss = loss_dict['valid'] + loss_dict['hole'] + loss_dict['tv'] + loss_dict['prc'] + loss_dict['style']
                iter_l1_loss.append(l1_loss.item())

                advg_loss = loss_dict['adv']
                iter_adv_g_loss.append(advg_loss.item())

                advd_loss = loss_dict['dis']
                iter_adv_d_loss.append(advd_loss.item())

                sys.stdout.flush()
            
            # logging
            if self.rank == 0 and self.curr_step % self.args.trainer[
                    'print_freq'] == 0:
                loss_str = ""
                
                for k in recorder.keys():
                    loss_str += '{}: {loss.val:.4g} ({loss.avg:.4g})\t'.format(
                        k, loss=recorder[k])
                
                description = 'Training Iter: [{0}/{1}], '.format(self.curr_step,
                                           len(self.train_loader)) + \
                                               'Rec Loss: {}, Adv G Loss: {}, Adv D Loss: {}'.format(l1_loss, advg_loss, advd_loss)
                ic(description)
                k = self.curr_step // self.args.trainer['print_freq']
                self.plot_iter_loss(k, iter_l1_loss, iter_adv_g_loss, iter_adv_d_loss) 
            
            # save
            if (self.rank == 0 and
                (self.curr_step % self.args.trainer['save_freq'] == 0 or
                 self.curr_step == self.args.model['total_iter'])):
                self.model.save_state(
                    "{}/saved_checkpoints".format(self.args.exp_path),
                    self.curr_step)
                print('### Model Saved at Iteration {}'.format(self.curr_step))

            # validate
            if (self.curr_step % self.args.trainer['val_freq'] == 0 or
                self.curr_step == self.args.model['total_iter']):
                self.validate('val')
            

    def validate(self, phase):
        btime_rec = utils.AverageMeter(0)
        dtime_rec = utils.AverageMeter(0)
        recorder = {}
        for rec in self.args.trainer['loss_record']:
            recorder[rec] = utils.AverageMeter(10)

        self.model.switch_to('eval')

        end = time.time()
        for i, inputs in enumerate(self.val_loader):
            if ('val_iter' in self.args.trainer and
                    self.args.trainer['val_iter'] != -1 and
                    i == self.args.trainer['val_iter']):
                break

            dtime_rec.update(time.time() - end)

            self.model.set_input(*inputs)
            tensor_dict, loss_dict = self.model.forward_only()
            for k in loss_dict.keys():
                recorder[k].update(utils.reduce_tensors(loss_dict[k]).item())
            btime_rec.update(time.time() - end)
            end = time.time()

            # logging
            if self.rank == 0 and (i+1) % self.args.trainer['val_iter'] == 0:
                loss_str = ""
                for k in recorder.keys():
                    loss_str += '{}: {loss.val:.4g} ({loss.avg:.4g})\t'.format(
                        k, loss=recorder[k])

                description = 'Validation Iter: [{0}]'.format(self.curr_step)
                ic(description)
                ic(loss_dict)
                k = self.curr_step // self.args.trainer['val_freq']
                
                incomp_masks = tensor_dict['mask_tensors'][0]
                comp_masks = tensor_dict['mask_tensors'][1]

                images = tensor_dict['common_tensors'][2]
                inputs = tensor_dict['common_tensors'][0]
                comp_img = tensor_dict['common_tensors'][1]
                pred_img = tensor_dict['common_tensors'][3]

                visualize_run(k, phase, images, comp_masks, incomp_masks, inputs, pred_img.detach(), comp_img.detach(), self.args.data)
        self.model.switch_to('train')
