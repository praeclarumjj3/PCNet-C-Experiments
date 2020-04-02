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
            if not os.path.exists('{}/events'.format(args.exp_path)):
                os.makedirs('{}/events'.format(args.exp_path))
            if not os.path.exists('{}/images'.format(args.exp_path)):
                os.makedirs('{}/images'.format(args.exp_path))
            if not os.path.exists('{}/logs'.format(args.exp_path)):
                os.makedirs('{}/logs'.format(args.exp_path))
            if not os.path.exists('{}/checkpoints'.format(args.exp_path)):
                os.makedirs('{}/checkpoints'.format(args.exp_path))

            # logger
            if args.trainer['tensorboard'] and not (args.extract or args.evaluate):
                try:
                    from tensorboardX import SummaryWriter
                except:
                    raise Exception("Please switch off \"tensorboard\" "
                                    "in your config file if you do not "
                                    "want to use it, otherwise install it.")
                self.tb_logger = SummaryWriter('{}/events'.format(
                    args.exp_path))
            else:
                self.tb_logger = None
            if args.validate:
                self.logger = utils.create_logger(
                    'global_logger',
                    '{}/logs/log_offline_val.txt'.format(args.exp_path))
            elif args.extract:
                self.logger = utils.create_logger(
                    'global_logger',
                    '{}/logs/log_extract.txt'.format(args.exp_path))
            elif args.evaluate:
                self.logger = utils.create_logger(
                    'global_logger',
                    '{}/logs/log_evaluate.txt'.format(args.exp_path))
            else:
                self.logger = utils.create_logger(
                    'global_logger',
                    '{}/logs/log_train.txt'.format(args.exp_path))

        # create model
        self.model = models.__dict__[args.model['algo']](
            args.model, load_pretrain=args.load_pretrain, dist_model=True)

        # optionally resume from a checkpoint
        assert not (args.load_iter is not None and args.load_pretrain is not None), \
            "load_iter and load_pretrain are exclusive."

        if args.load_iter is not None:
            self.model.load_state("{}/checkpoints".format(args.exp_path),
                                  args.load_iter, args.resume)
            self.start_iter = args.load_iter
        else:
            self.start_iter = 0

        self.curr_step = self.start_iter

        # lr scheduler & datasets
        trainval_class = datasets.__dict__[args.data['trainval_dataset']]
        eval_class = (None if args.data['eval_dataset']
                      is None else datasets.__dict__[args.data['eval_dataset']])
        extract_class = (None if args.data['extract_dataset']
                         is None else datasets.__dict__[args.data['extract_dataset']])

        if not (args.validate or args.extract or args.evaluate):  # train
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

        if not (args.extract or args.evaluate):  # train or offline validation
            val_dataset = trainval_class(args.data, 'val')
            val_sampler = utils.DistributedSequentialSampler(val_dataset)
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=args.data['batch_size_val'],
                shuffle=False,
                num_workers=args.data['workers'],
                pin_memory=False,
                sampler=val_sampler)

        if not (args.validate or args.extract) and eval_class is not None: # train or offline evaluation
            eval_dataset = eval_class(args.data, 'val')
            eval_sampler = utils.DistributedSequentialSampler(
                eval_dataset)
            self.eval_loader = DataLoader(eval_dataset,
                                          batch_size=1,
                                          shuffle=False,
                                          num_workers=1,
                                          pin_memory=False,
                                          sampler=eval_sampler)

        if args.extract:  # extract
            assert extract_class is not None, 'Please specify extract_dataset'
            extract_dataset = extract_class(args.data, 'val')
            extract_sampler = utils.DistributedSequentialSampler(
                extract_dataset)
            self.extract_loader = DataLoader(extract_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=1,
                                             pin_memory=False,
                                             sampler=extract_sampler)

        self.args = args

    def run(self):

        assert self.args.validate + self.args.extract + self.args.evaluate < 2

        # offline validate
        if self.args.validate:
            self.validate('off_val')
            return

        # extract
        if self.args.extract:
            self.extract()
            return

        # evaluate
        if self.args.evaluate:
            self.evaluate('off_eval')
            return

        if self.args.trainer['initial_val']:
            self.validate('on_val')

        if self.args.trainer['eval'] and self.args.trainer['initial_eval']:
            self.evaluate('on_eval')

        # train
        self.train()

    def train(self):

        btime_rec = utils.AverageMeter(10)
        dtime_rec = utils.AverageMeter(10)
        recorder = {}
        for rec in self.args.trainer['loss_record']:
            recorder[rec] = utils.AverageMeter(10)

        self.model.switch_to('train')

        end = time.time()
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

            # logging
            if self.rank == 0 and self.curr_step % self.args.trainer[
                    'print_freq'] == 0:
                loss_str = ""
                if self.tb_logger is not None:
                    self.tb_logger.add_scalar('lr', curr_lr, self.curr_step)
                for k in recorder.keys():
                    if self.tb_logger is not None:
                        self.tb_logger.add_scalar('train_{}'.format(k),
                                                  recorder[k].avg,
                                                  self.curr_step)
                    loss_str += '{}: {loss.val:.4g} ({loss.avg:.4g})\t'.format(
                        k, loss=recorder[k])

                self.logger.info(
                    'Iter: [{0}/{1}]\t'.format(self.curr_step,
                                               len(self.train_loader)) +
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                        batch_time=btime_rec) +
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                        data_time=dtime_rec) + loss_str +
                    'lr {lr:.2g}'.format(lr=curr_lr))

            # save
            if (self.rank == 0 and
                (self.curr_step % self.args.trainer['save_freq'] == 0 or
                 self.curr_step == self.args.model['total_iter'])):
                self.model.save_state(
                    "{}/checkpoints".format(self.args.exp_path),
                    self.curr_step)

            # validate
            if (self.curr_step % self.args.trainer['val_freq'] == 0 or
                self.curr_step == self.args.model['total_iter']):
                self.validate('on_val')

            if (self.curr_step % self.args.trainer['eval_freq'] == 0 or
                self.curr_step == self.args.model['total_iter']) and self.args.trainer['eval']:
                self.evaluate('on_eval')

            

    def validate(self, phase):
        btime_rec = utils.AverageMeter(0)
        dtime_rec = utils.AverageMeter(0)
        recorder = {}
        for rec in self.args.trainer['loss_record']:
            recorder[rec] = utils.AverageMeter(10)

        self.model.switch_to('eval')

        end = time.time()
        all_together = []
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

            # tb visualize
            if self.rank == 0:
                disp_start = max(self.args.trainer['val_disp_start_iter'], 0)
                disp_end = min(self.args.trainer['val_disp_end_iter'], len(self.val_loader))
                if (i >= disp_start and i < disp_end):
                    all_together.append(
                        utils.visualize_tensor(tensor_dict,
                        self.args.data.get('data_mean', [0,0,0]),
                        self.args.data.get('data_std', [1,1,1])))
                if (i == disp_end - 1 and disp_end > disp_start):
                    all_together = torch.cat(all_together, dim=2)
                    grid = vutils.make_grid(all_together,
                                            nrow=1,
                                            normalize=True,
                                            range=(0, 255),
                                            scale_each=False)
                    if self.tb_logger is not None:
                        self.tb_logger.add_image('Image_' + phase, grid,
                                                 self.curr_step)
                    cv2.imwrite("{}/images/{}_{}.png".format(
                        self.args.exp_path, phase, self.curr_step),
                        grid.permute(1, 2, 0).numpy())

        # logging
        if self.rank == 0:
            loss_str = ""
            for k in recorder.keys():
                if self.tb_logger is not None and phase == 'on_val':
                    self.tb_logger.add_scalar('val_{}'.format(k),
                                              recorder[k].avg,
                                              self.curr_step)
                loss_str += '{}: {loss.val:.4g} ({loss.avg:.4g})\t'.format(
                    k, loss=recorder[k])

            self.logger.info(
                'Validation Iter: [{0}]\t'.format(self.curr_step) +
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                    batch_time=btime_rec) +
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                    data_time=dtime_rec) + loss_str)

        self.model.switch_to('train')

    def extract(self):
        raise NotImplemented

    def evaluate(self, phase): # padded samples are evalauted twice
        btime_rec = utils.AverageMeter()
        allpair_true_rec = utils.AverageMeter()
        allpair_rec = utils.AverageMeter()
        occpair_true_rec = utils.AverageMeter()
        occpair_rec = utils.AverageMeter()
        intersection_rec = utils.AverageMeter()
        union_rec = utils.AverageMeter()
        target_rec = utils.AverageMeter()

        self.model.switch_to('eval')

        end = time.time()
        for i, (image, modal, category, bboxes, amodal, gt_order_matrix) in enumerate(self.eval_loader):
            image = image.squeeze(0).numpy()
            modal = modal.squeeze(0).numpy()
            category = category.squeeze(0).numpy()
            bboxes = bboxes.squeeze(0).numpy()
            amodal = amodal.squeeze(0).numpy()
            gt_order_matrix = gt_order_matrix.squeeze(0).numpy()

            allpair_true, allpair, occpair_true, occpair, intersection, union, target = \
                self.model.evaluate(image, modal, category, bboxes, amodal,
                gt_order_matrix, self.args.data['input_size'])

            intersection_rec.update(intersection)
            union_rec.update(union)
            target_rec.update(target)
    
            allpair_true_rec.update(allpair_true)
            allpair_rec.update(allpair)
            occpair_true_rec.update(occpair_true)
            occpair_rec.update(occpair)

            btime_rec.update(time.time() - end)
            end = time.time()

            if self.rank == 0 and i % self.args.trainer['print_freq'] == 0:
                self.logger.info(
                    'Eval Iter: [{0}] ({1}/{2})\t'.format(self.curr_step, i, len(self.eval_loader)) +
                    'Time {btime.val:.3f} ({btime.avg:.3f})\t'.format(btime=btime_rec))

        reduced_allpair_true = utils.reduce_tensors(
            torch.FloatTensor([allpair_true_rec.sum / self.world_size]).cuda()).item()
        reduced_allpair = utils.reduce_tensors(
            torch.FloatTensor([allpair_rec.sum / self.world_size]).cuda()).item()
        reduced_occpair_true = utils.reduce_tensors(
            torch.FloatTensor([occpair_true_rec.sum / self.world_size]).cuda()).item()
        reduced_occpair = utils.reduce_tensors(
            torch.FloatTensor([occpair_rec.sum / self.world_size]).cuda()).item()

        reduced_intersection = utils.reduce_tensors(
            torch.FloatTensor([intersection_rec.sum / self.world_size]).cuda()).item()
        reduced_union = utils.reduce_tensors(
            torch.FloatTensor([union_rec.sum / self.world_size]).cuda()).item()
        reduced_target = utils.reduce_tensors(
            torch.FloatTensor([target_rec.sum / self.world_size]).cuda()).item()

        acc_allpair = reduced_allpair_true / reduced_allpair
        acc_occpair = reduced_occpair_true / reduced_occpair

        iou = reduced_intersection / (reduced_union + 1e-10)
        acc = reduced_intersection / (reduced_target + 1e-10)

        if self.rank == 0:
            self.logger.info(
                'Order_acc_all: {:.5g}\tOrder_acc_occ: {:.5g}\t'
                'Amodal_mIoU: {:.5g}\tAmodal_acc: {:.5g}'.format(
                    acc_allpair, acc_occpair, iou, acc))

            if self.tb_logger is not None and phase == 'on_eval':
                self.tb_logger.add_scalar('val_acc_all', acc_allpair, self.curr_step)
                self.tb_logger.add_scalar('val_acc_occ', acc_occpair, self.curr_step)
                self.tb_logger.add_scalar('val_amodal_miou', iou, self.curr_step)
                self.tb_logger.add_scalar('val_amodal_acc', acc, self.curr_step)

        self.model.switch_to('train')
