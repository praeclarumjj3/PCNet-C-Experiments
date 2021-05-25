import numpy as np
import cv2
import os
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import utils
from . import reader

class DemoPartialCompContentDataset(Dataset):

    def __init__(self, config, phase):
        self.dataset = config['dataset']
        if self.dataset == 'COCOA':
            self.data_reader = reader.COCOADataset(config['{}_annot_file'.format(phase)])
        else:
            self.data_reader = reader.KINSLVISDataset(
                self.dataset, config['{}_annot_file'.format(phase)])

        self.img_transform = transforms.Compose([
            transforms.Normalize(config['data_mean'], config['data_std'])
        ])
        self.eraser_setter = utils.EraserSetter(config['eraser_setter'])
        self.sz = config['input_size']
        self.phase = phase

        self.config = config

        self.memcached = config.get('memcached', False)
        self.initialized = False
        self.memcached_client = config.get('memcached_client', None)

    def _load_image(self, fn):
        return Image.open(fn).convert('RGB')

    def __len__(self):
        return self.data_reader.get_image_length()

    def _get_inst(self, idx, randshift=False):
        modal, category, bbox, amodal, imgfn = self.data_reader.get_image_instances(idx, with_gt=True)
        
        modals = []
        amodals = []

        for i in range(modal.shape[0]):
            m = cv2.resize(modal[i],
                (self.sz, self.sz), interpolation=cv2.INTER_NEAREST)

            am = cv2.resize(amodal[i],
                (self.sz, self.sz), interpolation=cv2.INTER_NEAREST)

            modals.append(m)
            amodals.append(am)
        
        rgb = np.array(self._load_image(os.path.join(
                self.config['{}_image_root'.format(self.phase)], imgfn))) # uint8
        rgb = cv2.resize(rgb,
            (self.sz, self.sz), interpolation=cv2.INTER_CUBIC)
        rgb = torch.from_numpy(rgb.astype(np.float32).transpose((2, 0, 1)) / 255.)
        rgb = self.img_transform(rgb) # CHW
    
        return amodals, modals, category, rgb
        

    def __getitem__(self, idx):
        amodal, modal, categories, rgb = self._get_inst(idx, randshift=True) # modal, uint8 {0, 1}

        amodals = []
        vsb_masks = []
        rgbs_erased = []
        erased_masks = []
        for i in range(len(amodal)):

            # get mask 
            invisible_mask = ((amodal[i] == 1) & (modal[i] == 0)) # intersection
            #visible_mask = np.tile((~invisible_mask).astype(np.float32)[np.newaxis,:,:], (3,1,1))
            visible_mask = (~invisible_mask).astype(np.float32)[np.newaxis,:,:]

            # erase modal
            erased_modal = modal[i].copy().astype(np.float32)
            erased_modal = erased_modal * categories[i]

            # convert to tensors
            visible_mask_tensor = torch.from_numpy(visible_mask)
            am = amodal[i].astype(np.float32)[np.newaxis,:,:]
            amodal_tensor = torch.from_numpy(am)
            rgb_erased = rgb.clone()
            rgb_erased = rgb_erased * visible_mask_tensor # erase rgb
            erased_modal_tensor = torch.from_numpy(
                erased_modal.astype(np.float32)).unsqueeze(0) # 1HW
            
            rgbs_erased.append(rgb_erased)
            amodals.append(amodal_tensor)
            vsb_masks.append(visible_mask_tensor)            
            erased_masks.append(erased_modal_tensor)
        
        rgb_erased = torch.stack(rgbs_erased, dim=0)
        amodal_tensor = torch.stack(amodals, dim=0)
        visible_mask_tensor = torch.stack(vsb_masks, dim=0)
        erased_modal_tensor = torch.stack(erased_masks, dim=0)

        return rgb_erased, visible_mask_tensor, erased_modal_tensor, rgb