import numpy as np

import torch

import torch
import matplotlib.pyplot as plt
import os
import numpy as np

def visualize_tensor(tensors_dict, mean, div):
    together = []
    for ct in tensors_dict['common_tensors']:
        ct = unormalize(ct.detach().cpu(), mean, div)
        ct *= 255
        ct = torch.clamp(ct, 0, 255)
        together.append(ct)
    for mt in tensors_dict['mask_tensors']:
        if mt.size(1) == 1:
            mt = mt.repeat(1,3,1,1)
        mt = mt.float().detach().cpu() * 255
        together.append(mt)
    if len(together) == 0:
        return None
    together = torch.cat(together, dim=3)
    together = together.permute(1,0,2,3).contiguous()
    together = together.view(together.size(0), -1, together.size(3))
    return together

def unormalize(tensor, mean, div):
    for c, (m, d) in enumerate(zip(mean, div)):
        tensor[:,c,:,:].mul_(d).add_(m)
    return tensor

def denormalize(tensor, args):
    pixel_mean = torch.Tensor(args['data_mean']).view(3, 1, 1)
    pixel_std = torch.Tensor(args['data_std']).view(3, 1, 1)
    denormalizer = lambda x: torch.clamp((x * pixel_std) + pixel_mean, 0, 1.)

    return denormalizer(tensor)

def visualize_single_map(mapi, name):
    if not os.path.exists('visualizations/'):
            os.makedirs('visualizations/')
    x = torch.stack([mapi[0].cpu() * torch.tensor(255.)]*3, dim=0).squeeze(1)
    x = x.permute(1,2,0)
    x = np.uint8(x)
    plt.imsave('visualizations/{}.jpg'.format(name), np.squeeze(x))

def save_image(img, name, args):
    if not os.path.exists('visualizations/'):
            os.makedirs('visualizations/')
    x = denormalize(img[0].cpu(), args) 
    x = x.permute(1, 2, 0).numpy()
    plt.imsave('visualizations/'+name+'.jpg', x)

def visualize_run(i, phase, img, amodal_mask, erased_modal_mask, erased_img, pred_img, target_img, args):
    plt.rcParams.update({'font.size': 10})

    img = denormalize(img[0].cpu(), args) 
    img = img.permute(1, 2, 0).numpy()

    amodal_mask = torch.stack([amodal_mask[0].cpu() * torch.tensor(255.)]*3, dim=0).squeeze(1)
    amodal_mask = amodal_mask.permute(1, 2, 0).numpy()
    amodal_mask = np.uint8(amodal_mask)

    erased_modal_mask = torch.stack([erased_modal_mask[0].cpu() * torch.tensor(255.)]*3, dim=0).squeeze(1)
    erased_modal_mask = erased_modal_mask.permute(1, 2, 0).numpy()
    erased_modal_mask = np.uint8(erased_modal_mask)

    erased_img = denormalize(erased_img[0].cpu(), args) 
    erased_img = erased_img.permute(1, 2, 0).numpy()

    target_img = denormalize(target_img[0].cpu(), args) 
    target_img = target_img.permute(1, 2, 0).numpy()
    
    if pred_img is not None:
        pred_img = denormalize(pred_img[0].cpu(), args) 
        pred_img = pred_img.permute(1, 2, 0).numpy()
        f, (ax1,ax2,ax3,ax4,ax6,ax5) = plt.subplots(1,6, figsize=(15, 3))
        ax6.imshow(pred_img)
        ax6.set_title("Predicted Object")
        ax6.axis('off')
    else:
        f, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(1,5, figsize=(12, 3))
    
    ax1.imshow(img)
    ax1.set_title("Original Image")
    ax1.axis('off')

    ax2.imshow(amodal_mask)
    ax2.set_title("Complete Mask")
    ax2.axis('off')

    ax3.imshow(erased_modal_mask)
    ax3.set_title("Incomplete Mask")
    ax3.axis('off')
    
    ax4.imshow(erased_img)
    ax4.set_title("Incomplete Object")
    ax4.axis('off')

    ax5.imshow(target_img)
    ax5.set_title("Completed Object")
    ax5.axis('off')

    f.savefig('visualizations/runs/' + phase + '/run' + str(i) + '.jpg')
    plt.close(f)

def visualize_demo(i, j, img, amodal_mask, erased_modal_mask, erased_img, pred_img, comp_img, args):
    plt.rcParams.update({'font.size': 10})

    img = denormalize(img[0].cpu(), args) 
    img = img.permute(1, 2, 0).numpy()

    amodal_mask = torch.stack([amodal_mask[0].cpu() * torch.tensor(255.)]*3, dim=0).squeeze(1)
    amodal_mask = amodal_mask.permute(1, 2, 0).numpy()
    amodal_mask = np.uint8(amodal_mask)

    erased_modal_mask = torch.stack([erased_modal_mask[0].cpu() * torch.tensor(255.)]*3, dim=0).squeeze(1)
    erased_modal_mask = erased_modal_mask.permute(1, 2, 0).numpy()
    erased_modal_mask = np.uint8(erased_modal_mask)

    erased_img = denormalize(erased_img[0].cpu(), args) 
    erased_img = erased_img.permute(1, 2, 0).numpy()

    comp_img = denormalize(comp_img[0].cpu(), args) 
    comp_img = comp_img.permute(1, 2, 0).numpy()
    
    pred_img = denormalize(pred_img[0].cpu(), args) 
    pred_img = pred_img.permute(1, 2, 0).numpy()
    
    f, (ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(1,6, figsize=(15, 3))
    
    ax1.imshow(img)
    ax1.set_title("Original Image")
    ax1.axis('off')

    ax2.imshow(amodal_mask)
    ax2.set_title("Complete Mask")
    ax2.axis('off')

    ax3.imshow(erased_modal_mask)
    ax3.set_title("Incomplete Mask")
    ax3.axis('off')
    
    ax4.imshow(erased_img)
    ax4.set_title("Incomplete Object")
    ax4.axis('off')

    ax5.imshow(pred_img)
    ax5.set_title("Predicted Object")
    ax5.axis('off')

    ax6.imshow(comp_img)
    ax6.set_title("Completed Image")
    ax6.axis('off')

    f.savefig('visualizations/demos/image_{:04d}/'.format(i) + str(j) + '.jpg')
    plt.close(f)