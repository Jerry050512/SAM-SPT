import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
import random
from typing import Dict, List, Tuple
import logging
import pandas as pd

from utils.dataloader import get_im_gt_name_dict, create_dataloaders, RandomHFlip, Resize, LargeScaleJitter
from utils.loss_mask import loss_masks
import utils.misc as misc
from lora import Linear, MergedLinear, ConvLoRA, lora_state_dict
from segment_anything_training import sam_model_registry
from segment_anything_training.modeling.transformer import Attention
from segment_anything_training.modeling.image_encoder import Attention as EncoderAttention


def get_args_parser():
    parser = argparse.ArgumentParser('spt', add_help=False)
    parser.add_argument("--output", type=str, default='/hy-tmp/output', help="Path to the directory where masks and checkpoints will be output")
    parser.add_argument("--model-type", type=str, default="vit_b", help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']")
    parser.add_argument("--checkpoint", type=str, default="pretrained_checkpoint/vit_b.pth", help="The path to the pretrained SAM checkpoint.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on.")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--lr_drop_epoch', default=10, type=int)
    parser.add_argument('--max_epoch_num', default=50, type=int)
    parser.add_argument('--input_size', default=[1024, 1024], type=list)
    parser.add_argument('--batch_size_train', default=2, type=int)
    parser.add_argument('--batch_size_valid', default=1, type=int)
    parser.add_argument('--model_save_fre', default=1, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument("--restore-model", type=str, help="Path to trained checkpoint for testing.")
    parser.add_argument("--train_data_path", type=str, default='../datasets/NEU-RSDDS-AUG', help="Path to the training data.")
    parser.add_argument("--test_data_path", type=str, default='../datasets/NEU-RSDDS-AUG', help="Path to the testing data.")
    parser.add_argument("--val-point-prompt", default=-1, type=int)
    parser.add_argument("--lora-r", default=8, type=int)
    parser.add_argument("--vatt_alpha", default=0.0, type=float)
    parser.add_argument("--vatt_mask_detach", default=0, type=int)
    parser.add_argument("--vatt_init", default="zero", type=str, choices=["zero", "random"])
    parser.add_argument("--vatt_pos", default=-1, type=int)

    return parser.parse_args()


def prepare_lora(model_type, model: nn.Module, r):
    for name, module in model.named_children():
        if 'neck' in name:
            continue
        if isinstance(module, Attention):
            q_proj = module.q_proj
            v_proj = module.v_proj
            new_q_proj = Linear(q_proj.in_features, q_proj.out_features, r=r)
            new_v_proj = Linear(v_proj.in_features, v_proj.out_features, r=r)
            setattr(module, 'q_proj', new_q_proj)
            setattr(module, 'v_proj', new_v_proj)
        elif isinstance(module, EncoderAttention):
            qkv = module.qkv
            setattr(module, 'qkv', MergedLinear(qkv.in_features, qkv.out_features, r, enable_lora=[True, False, True]))
        elif ('rep' in model_type) and isinstance(module, nn.Conv2d) and module.kernel_size[0] == 1 and module.groups==1:
            setattr(model, name, ConvLoRA(module, module.in_channels, module.out_channels, 1, r=r))
        else:
            prepare_lora(model_type, module, r)     

def freeze_bn_stats(model: nn.Module):
    for _, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

def main(args):
    net = sam_model_registry[args.model_type](checkpoint=args.checkpoint, args=args)
    
    if os.path.exists(args.checkpoint):
        state = net.state_dict()
        prepare_lora(args.model_type, net, args.lora_r)
        net.load_state_dict(state, strict=False)
    else:
        # If the checkpoint doesn't exist, log a warning and proceed without loading.
        # This is useful for initial training where a checkpoint may not be available.
        logging.warning(f"Checkpoint file not found at '{args.checkpoint}'. Starting with a fresh model.")
        prepare_lora(args.model_type, net, args.lora_r)

    for n, p in net.named_parameters():
        if 'lora' not in n:
            p.requires_grad = False

    net.to(args.device)

    if args.eval:
        test_datasets = [{
            "name": "NEU-RSDDS-AUG",
            "im_dir": os.path.join(args.test_data_path, "Image_test"),
            "depth_dir": os.path.join(args.test_data_path, "Depth_test"),
            "gt_dir": "",
            "im_ext": ".bmp",
            "depth_ext": ".tiff",
            "gt_ext": ".png"
        }]
        test_im_gt_list = get_im_gt_name_dict(test_datasets, flag="test")
        test_dataloaders, _ = create_dataloaders(test_im_gt_list, my_transforms=[Resize(args.input_size)], batch_size=args.batch_size_valid, training=False)

        logging.info(f"restore model from: {args.restore_model}")
        net.load_state_dict(torch.load(args.restore_model, map_location=args.device), strict=False)
        evaluate(args, net, test_dataloaders[0])
        return

    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_drop_epoch)

    train_datasets = [{
        "name": "NEU-RSDDS-AUG",
        "im_dir": os.path.join(args.train_data_path, "Image_train"),
        "depth_dir": os.path.join(args.train_data_path, "Depth_train"),
        "gt_dir": os.path.join(args.train_data_path, "GT_train"),
        "im_ext": ".bmp",
        "depth_ext": ".tiff",
        "gt_ext": ".png"
    }]
    train_im_gt_list = get_im_gt_name_dict(train_datasets, flag="train")
    train_dataloader, _ = create_dataloaders(train_im_gt_list, my_transforms=[Resize(args.input_size), RandomHFlip()], batch_size=args.batch_size_train, training=True)

    for epoch in range(args.start_epoch, args.max_epoch_num):
        train_one_epoch(args, net, optimizer, train_dataloader, epoch)
        lr_scheduler.step()

        if (epoch + 1) % args.model_save_fre == 0:
            torch.save(net.state_dict(), f"{args.output}/checkpoint.pth")


def compute_iou(preds, target):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    for i in range(0,len(preds)):
        iou = iou + misc.mask_iou(postprocess_preds[i],target[i])
    return iou / len(preds)

def compute_boundary_iou(preds, target):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    for i in range(0,len(preds)):
        iou = iou + misc.boundary_iou(target[i],postprocess_preds[i])
    return iou / len(preds)



def train_one_epoch(args, net, optimizer, train_dataloader, epoch):
    net.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for data in metric_logger.log_every(train_dataloader, print_freq, header, logging):
        inputs, labels = data['image'].to(args.device), data['label'].to(args.device)

        imgs = inputs.permute(0, 2, 3, 1).cpu().numpy()
        labels_box = misc.masks_to_boxes(labels[:,0,:,:])

        batched_input = []
        for b_i in range(len(imgs)):
            dict_input = dict()
            input_image = torch.as_tensor(imgs[b_i].astype(dtype=np.uint8), device=net.device).permute(2, 0, 1).contiguous()
            dict_input['image'] = input_image
            dict_input['boxes'] = labels_box[b_i:b_i+1]
            dict_input['original_size'] = imgs[b_i].shape[:2]
            batched_input.append(dict_input)

        batched_output, interm_embeddings = net(batched_input, multimask_output=False)

        loss = loss_masks(batched_output, labels, len(batched_output))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    logging.info(f"Averaged stats: {metric_logger}")


@torch.no_grad()
def evaluate(args, net, test_dataloader):
    net.eval()
    logging.info("Testing...")
    output_folder = os.path.join(args.output, "predictions")
    os.makedirs(output_folder, exist_ok=True)

    for data in test_dataloader:
        inputs = data['image'].to(args.device)
        im_name = data['ori_im_path'][0].split('/')[-1].split('.')[0]
        original_size = data['ori_shape']

        img_b, _, h, w = inputs.shape

        batched_input = []
        for b_i in range(img_b):
            dict_input = dict()
            dict_input['image'] = inputs[b_i]
            dict_input['original_size'] = (h, w)
            
            # Use the whole image as a bounding box prompt
            box = torch.tensor([0, 0, w, h], device=net.device)
            dict_input['boxes'] = box.unsqueeze(0)
            batched_input.append(dict_input)

        batched_output, _ = net(batched_input, multimask_output=False)

        masks = [m['masks'] for m in batched_output]

        for i, mask in enumerate(masks):
            # Resize the mask to the original image size
            mask = F.interpolate(mask, tuple(original_size[i].numpy()), mode='bilinear', align_corners=False)
            mask = (mask.squeeze() > 0).cpu().numpy().astype(np.uint8) * 255

            save_path = os.path.join(output_folder, f"{im_name}.png")
            cv2.imwrite(save_path, mask)

    logging.info("Testing finished.")
    return {}, 0.0


if __name__ == "__main__":
    args = get_args_parser()
    
    os.makedirs(args.output, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output, "result.log")),
            logging.StreamHandler()
        ]
    )

    # It is recommended to download the pre-trained model checkpoint from
    # https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    # and place it in the 'pretrained_checkpoint' directory.

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    main(args)
