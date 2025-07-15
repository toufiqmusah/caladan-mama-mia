# nnunet_mednext/training/network_training/MedNeXt/nnUNetTrainerV2_MedNeXt_HDLoss.py

import torch
import numpy as np
import torch.nn as nn
from torch.cuda.amp import autocast

from monai.losses import HausdorffDTLoss # Import HausdorffDTLoss
from nnunet_mednext.training.loss_functions.dice_loss import DC_and_CE_loss
from nnunet_mednext.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet_mednext.training.network_training.MedNeXt.nnUNetTrainerV2_MedNeXt import nnUNetTrainerV2_MedNeXt_M_kernel3, nnUNetTrainerV2_MedNeXt_M_kernel5

# Defining a combined loss class

class DC_CE_HD_Loss(nn.Module):
    def __init__(self, dc_ce_kwargs, hd_kwargs, weight_dcce=1, weight_hd=1, ignore_label=None):
        super(DC_CE_HD_Loss, self).__init__()
        self.weight_dcce = weight_dcce
        self.weight_hd = weight_hd
        self.dc_ce_loss = DC_and_CE_loss(soft_dice_kwargs=dc_ce_kwargs.get('soft_dice_kwargs', {'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}),
                                         ce_kwargs=dc_ce_kwargs.get('ce_kwargs', {}),
                                         ignore_label=ignore_label)
										 
        # Instantiating Hausdorff loss
	
        self.hd_loss = HausdorffDTLoss(
            include_background=hd_kwargs.get('include_background', False),
            to_onehot_y=hd_kwargs.get('to_onehot_y', True), 
            softmax=hd_kwargs.get('softmax', True),         
            batch=hd_kwargs.get('batch', False),           
            reduction=hd_kwargs.get('reduction', 'mean'),
            alpha=hd_kwargs.get('alpha', 2.0)
        )

    def forward(self, net_output, target):
        target_for_dc_ce = target
        if len(target.shape) == len(net_output.shape):
             # Assuming target is one-hot if shapes match fully except channel dim potentially
             # This might need adjustment based on actual target format
             pass # HD Loss handles one-hot conversion if needed
        elif len(target.shape) == len(net_output.shape) - 1:
             # Assuming target is index based and needs channel dim for DC_CE
             target_for_dc_ce = target.unsqueeze(1)
        else:
             raise ValueError(f"Unexpected target shape {target.shape} for net_output shape {net_output.shape}")

        dc_ce_l = self.dc_ce_loss(net_output, target_for_dc_ce)
        hd_l = self.hd_loss(net_output, target) # HD loss handles one-hot internally if configured

        total_loss = self.weight_dcce * dc_ce_l + self.weight_hd * hd_l
        return total_loss

class nnUNetTrainerV2_MedNeXt_M_kernel3_HDLoss(nnUNetTrainerV2_MedNeXt_M_kernel3): # Inherit from your MedNeXt trainer

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)

        self.loss = DC_CE_HD_Loss(
            # Keep original DC+CE settings from parent class 
            dc_ce_kwargs={'soft_dice_kwargs': {'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}},
            # Configure HD Loss 
            hd_kwargs={'to_onehot_y': True, 'softmax': True, 'include_background': False},
            weight_dcce=1,  
            weight_hd=0.8     
        )
        # Wrap for deep supervision if used (nnUNetTrainerV2 does this in initialize)
        # We need to re-wrap because we redefined self.loss
        if self.ds_loss_weights is not None:
             self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)


class nnUNetTrainerV2_MedNeXt_M_kernel5_HDLoss(nnUNetTrainerV2_MedNeXt_M_kernel5): # Inherit from your MedNeXt trainer

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)

        self.loss = DC_CE_HD_Loss(
            # Keep original DC+CE settings from parent class 
            dc_ce_kwargs={'soft_dice_kwargs': {'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}},
            # Configure HD Loss 
            hd_kwargs={'to_onehot_y': True, 'softmax': True, 'include_background': False},
            weight_dcce=1,  
            weight_hd=0.8     
        )

        if self.ds_loss_weights is not None:
             self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)