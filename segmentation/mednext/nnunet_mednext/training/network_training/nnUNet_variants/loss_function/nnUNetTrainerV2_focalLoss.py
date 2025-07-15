#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from torch import nn
from nnunet_mednext.training.loss_functions.focal_loss import FocalLossV2
# from nnunet_mednext.training.loss_functions.dice_loss import GDL_and_CE_loss
from nnunet_mednext.training.loss_functions.dice_loss import DC_and_CE_loss

from nnunet_mednext.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2


class nnUNetTrainerV2_SegLoss_Focal(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                              unpack_data, deterministic, fp16)
        print("Setting up self.loss = Focal_loss({'alpha':0.75, 'gamma':2, 'smooth':1e-4})")

        soft_dice_kwargs = {'batch_dice': self.batch_dice, 'smooth': 1e-4, 'do_bg': False}
        ce_kwargs = {'ignore_index': -1}  
        self.dc_ce_loss = DC_and_CE_loss(
            soft_dice_kwargs=soft_dice_kwargs,
            ce_kwargs=ce_kwargs,
            aggregate="sum",
            square_dice=False,
            weight_ce=1,
            weight_dice=1
        )
        self.loss = lambda x, y: self.focal_loss(x, y) + (0.75 * self.dc_ce_loss(x, y))