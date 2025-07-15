source mia-env/bin/activate

cd caladan-mama-mia/segmentation/

export nnUNet_raw_data_base="caladan-mama-mia/segmentation/nnUNet_raw_data_base"
export nnUNet_preprocessed="caladan-mama-mia/segmentation/nnUNet_preprocessed"
export RESULTS_FOLDER="caladan-mama-mia/segmentation/nnUNet_results/"

nvidia-smi
# mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_M_kernel3 2025 0 -p nnUNetPlansv2.1_trgSp_1x1x1
# mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_M_kernel3 2025 1 -p nnUNetPlansv2.1_trgSp_1x1x1
# mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_M_kernel3 2025 2 -p nnUNetPlansv2.1_trgSp_1x1x1
# mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_M_kernel3 2025 3 -p nnUNetPlansv2.1_trgSp_1x1x1
# mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_M_kernel3 2025 4 -p nnUNetPlansv2.1_trgSp_1x1x1


# UPKERN
# mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_M_kernel5 2025 0 -p nnUNetPlansv2.1_trgSp_1x1x1 -pretrained_weights "/scratch/spark7/MAMA-MIA/nnUNet_results/nnUNet/3d_fullres/Task2025_MIA/nnUNetTrainerV2_MedNeXt_M_kernel3__nnUNetPlansv2.1_trgSp_1x1x1/fold_0/model_best.model" -resample_weights
# mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_M_kernel5 2025 1 -p nnUNetPlansv2.1_trgSp_1x1x1 -pretrained_weights "/scratch/spark7/MAMA-MIA/nnUNet_results/nnUNet/3d_fullres/Task2025_MIA/nnUNetTrainerV2_MedNeXt_M_kernel3__nnUNetPlansv2.1_trgSp_1x1x1/fold_1/model_best.model" -resample_weights
# mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_M_kernel5 2025 2 -p nnUNetPlansv2.1_trgSp_1x1x1 -pretrained_weights "/scratch/spark7/MAMA-MIA/nnUNet_results/nnUNet/3d_fullres/Task2025_MIA/nnUNetTrainerV2_MedNeXt_M_kernel3__nnUNetPlansv2.1_trgSp_1x1x1/fold_2/model_best.model" -resample_weights
# mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_M_kernel5 2025 3 -p nnUNetPlansv2.1_trgSp_1x1x1 -pretrained_weights "/scratch/spark7/MAMA-MIA/nnUNet_results/nnUNet/3d_fullres/Task2025_MIA/nnUNetTrainerV2_MedNeXt_M_kernel3__nnUNetPlansv2.1_trgSp_1x1x1/fold_3/model_best.model" -resample_weights
# mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_M_kernel5 2025 4 -p nnUNetPlansv2.1_trgSp_1x1x1 -pretrained_weights "/scratch/spark7/MAMA-MIA/nnUNet_results/nnUNet/3d_fullres/Task2025_MIA/nnUNetTrainerV2_MedNeXt_M_kernel3__nnUNetPlansv2.1_trgSp_1x1x1/fold_4/model_best.model" -resample_weights


# MAMA-MIA Runs-1
# CUDA_VISIBLE_DEVICES=0 mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_M_kernel3 2025 0 -p nnUNetPlansv2.1_trgSp_1x1x1 & 
# CUDA_VISIBLE_DEVICES=1 mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_M_kernel3 2025 1 -p nnUNetPlansv2.1_trgSp_1x1x1 & 
# CUDA_VISIBLE_DEVICES=2 mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_M_kernel3 2025 2 -p nnUNetPlansv2.1_trgSp_1x1x1 & 
# CUDA_VISIBLE_DEVICES=3 mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_M_kernel3 2025 3 -p nnUNetPlansv2.1_trgSp_1x1x1 & 
# CUDA_VISIBLE_DEVICES=4 mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_M_kernel3 2025 4 -p nnUNetPlansv2.1_trgSp_1x1x1 & 

# wait

# UpKern
# mia-final-runs-250-3x3-1
# mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_M_kernel5 2025 4 -p nnUNetPlansv2.1_trgSp_1x1x1 -pretrained_weights "/scratch/spark7/MAMA-MIA/nnUNet_results_ft4/nnUNet/3d_fullres/Task2025_MIA/nnUNetTrainerV2_MedNeXt_M_kernel5__nnUNetPlansv2.1_trgSp_1x1x1/fold_3/model_best.model"

# UPKERN + Combo Fold + 1250e
# mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_M_kernel5 2025 0 -p nnUNetPlansv2.1_trgSp_1x1x1 -pretrained_weights "/scratch/spark7/MAMA-MIA/nnUNet_results/nnUNet/3d_fullres/Task2025_MIA/nnUNetTrainerV2_MedNeXt_M_kernel3__nnUNetPlansv2.1_trgSp_1x1x1/fold_2/model_best.model" -resample_weights

# UPKERN + Combo FocalGDL + 1000e
#mednextv1_train 3d_fullres nnUNetTrainerV2_FocalGDL_MedNeXt_M_kernel5 2025 0 -p nnUNetPlansv2.1_trgSp_1x1x1 -pretrained_weights "/scratch/spark7/MAMA-MIA/nnUNet_results/nnUNet/3d_fullres/Task2025_MIA/nnUNetTrainerV2_MedNeXt_M_kernel5__nnUNetPlansv2.1_trgSp_1x1x1/fold_0/model_best.model" # -resample_weights

#nnUNetTrainerV2_insaneDA
# mednextv1_train 3d_fullres nnUNetTrainerV2_insaneDA_MedNeXt_M_kernel5 2025 0 -p nnUNetPlansv2.1_trgSp_1x1x1 -pretrained_weights "/scratch/spark7/MAMA-MIA/nnUNet_results/nnUNet/3d_fullres/Task2025_MIA/nnUNetTrainerV2_MedNeXt_M_kernel5__nnUNetPlansv2.1_trgSp_1x1x1/fold_0/model_best.model" # -resample_weights