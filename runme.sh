#!/bin/bash
# You need to modify this path to your downloaded dataset directory
DATASET_DIR="/home/liangjh/cs4347/weakly_supervised_SED/data"

# You need to modify this path to your workspace to store features and models
WORKSPACE="/home/liangjh/cs4347/weakly_supervised_SED/workspace"


# Set model type
MODEL_TYPE="Cnn_Gru"


# Pack waveforms to hdf5
python3 utils/features.py pack_audio_files_to_hdf5 --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --data_type='testing'
python3 utils/features.py pack_audio_files_to_hdf5 --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --data_type='training'


# ------ Train ------
python3 pytorch/main.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --holdout_fold=1 --model_type=$MODEL_TYPE --loss_type='clip_bce' --augmentation='mixup' --learning_rate=1e-3 --batch_size=32 --resume_iteration=0 --stop_iteration=50000 --cuda


# ------ Calculate metrics ------
# Calculate statistics
python3 utils/calculate_metrics.py calculate_metrics --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --filename='main' --holdout_fold=1 --model_type=$MODEL_TYPE --loss_type='clip_bce' --augmentation='mixup' --batch_size=32 --iteration=50000 --data_type='test'


"""
# ------ Inference and dump predicted probabilites ------
# The following code will not work untill the evaluation set is available
python3 utils/features.py pack_audio_files_to_hdf5 --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --data_type='evaluation'
python3 pytorch/main.py inference_prob --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --holdout_fold=1 --model_type=$MODEL_TYPE --loss_type='clip_bce' --augmentation='mixup' --batch_size=32 --iteration=50000 --cuda
"""