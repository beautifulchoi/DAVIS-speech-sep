#!/bin/bash 

OPTS=""
OPTS+="--id muddy-clip "

OPTS+="--list_train /home/prj/data/valid_muddy_mix_audios_train.csv "
OPTS+="--list_val /home/prj/data/valid_muddy_mix_audios_val.csv "

# Models
OPTS+="--img_pool maxpool "
OPTS+="--num_channels 64 "
OPTS+="--loss l1 "
OPTS+="--weighted_loss 0 "

# logscale in frequency
OPTS+="--num_mix 2 "
OPTS+="--log_freq 1 "

# frames-related
OPTS+="--arch_frame clip " # [resnet18, clip]
OPTS+="--num_frames 11 "
OPTS+="--stride_frames 1 "
OPTS+="--frameRate 1 "

# audio-related
OPTS+="--audLen 65535 " # 65535
OPTS+="--audRate 11025 " #11025

# learning params
OPTS+="--num_gpus 1 "
OPTS+="--gpu_ids 1 "
OPTS+="--workers 4 "
OPTS+="--batch_size_per_gpu 16 "
OPTS+="--lr_frame 1e-4 " #1e-4
OPTS+="--lr_unet 1e-4 " #1e-4
OPTS+="--num_epoch 400 "
OPTS+="--lr_steps 100 200 300 "
OPTS+="--dup_trainset 5 "
OPTS+="--eval_epoch 1 "

# where to save the results
OPTS+="--ckpt /home/prj/DAVIS/result/muddy_mix_speech_sep "

# display, viz
OPTS+="--disp_iter 200 "
OPTS+="--num_vis 3 "
# OPTS+="--num_val 26 "

OPTS+="--wandb_run_name muddy_mix_speech_sep "
OPTS+="--wandb_mode online " #online offline disabled

#OPTS+="--dev_mode "
OPTS+="--split test "
OPTS+="--mode train"

CUDA_VISIBLE_DEVICES=1 python -u ../main_fm_muddy.py $OPTS
