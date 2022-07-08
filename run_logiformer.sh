#!/usr/bin/env bash
export RECLOR_DIR=reclor_data
export TASK_NAME=reclor
# export RECLOR_DIR=logiqa_data
# export TASK_NAME=logiqa
export MODEL_DIR=roberta-large
export SAVE_DIR=logiformer


CUDA_VISIBLE_DEVICES=1 python run_multiple_choice.py \
    --task_name $TASK_NAME \
    --model_name_or_path $MODEL_DIR \
    --init_weights \
    --do_train \
    --do_eval \
    --do_predict \
    --data_dir $RECLOR_DIR \
    --max_seq_length 256 \
    --per_device_eval_batch_size 2 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --roberta_lr 5e-6 \
    --gt_lr 5e-6 \
    --proj_lr 5e-6 \
    --num_train_epochs 12 \
    --output_dir checkpoints/$TASK_NAME/${SAVE_DIR} \
    --fp16 \
    --logging_steps 200 \
    --adam_epsilon 1e-6 \
    --weight_decay 0.01 \
    --overwrite_output_dir \
    --load_best_model_at_end \
    --metric_for_best_model eval_acc \
    --save_strategy no \
    --overwrite_cache \
    # --evaluation_strategy epoch\
