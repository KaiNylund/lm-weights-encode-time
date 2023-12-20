#!/bin/bash
#SBATCH --partition=gpu-titan
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#-C 'a40|a100|titan'

wmt_dir=""
poli_aff_dir=""

TASK="wmt_lm"
MODEL="t5-3b"
SEED=42
LR=0.0002

'''
out_dir="/models/${MODEL}_${TASK}_updating_in_order_test/"
train_file="${poli_aff_dir}train/indivs/combined_years_in_order.jsonl"
python -u run_finetune_t5.py \
        --model_name_or_path $MODEL \
        --train_file $train_file \
        --dataset_name "yelp_polarity" \
        --dataset_config "plain_text" \
        --do_train \
        --train_eval_files "${poli_aff_dir}dev/2015.jsonl" "${poli_aff_dir}dev/2016.jsonl" \
                           "${poli_aff_dir}dev/2017.jsonl" "${poli_aff_dir}dev/2018.jsonl" \
                           "${poli_aff_dir}dev/2019.jsonl" "${poli_aff_dir}dev/2020.jsonl" \
        --input_column_1 "text" \
        --output_dir $out_dir \
        --seed $SEED \
        --save_steps 200 \
        --save_strategy no \
        --source_prefix_1 "lm:" \
        --target_label label \
        --learning_rate $LR \
        --max_predict_samples 1000 \
        --max_source_length 128 \
        --max_target_length 128 \
        --preprocessing_num_workers 1 \
        --gradient_accumulation_steps 8 \
        --ddp_find_unused_parameters False \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --predict_with_generate \
        --patience 3 \
        --lora \
        #--lm \

'''
out_dir=""
mkdir -p $out_dir

prev_ckpt=$MODEL
for train_year in {2012..2016}
do
    for train_month in {0..11}
    do
        #train_file="${poli_aff_dir}train/indivs/${train_year}_${train_month}_with_month_flag.jsonl"
        #train_file="${twitter_dir}train/year_${train_year}_month_${train_month}_30000000_bytes"
        train_file="${wmt_dir}train_json/year_${train_year}_month_${train_month}_30000000_bytes"

        if [ ! -f $train_file ]; then
            echo "Missing train file for ${train_year}_${train_month}!"
        else
            echo "Training on ${train_year}_${train_month}!"
            cur_out_dir="${out_dir}up_to_${train_year}_${train_month}"
            python -u finetune_t5.py  \
                --model_name_or_path $prev_ckpt \
                --train_file $train_file \
                --dataset_name "yelp_polarity" \
                --dataset_config "plain_text" \
                --do_train \
                --input_column_1 "text" \
                --output_dir $cur_out_dir \
                --seed $SEED \
                --save_steps 200 \
                --save_strategy no \
                --source_prefix_1 "lm:" \
                --target_label label \
                --learning_rate $LR \
                --max_predict_samples 1000 \
                --max_source_length 128 \
                --max_target_length 128 \
                --gradient_accumulation_steps 8 \
                --ddp_find_unused_parameters False \
                --per_device_train_batch_size 2 \
                --per_device_eval_batch_size 2 \
                --predict_with_generate \
                --patience 3 \
                --lora \
                --lm \
            prev_ckpt=$cur_out_dir
        fi
    done
done
