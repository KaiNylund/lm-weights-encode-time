#!/bin/bash

# TODO: Modify poli_aff path and the train file path below to finetune on PoliAff 
# task from "time waits for no one!" (Luu et al., 2022)
poli_aff_train_dir=""

SEED=42
LR=0.0008
MODEL="t5-small"
OUT_DIR="./models_seed_${SEED}/"

# Finetune WMT LM month models
dataset_name="KaiNylund/WMT-month-splits"
for train_year in {2012..2016}
do
    for train_month in {0..11}
    do
        if ([[ train_year == 2012 ]] && [[ train_month == 7 ]]) ||
           ([[ train_year == 2016 ]] && [[ train_month == 5 ]]);
        then
            echo "Missing split for WMT LM ${train_year}_${train_month}!"
        else
            echo "Training on WMT LM ${train_year}_${train_month}!"
            train_split="${train_year}_${train_month}_train"
            python -u ../finetuning_scripts/finetune_t5.py  \
                --model_name_or_path $MODEL \
                --dataset_name $dataset_name \
                --train_split $train_split \
                --do_train \
                --input_column_1 "text" \
                --output_dir "${OUT_DIR}wmt_lm_${train_year}_${train_month}" \
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
                --lm
        fi
    done
done


# Finetune PoliAff month models
if [[ $poli_aff_train_dir != ""]]
then
    for train_year in {2015..2020}
    do
        for train_month in {0..11}
        do
            echo "Training on PoliAff ${train_year}_${train_month}!"
            # TODO: possibly modify to match how you store poli_aff splits
            train_file="${poli_aff_train_dir}${train_year}_${train_month}"
            python -u ../finetuning_scripts/finetune_t5.py  \
                --model_name_or_path $MODEL \
                --dataset_name "yelp_polarity" \
                --dataset_config "plain_text" \
                --train_file $train_file \
                --do_train \
                --input_column_1 "text" \
                --output_dir "${OUT_DIR}poli_aff_${train_year}_${train_month}" \
                --seed $SEED \
                --save_steps 200 \
                --save_strategy no \
                --source_prefix_1 "text:" \
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
                --patience 3
        done
    done
fi