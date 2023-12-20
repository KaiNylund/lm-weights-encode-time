#!/bin/bash

# TODO: Modify poli_aff path and the train file path below to finetune on PoliAff
# task from "time waits for no one!" (Luu et al., 2021)
poli_aff_train_dir=""

SEED=42
LR=0.0008
HF_MODEL="t5-60M"
OUT_DIR="./models_seed_${SEED}/"

# Evaluate WMT LM month models on all month test splits
dataset_name="KaiNylund/WMT-month-splits"
for train_year in {2012..2016}
do
    for train_month in {0..11}
    do
        # TODO: change to path if you're training from scratch
        month_model="KaiNylund/${HF_MODEL}-lm-wmt-${train_year}-${train_month}"
        for eval_year in {2012..2016}
        do
            for eval_month in {0..11}
            do
                if ([[ train_year == 2012 ]] && [[ train_month == 7 ]]) ||
                   ([[ train_year == 2016 ]] && [[ train_month == 5 ]]) ||
                   ([[ eval_year == 2012 ]] && [[ eval_month == 7 ]]) ||
                   ([[ eval_year == 2016 ]] && [[ eval_month == 6 ]]);
                then
                    echo "Missing split for WMT LM ${train_year}_${train_month}!"
                else
                    echo "Evaluating ${train_year}_${train_month} WMT LM model \
                                  on ${eval_year}_${eval_month}!"
                    eval_split="${eval_year}_${eval_month}_test"
                    python -u ../finetuning_scripts/finetune_t5.py  \
                        --model_name_or_path $month_model \
                        --dataset_name $dataset_name \
                        --validation_split $eval_split \
                        --do_eval \
                        --input_column_1 "text" \
                        --output_dir "${OUT_DIR}wmt_lm_tr${train_year}_${train_month}_ev${eval_year}_${eval_month}" \
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
    done
done


# Evaluate PoliAff month models on all month test splits
if [[ $poli_aff_train_dir != ""]]
then
    for train_year in {2015..2020}
    do
        for train_month in {0..11}
        do
            # TODO: change to path if you're training from scratch
            month_model="KaiNylund/${HF_MODEL}-poli_aff-${train_year}-${train_month}"
            for eval_year in {2015..2020}
            do
                for eval_month in {0..11}
                do
                    echo "Evaluating ${train_year}_${train_month} PoliAff model \
                                  on ${eval_year}_${eval_month}!"
                    # TODO: possibly modify to match how you store poli_aff splits
                    eval_file="${poli_aff_train_dir}${eval_year}_${eval_month}"
                    python -u ../finetuning_scripts/finetune_t5.py  \
                        --model_name_or_path $MODEL \
                        --dataset_name "yelp_polarity" \
                        --dataset_config "plain_text" \
                        --validation_file $eval_file \
                        --do_eval \
                        --input_column_1 "text" \
                        --output_dir "${OUT_DIR}poli_aff_tr${train_year}_${train_month}_ev${eval_year}_${eval_month}" \
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
        done
    done
fi