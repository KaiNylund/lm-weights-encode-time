#!/bin/bash

SEED=42
PRETRAINED_MODEL="t5-small"
ALPHA1S=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
ALPHA2S=(1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.0)

LR=0.0008
HF_MODEL="t5-60M"
interp_out_dir="${PRETRAINED_MODEL}_interp_evals/"
vec_out_dir="${PRETRAINED_MODEL}_vecs/"

# ------------------------------------------------------------------------------------------------
# WMT LM monthly interpolation experiments
# ------------------------------------------------------------------------------------------------
dataset_name="KaiNylund/WMT-year-splits"

for eval_year in {2013..2015}
do
    # TODO: replace if training from scratch
    SOURCE_MODEL1="KaiNylund/${HF_MODEL}-lm-wmt-${eval_year}_0"
    SOURCE_MODEL2="KaiNylund/${HF_MODEL}-lm-wmt-${eval_year}_11"

    # Get time vectors for eval year jan. and dec. wmt lm models
    python -u ../task_vectors/get_task_vector.py \
        --path_to_pretrained_model $PRETRAINED_MODEL \
        --path_to_finetuned_model $SOURCE_MODEL1 \
        --alpha 1.0 \
        --output_dir "${vec_out_dir}wmt_lm_${eval_year}_0_vec" \

    python -u ../task_vectors/get_task_vector.py \
        --path_to_pretrained_model $PRETRAINED_MODEL \
        --path_to_finetuned_model $SOURCE_MODEL2 \
        --alpha 1.0 \
        --output_dir "${vec_out_dir}wmt_lm_${eval_year}_11_vec" \

    # Add time vectors together with each alpha value and evaluate on all years
    for i in "${!ALPHA1S[@]}"
    do
        ALPHA1=${ALPHA1S[$i]}
        ALPHA2=${ALPHA2S[$i]}
        python -u ../task_vectors/multi_task.py \
            --path_to_source_model ${MODEL} \
            --task_vectors "${vec_out_dir}wmt_lm_${eval_year}_0_vec" \
                           "${vec_out_dir}wmt_lm_${eval_year}_11_vec" \
            --lambdas $ALPHA1 $ALPHA2 \
            --output_dir "${vec_out_dir}wmt_lm_${eval_year}_jan_dec_interp_${ALPHA1}_${ALPHA2}"

        for eval_month in {0..11}
        do
            eval_split="${eval_year}_${eval_month}_test"
            python -u ../finetuning_scripts/finetune_t5.py  \
                --model_name_or_path "${vec_out_dir}wmt_lm_${eval_year}_jan_dec_interp_${ALPHA1}_${ALPHA2}" \
                --dataset_name $dataset_name \
                --validation_split $eval_split \
                --do_eval \
                --input_column_1 "text" \
                --output_dir ${interp_out_dir}_wmt_lm_${eval_year}_jan_dec_interp_${ALPHA1}_${ALPHA2}_eval_${eval_year}_${eval_month} \
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
        done
        rm -rf "${vec_out_dir}wmt_lm_${eval_year}_jan_dec_interp_${ALPHA1}_${ALPHA2}"
    done
done