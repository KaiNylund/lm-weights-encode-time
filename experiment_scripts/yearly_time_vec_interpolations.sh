#!/bin/bash

# TODO: Modify poli_aff path and the train file path below to finetune on PoliAff
# task from "time waits for no one!" (Luu et al., 2022)
news_sum_eval_dir = ""
poli_aff_eval_dir = ""

SEED=42
PRETRAINED_MODEL=$1
ALPHA1S=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
ALPHA2S=(1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.0)

eval_out_dir="${PRETRAINED_MODEL}_interp_evals/"
vec_out_dir="${PRETRAINED_MODEL}_vecs/"
if [ $PRETRAINED_MODEL == "t5-3b" ]; then
    LR=0.0002
    LORA_PHRASE="--lora"
    HF_MODEL="t5-3b"
elif [ $PRETRAINED_MODEL == "t5-large" ]; then
    LR=0.0008
    LORA_PHRASE="--lora"
    HF_MODEL="t5-770M"
else
    LR=0.0008
    LORA_PHRASE=""
    HF_MODEL="t5-60M"
fi


# ------------------------------------------------------------------------------------------------
# WMT LM interpolation experiments
# ------------------------------------------------------------------------------------------------
dataset_name="KaiNylund/WMT-year-splits"
# TODO: replace if training from scratch
SOURCE_MODEL1="KaiNylund/${HF_MODEL}-lm-wmt-2012"
SOURCE_MODEL2="KaiNylund/${HF_MODEL}-lm-wmt-2016"
EVAL_YEARS=("2012" "2013" "2014" "2015" "2016")

# Get time vectors for 2012 and 2016 wmt lm models
python -u ../task_vectors/get_task_vector.py \
    --path_to_pretrained_model $PRETRAINED_MODEL \
    --path_to_finetuned_model $SOURCE_MODEL1 \
    --alpha 1.0 \
    --output_dir "${vec_out_dir}wmt_lm_2012_vec" \
    $LORA_PHRASE

python -u ../task_vectors/get_task_vector.py \
    --path_to_pretrained_model $PRETRAINED_MODEL \
    --path_to_finetuned_model $SOURCE_MODEL2 \
    --alpha 1.0 \
    --output_dir "${vec_out_dir}wmt_lm_2016_vec" \
    $LORA_PHRASE

# Add time vectors together with each alpha value and evaluate on all years
for i in "${!ALPHA1S[@]}"
do
    ALPHA1=${ALPHA1S[$i]}
    ALPHA2=${ALPHA2S[$i]}
    python -u ../task_vectors/multi_task.py \
        --path_to_source_model ${MODEL} \
        --task_vectors "${vec_out_dir}wmt_lm_2012_vec" \
                       "${vec_out_dir}wmt_lm_2016_vec" \
        --lambdas $ALPHA1 $ALPHA2 \
        --output_dir "${vec_out_dir}wmt_lm_2012_2016_interp_${ALPHA1}_${ALPHA2}"

    for i in "${!EVAL_YEARS[@]}"
    do
        eval_year=${EVAL_YEARS[$i]}
        eval_split="${eval_year}_test"
        python -u ../finetuning_scripts/finetune_t5.py  \
            --model_name_or_path "${vec_out_dir}wmt_lm_2012_2016_interp_${ALPHA1}_${ALPHA2}" \
            --dataset_name $dataset_name \
            --validation_split $eval_split \
            --do_eval \
            --input_column_1 "text" \
            --output_dir ${eval_out_dir}wmt_lm_2012_2016_interp_${ALPHA1}_${ALPHA2}_eval_${eval_year} \
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
            --lm \
            $LORA_PHRASE
    done
    rm -rf "${vec_out_dir}wmt_lm_2012_2016_interp_${ALPHA1}_${ALPHA2}"
done


# ------------------------------------------------------------------------------------------------
# NewsSum interpolation experiments
# ------------------------------------------------------------------------------------------------

if [[ $news_sum_eval_dir != "" ]];
then
    # TODO: replace if training from scratch
    SOURCE_MODEL1="KaiNylund/${HF_MODEL}-news_sum-2012"
    SOURCE_MODEL2="KaiNylund/${HF_MODEL}-news_sum-2016"
    EVAL_YEARS=("2012" "2013" "2014" "2015" "2016")

    # Get time vectors for 2012 and 2016 newsSum models
    python -u ../task_vectors/get_task_vector.py \
        --path_to_pretrained_model $PRETRAINED_MODEL \
        --path_to_finetuned_model $SOURCE_MODEL1 \
        --alpha 1.0 \
        --output_dir "${vec_out_dir}news_sum_2012_vec" \
        $LORA_PHRASE

    python -u ../task_vectors/get_task_vector.py \
        --path_to_pretrained_model $PRETRAINED_MODEL \
        --path_to_finetuned_model $SOURCE_MODEL2 \
        --alpha 1.0 \
        --output_dir "${vec_out_dir}news_sum_2016_vec" \
        $LORA_PHRASE

    # Add time vectors together with each alpha value and evaluate on all years
    for i in "${!ALPHA1S[@]}"
    do
        ALPHA1=${ALPHA1S[$i]}
        ALPHA2=${ALPHA2S[$i]}
        python -u ../task_vectors/multi_task.py \
            --path_to_source_model ${MODEL} \
            --task_vectors "${vec_out_dir}news_sum_2012_vec" \
                        "${vec_out_dir}news_sum_2016_vec" \
            --lambdas $ALPHA1 $ALPHA2 \
            --output_dir "${vec_out_dir}news_sum_2012_2016_interp_${ALPHA1}_${ALPHA2}"

        for i in "${!EVAL_YEARS[@]}"
        do
            eval_year=${EVAL_YEARS[$i]}
            # TODO: possibly change based on news_sum split format
            eval_file="${news_sum_eval_dir}${eval_year}"
            python -u ../finetuning_scripts/finetune_t5_summarization.py \
                    --model_name_or_path "${vec_out_dir}news_sum_2012_2016_interp_${ALPHA1}_${ALPHA2}" \
                    --do_eval \
                    --validation_file $eval_file \
                    --text_column "text" \
                    --summary_column "summary" \
                    --output_dir ${eval_out_dir}_news_sum_2012_2016_interp_${ALPHA1}_${ALPHA2}_eval_${eval_year} \
                    --seed $SEED \
                    --save_steps 200 \
                    --save_strategy no \
                    --learning_rate $LR \
                    --gradient_accumulation_steps 8 \
                    --ddp_find_unused_parameters False \
                    --per_device_train_batch_size 2 \
                    --per_device_eval_batch_size 2 \
                    --predict_with_generate \
                    --overwrite_output_dir \
                    $LORA_PHRASE
        done
        rm -rf "${vec_out_dir}news_sum_2012_2016_interp_${ALPHA1}_${ALPHA2}"
    done
fi


# ------------------------------------------------------------------------------------------------
# PoliAff interpolation experiments
# ------------------------------------------------------------------------------------------------

if [[ $poli_aff_eval_dir != "" ]];
then
    # TODO: replace if training from scratch
    SOURCE_MODEL1="KaiNylund/${HF_MODEL}-poli_aff-2015"
    SOURCE_MODEL2="KaiNylund/${HF_MODEL}-poli_aff-2019"
    EVAL_YEARS=("2015" "2016" "2017" "2018" "2019")

    # Get time vectors for 2015 and 2019 poliAff models
    python -u ../task_vectors/get_task_vector.py \
        --path_to_pretrained_model $PRETRAINED_MODEL \
        --path_to_finetuned_model $SOURCE_MODEL1 \
        --alpha 1.0 \
        --output_dir "${vec_out_dir}poli_aff_2015_vec" \
        $LORA_PHRASE

    python -u ../task_vectors/get_task_vector.py \
        --path_to_pretrained_model $PRETRAINED_MODEL \
        --path_to_finetuned_model $SOURCE_MODEL2 \
        --alpha 1.0 \
        --output_dir "${vec_out_dir}poli_aff_2019_vec" \
        $LORA_PHRASE

    # Add time vectors together with each alpha value and evaluate on all years
    for i in "${!ALPHA1S[@]}"
    do
        ALPHA1=${ALPHA1S[$i]}
        ALPHA2=${ALPHA2S[$i]}
        python -u ../task_vectors/multi_task.py \
            --path_to_source_model ${MODEL} \
            --task_vectors "${vec_out_dir}poli_aff_2015_vec" \
                           "${vec_out_dir}poli_aff_2019_vec" \
            --lambdas $ALPHA1 $ALPHA2 \
            --output_dir "${vec_out_dir}poli_aff_2015_2019_interp_${ALPHA1}_${ALPHA2}"

        for i in "${!EVAL_YEARS[@]}"
        do
            eval_year=${EVAL_YEARS[$i]}
            # TODO: possibly change based on poliAff split format
            eval_file="${poli_aff_eval_dir}${eval_year}"
            python -u ../finetuning_scripts/finetune_t5.py  \
                --model_name_or_path "${vec_out_dir}poli_aff_2015_2019_interp_${ALPHA1}_${ALPHA2}" \
                --dataset_name "yelp_polarity" \
                --dataset_config "plain_text" \
                --validation_file $eval_file \
                --do_eval \
                --input_column_1 "text" \
                --output_dir ${eval_out_dir}_poli_aff_2015_2019_interp_${ALPHA1}_${ALPHA2}_eval_${eval_year} \
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
                --patience 3 \
                $LORA_PHRASE
        done
        rm -rf "${vec_out_dir}poli_aff_2015_2019_interp_${ALPHA1}_${ALPHA2}"
    done
fi