#!/bin/bash

# TODO: Modify these paths and the test file paths below to evaluate on tasks
# from "time waits for no one!" (Luu et al., 2021)
news_sum_eval_dir=""
news_cls_eval_dir=""
poli_aff_eval_dir=""
aic_eval_dir=""

MODEL=$1
SEED=42
OUT_DIR="./evaluations_seed_${SEED}/"

if [[ $MODEL == "t5-3b" ]]; then
    LR=0.0002
    LORA_PHRASE="--lora"
    HF_MODEL="t5-3b"
elif [[ $MODEL == "t5-large" ]]; then
    LR=0.0008
    LORA_PHRASE="--lora"
    HF_MODEL="t5-770M"
else
    LR=0.0008
    LORA_PHRASE=""
    HF_MODEL="t5-60M"
fi


# Evaluate WMT LM year models on all year test splits
dataset_name="KaiNylund/WMT-year-splits"
for train_year in {2012..2016}
do
    # TODO: change to path if you're training from scratch
    year_model="KaiNylund/${HF_MODEL}-lm-wmt-${train_year}"
    for eval_year in {2012..2016}
    do
        echo "Evaluating ${train_year} WMT LM model on ${eval_year}!"
        eval_split="${train_year}_test"
        python -u finetune_t5.py  \
            --model_name_or_path $year_model \
            --dataset_name $dataset_name \
            --validation_split $eval_split \
            --do_eval \
            --input_column_1 "text" \
            --output_dir "${OUT_DIR}wmt_lm_tr_${train_year}_ev_${eval_year}" \
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
done


# Evaluate NewsSum year models on all year test splits
if [[ $news_sum_eval_dir != ""]]
then
    for train_year in {2012..2016}
    do
        # TODO: change to path if you're training from scratch
        year_model="KaiNylund/${HF_MODEL}-news_sum-${train_year}"
        for eval_year in {2012..2016}
        do
            echo "Evaluating ${train_year} NewsSum model on ${eval_year}!"
            # TODO: possibly modify to match how you store news_sum splits
            eval_file="${news_sum_eval_dir}${eval_year}"
            python -u finetune_t5_summarization.py \
                --model_name_or_path $year_model \
                --do_eval \
                --validation_file $eval_file \
                --text_column "text" \
                --summary_column "summary" \
                --output_dir "${OUT_DIR}news_sum_tr_${train_year}_ev_${eval_year}" \
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
    done
fi


# Evaluate NewsCls year models on all year test splits
if [[ $news_cls_eval_dir != ""]]
then
    for train_year in {2012..2016}
    do
        # TODO: change to path if you're training from scratch
        year_model="KaiNylund/${HF_MODEL}-news_cls-${train_year}"
        for eval_year in {2012..2016}
        do
            echo "Evaluating ${train_year} NewsCls model on ${eval_year}!"
            # TODO: possibly modify to match how you store news_cls splits
            eval_file="${news_cls_eval_dir}${eval_year}"
            python -u finetune_t5.py  \
                --model_name_or_path $MODEL \
                --dataset_name "yelp_polarity" \
                --dataset_config "plain_text" \
                --validation_file $eval_file \
                --do_eval \
                --input_column_1 "text" \
                --output_dir "${OUT_DIR}news_cls_tr_${train_year}_ev_${eval_year}" \
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
    done
fi

# Evaluate PoliAff year models on all year test splits
if [[ $poli_aff_eval_dir != ""]]
then
    for train_year in {2015..2020}
    do
        # TODO: change to path if you're training from scratch
        year_model="KaiNylund/${HF_MODEL}-poli_aff-${train_year}"
        for eval_year in {2015..2020}
        do
            echo "Evaluating ${train_year} PoliAff model on ${eval_year}!"
            # TODO: possibly modify to match how you store news_cls splits
            eval_file="${poli_aff_eval_dir}${eval_year}"
            python -u ../finetuning_scripts/finetune_t5.py  \
                --model_name_or_path $MODEL \
                --dataset_name "yelp_polarity" \
                --dataset_config "plain_text" \
                --validation_file $eval_file \
                --do_eval \
                --input_column_1 "text" \
                --output_dir "${OUT_DIR}poli_aff_tr_${train_year}_ev_${eval_year}" \
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
    done
fi

# Evaluate AIC year models on all year test splits
if [[ $aic_eval_dir != ""]]
then
    for train_year in "2006-2008" "2009-2011" "2012-2014" "2015-2017" "2018-2020"
    do
        # TODO: change to path if you're training from scratch
        year_model="KaiNylund/${HF_MODEL}-aic-${train_year}"
        for eval_year in "2006-2008" "2009-2011" "2012-2014" "2015-2017" "2018-2020"
        do
            echo "Evaluating ${train_year} NewsCls model on ${eval_year}!"
            # TODO: possibly modify to match how you store news_cls splits
            eval_file="${aic_eval_dir}${eval_year}"
            python -u ../finetuning_scripts/finetune_t5.py  \
                --model_name_or_path $MODEL \
                --dataset_name "yelp_polarity" \
                --dataset_config "plain_text" \
                --validation_file $eval_file \
                --do_eval \
                --input_column_1 "text" \
                --output_dir "${OUT_DIR}aic_tr_${train_year}_ev_${eval_year}" \
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
    done
fi
