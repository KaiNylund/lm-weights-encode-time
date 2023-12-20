#!/bin/bash

# TODO: Modify these paths and the train file paths below to finetune on tasks
# from "time waits for no one!" (Luu et al., 2022)
news_sum_train_dir=""
news_cls_train_dir=""
poli_aff_train_dir=""
aic_train_dir=""

MODEL=$1
SEED=42
OUT_DIR="./models_seed_${SEED}/"

if [[ $MODEL == "t5-3b" ]]; then
    LR=0.0002
    LORA_PHRASE="--lora"
elif [[ $MODEL == "t5-large" ]]; then
    LR=0.0008
    LORA_PHRASE="--lora"
else
    LR=0.0008
    LORA_PHRASE=""
fi


# Finetune WMT LM year models
dataset_name="KaiNylund/WMT-year-splits"
for train_year in {2012..2016}
do
    echo "Training on WMT LM ${train_year}!"
    train_split="${train_year}_train"
    python -u ../finetuning_scripts/finetune_t5.py  \
        --model_name_or_path $MODEL \
        --dataset_name $dataset_name \
        --train_split $train_split \
        --do_train \
        --input_column_1 "text" \
        --output_dir "${OUT_DIR}wmt_lm_${train_year}" \
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


# Finetune NewsSum year models
if [[ $news_sum_train_dir != ""]]
then
    for train_year in {2012..2016}
    do
        echo "Training on newsroom summarization ${train_year}!"
        # TODO: possibly modify to match how you store news_sum splits
        train_file="${news_sum_train_dir}${train_year}"
        python -u ../finetuning_scripts/finetune_t5_summarization.py \
            --model_name_or_path $MODEL \
            --do_train \
            --train_file $train_file \
            --text_column "text" \
            --summary_column "summary" \
            --output_dir "${OUT_DIR}news_sum_${train_year}" \
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
fi


# Finetune NewsCls year models
if [[ $news_cls_dir != ""]]
then
    for train_year in {2012..2016}
    do
        echo "Training on newsroom source classification ${train_year}!"
        # TODO: possibly modify to match how you store news_cls splits
        train_file="${news_cls_dir}${train_year}"
        python -u ../finetuning_scripts/finetune_t5.py  \
            --model_name_or_path $MODEL \
            --dataset_name "yelp_polarity" \
            --dataset_config "plain_text" \
            --train_file $train_file \
            --do_train \
            --input_column_1 "text" \
            --output_dir "${OUT_DIR}news_cls_${train_year}" \
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
fi

# Finetune PoliAff year models
if [[ $poli_aff_train_dir != ""]]
then
    for train_year in {2015..2020}
    do
        echo "Training on PoliAff ${train_year}!"
        # TODO: possibly modify to match how you store poli_aff splits
        train_file="${poli_aff_train_dir}${train_year}"
        python -u ../finetuning_scripts/finetune_t5.py  \
            --model_name_or_path $MODEL \
            --dataset_name "yelp_polarity" \
            --dataset_config "plain_text" \
            --train_file $train_file \
            --do_train \
            --input_column_1 "text" \
            --output_dir "${OUT_DIR}poli_aff_${train_year}" \
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
fi

# Finetune AIC year models
if [[ $aic_train_dir != ""]]
then
    for train_year in "2006-2008" "2009-2011" "2012-2014" "2015-2017" "2018-2020"
    do
        echo "Training on AI venue classification ${train_year}!"
        # TODO: possibly modify to match how you store aic splits
        train_file="${aic_train_dir}${train_year}"
        python -u ../finetuning_scripts/finetune_t5.py  \
            --model_name_or_path $MODEL \
            --dataset_name "yelp_polarity" \
            --dataset_config "plain_text" \
            --train_file $train_file \
            --do_train \
            --input_column_1 "text" \
            --output_dir "${OUT_DIR}aic_${train_year}" \
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
fi
