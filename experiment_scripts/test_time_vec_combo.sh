#!/bin/bash

SEED=42

PRETRAINED_MODEL=$1
ALPHA1=$2
ALPHA2=$3
ALPHA3=$4
ALPHA4=$5
ALPHA5=$6
ALPHA6=$7
VEC1=$8
VEC2=$9
VEC3=${10}
VEC4=${11}
VEC5=${12}
VEC6=${13}
EVAL_FILE=${14}
OUTPUT_DIR=${15}
LM=${16}
SUMMARIZATION=${17}


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


# Combine task vectors and add to the pretrained model
python -u ../task_vectors/multi_task.py \
    --path_to_source_model ${PRETRAINED_MODEL} \
    --task_vectors $VEC1 $VEC2 $VEC3 $VEC4 $VEC5 $VEC6 \
    --lambdas $ALPHA1 $ALPHA2 $ALPHA3 $ALPHA4 $ALPHA5 $ALPHA6 \
    --output_dir ${OUTPUT_DIR}${PRETRAINED_MODEL}_combo_${ALPHA1}_${ALPHA2}_${ALPHA3}_${ALPHA4}_${ALPHA5}_${ALPHA6}

# Evaluate the time vector combo
if [ $LM == "True" ]; then
    python -u ../finetuning_scripts/finetune_t5.py \
            --model_name_or_path ${OUTPUT_DIR}${PRETRAINED_MODEL}_combo_${ALPHA1}_${ALPHA2}_${ALPHA3}_${ALPHA4}_${ALPHA5}_${ALPHA6} \
            --do_eval \
            --validation_file $EVAL_FILE \
            --input_column_1 "text" \
            --output_dir ${OUTPUT_DIR}${PRETRAINED_MODEL}_combo_${ALPHA1}_${ALPHA2}_${ALPHA3}_${ALPHA4}_${ALPHA5}_${ALPHA6}_eval \
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

elif [ $SUMMARIZATION == "True" ]; then
    python -u ../finetuning_scripts/finetune_t5_summarization.py \
            --model_name_or_path ${OUTPUT_DIR}${PRETRAINED_MODEL}_combo_${ALPHA1}_${ALPHA2}_${ALPHA3}_${ALPHA4}_${ALPHA5}_${ALPHA6} \
            --do_eval \
            --validation_file $EVAL_FILE \
            --text_column "text" \
            --summary_column "summary" \
            --output_dir ${OUTPUT_DIR}${PRETRAINED_MODEL}_combo_${ALPHA1}_${ALPHA2}_${ALPHA3}_${ALPHA4}_${ALPHA5}_${ALPHA6}_eval \
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
else
    python -u ../finetuning_scripts/finetune_t5.py  \
            --model_name_or_path ${OUTPUT_DIR}${PRETRAINED_MODEL}_combo_${ALPHA1}_${ALPHA2}_${ALPHA3}_${ALPHA4}_${ALPHA5}_${ALPHA6} \
            --dataset_name "yelp_polarity" \
            --dataset_config "plain_text" \
            --validation_file $EVAL_FILE \
            --do_eval \
            --input_column_1 "text" \
            --output_dir ${OUTPUT_DIR}${PRETRAINED_MODEL}_combo_${ALPHA1}_${ALPHA2}_${ALPHA3}_${ALPHA4}_${ALPHA5}_${ALPHA6}_eval \
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
fi

rm -rf ${OUTPUT_DIR}${PRETRAINED_MODEL}_combo_${ALPHA1}_${ALPHA2}_${ALPHA3}_${ALPHA4}_${ALPHA5}_${ALPHA6}

