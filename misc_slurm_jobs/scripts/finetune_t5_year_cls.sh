#!/bin/bash

wmt_year_cls_dir="./tasks/datasets/wmt_year_cls/"
predict_path="./tasks/datasets/newsroom/summarization/test/combined_years_only_text.jsonl"
model_dir="./models/t5-large_wmt_year_cls"

python -m slurm_jobs.example_run_finetune_t5_sweep --model $model_dir --experiment baseline \
        --num-nodes 1 --num-gpus-per-node 1 \
        --do_predict \
        --eval_output_dir ./evals --seed 42 \
        --train_file ${wmt_year_cls_dir}train.jsonl \
        --valid_file ${wmt_year_cls_dir}dev.jsonl \
        --predict_file $predict_path \
        --identifier t5-large_wmt_year_cls_pred \
        --dataset wmt_year_cls --dataset_config plain_text