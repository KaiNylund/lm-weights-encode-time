#!/bin/bash

wmt_dir="./finetuning-data/WMTdata/en/"

#eval_file="./tasks/datasets/newsroom/newsroom_source_classification/dev/combined_years.jsonl"
#eval_file="./tasks/datasets/poli_tweets/dev/2020.jsonl"


seed=42
model="t5-small"
task="news_sum"
output_dir="./combo_outputs/${model}_${task}_combo_multitask/"
task_vec_path="./${model}_vecs/"

#[0.3333333333333333, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.0]

eval_file="./tasks/datasets/newsroom/summarization/dev/combined_years.jsonl"
#eval_file="./tasks/datasets/poli_tweets/dev/combined_years.jsonl"
#eval_file="${wmt_dir}test_json/combined_years_2012-2016"
python -m slurm_jobs.test_time_vec_combinations \
        --model $model --do_eval --seed $seed \
        --experiment baseline --num-nodes 1 --num-gpus-per-node 1 \
        --eval_output_dir $output_dir \
        --alpha1 0.3333333333333333 \
        --alpha2 0.16666666666666666 \
        --alpha3 0.16666666666666666 \
        --alpha4 0.16666666666666666 \
        --alpha5 0.16666666666666666 \
        --alpha6 0.0 \
        --vec1 "${task_vec_path}${task}/2012" \
        --vec2 "${task_vec_path}${task}/2013" \
        --vec3 "${task_vec_path}${task}/2014" \
        --vec4 "${task_vec_path}${task}/2015" \
        --vec5 "${task_vec_path}${task}/2016" \
        --vec6 "${task_vec_path}${task}/2016" \
        --eval_file $eval_file \
        --task "${task}_combo" \
        --summarization