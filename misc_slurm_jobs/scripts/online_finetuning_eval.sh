#!/bin/bash

poli_aff_dir=""

TASK="poli_aff"
MODEL="t5-3b"
SEED=42
LR=0.0002

out_dir=""

mkdir -p $out_dir

for train_year in {2015..2020}
do
    for train_month in {0..11}
    do
        model_path="../${MODEL}_${TASK}_linear_updating/up_to_${train_year}_${train_month}"
        if [ ! -d $train_file ]; then
            echo "Missing model for ${train_year}_${train_month}!"
        else
            for eval_year in {2015..2020}
            do
                echo "Evaluating on ${eval_year}!"
                python -m slurm_jobs.example_run_finetune_t5_sweep --experiment baseline \
                        --model $model_path --seed $SEED \
                        --num-nodes 1 --num-gpus-per-node 1 --do_eval \
                        --eval_output_dir "${out_dir}up_to_${train_year}_${train_month}_eval_${eval_year}" \
                        --valid_file ${poli_aff_dir}dev/${eval_year}_with_month_flag.jsonl \
                        --identifier ${model}_${train_year}_${train_month}_${TASK}_linear_updating_with_month_flag_eval_${eval_year} \
                        --dataset yelp_polarity --dataset_config plain_text \
                        --lora
            done
        fi
    done
done