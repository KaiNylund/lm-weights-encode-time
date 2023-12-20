#!/bin/bash
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1

wmt_dir="./finetuning-data/WMTdata/en/"
poli_aff_dir="./tasks/datasets/poli_tweets/"

model="t5-60M"
seed=42
#eval_output_dir="./poli_aff_months_eval/"
eval_output_dir="./wmt_months_eval/"


'''
missing_train_years=(2016 2016 2016 2016 2016 2016 2017 2017)
missing_train_months=(5 5 5 5 7 7 6 6)
missing_eval_years=(2015 2015 2015 2016 2015 2015 2015 2015)
missing_eval_months=(4 5 6 1 2 4 4 9)

for i in ${!missing_train_years[@]}
do
    finetune_year=${missing_train_years[$i]}
    finetune_month=${missing_train_months[$i]}
    eval_year=${missing_eval_years[$i]}
    eval_month=${missing_eval_months[$i]}
    month_model_path="KaiNylund/${model}-poli_aff-${finetune_year}-${finetune_month}"
    #month_model_path="KaiNylund/${model}-lm-wmt-${finetune_year}-${finetune_month}"
    eval_file=${poli_aff_dir}dev/${eval_year}_${eval_month}.jsonl
    #eval_file=${wmt_dir}test_json/year_${eval_year}_month_${eval_month}_3000000_bytes

    #python -m slurm_jobs.example_run_finetune_t5_sweep --experiment baseline \
    #        --model $month_model_path --seed $seed \
    #        --num-nodes 1 --num-gpus-per-node 1  --do_eval \
    #        --eval_output_dir ${eval_output_dir}train_${finetune_year}_${finetune_month}_eval_${eval_year}_${eval_month} \
    #        --valid_file $eval_file \
    #        --identifier t5-small_wmt_lm_${finetune_year}_${finetune_month}_eval_${eval_year}_${eval_month} \
    #        --lm \

    python -m slurm_jobs.example_run_finetune_t5_sweep --experiment baseline \
            --model $month_model_path --seed $seed \
            --num-nodes 1 --num-gpus-per-node 1 --do_eval \
            --eval_output_dir ${eval_output_dir}train_${finetune_year}_${finetune_month}_eval_${eval_year}_${eval_month} \
            --valid_file $eval_file \
            --identifier t5-small_poli_aff_${finetune_year}_${finetune_month}_eval_${eval_year}_${eval_month} \
            --dataset yelp_polarity --dataset_config plain_text
done



# Finetune news domain lm and tasks
#for finetune_year in 2018 #{2012..2021}
#do
#    for finetune_month in {0..11}
#    do
        #month_model_path="KaiNylund/${model}-poli_aff-${finetune_year}-${finetune_month}"
        #month_model_path="KaiNylund/${model}-lm-wmt-${finetune_year}-${finetune_month}"

'''

month_model_path="KaiNylund/${model}-lm-wmt-2021-11"
for eval_year in {2012..2021}
do
    for eval_month in {0..11}
    do
        #eval_file=${poli_aff_dir}dev/${eval_year}_${eval_month}.jsonl
        eval_file=${wmt_dir}test_json/year_${eval_year}_month_${eval_month}_3000000_bytes

        #python -m slurm_jobs.example_run_finetune_t5_sweep --experiment baseline \
        #    --model $month_model_path --seed $seed \
        #    --num-nodes 1 --num-gpus-per-node 1 --do_eval \
        #    --eval_output_dir ${eval_output_dir}train_${finetune_year}_${finetune_month}_eval_${eval_year}_${eval_month} \
        #    --valid_file $eval_file \
        #    --identifier t5-small_poli_aff_${finetune_year}_${finetune_month}_eval_${eval_year}_${eval_month} \
        #    --dataset yelp_polarity --dataset_config plain_text \

        # wmt lm
        python -m slurm_jobs.example_run_finetune_t5_sweep --experiment baseline \
                --model $month_model_path --seed $seed \
                --num-nodes 1 --num-gpus-per-node 1  --do_eval \
                --eval_output_dir ${eval_output_dir}train_2021_11_eval_${eval_year}_${eval_month} \
                --valid_file $eval_file \
                --identifier t5-small_wmt_lm_2021_11_eval_${eval_year}_${eval_month} \
                --lm \
                #--identifier t5-small_wmt_lm_${finetune_year}_${finetune_month}_eval_${eval_year}_${eval_month} \
                #--eval_output_dir ${eval_output_dir}train_${finetune_year}_${finetune_month}_eval_${eval_year}_${eval_month} \
        
        #python -u ./scripts/run_finetune_t5.py  \
        #        --model_name_or_path $month_model_path \
        #        --validation_file $eval_file \
        #        --lm \
        #        --dataset_name "yelp_polarity" \
        #        --dataset_config "plain_text" \
        #        --do_eval \
        #        --input_column_1 "text" \
        #        --output_dir ${eval_output_dir}t5-small_train_${finetune_year}_${finetune_month}_eval_${eval_year}_${eval_month} \
        #        --seed 42 \
        #        --save_steps 200 \
        #        --save_strategy no \
        #        --source_prefix_1 "lm:" \
        #        --target_label label \
        #        --learning_rate 0.0008 \
        #        --max_predict_samples 1000 \
        #        --max_source_length 128 \
        #        --max_target_length 128 \
        #        --gradient_accumulation_steps 8 \
        #        --ddp_find_unused_parameters False \
        #        --per_device_train_batch_size 2 \
        #        --per_device_eval_batch_size 2 \
        #        --predict_with_generate \
        #        --patience 3 \          
    done
done
#    done
#done
