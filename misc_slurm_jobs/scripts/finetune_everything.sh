#!/bin/bash

wmt_dir="./finetuning-data/WMTdata/en/"
sum_dir="./tasks/datasets/newsroom/summarization/"
news_cls_dir="./tasks/datasets/newsroom/newsroom_source_classification/"

arxiv_dir="./finetuning-data/arxiv_data/"
aic_dir="./tasks/datasets/aic/"

twitter_dir="./finetuning-data/twitter_data/"
poli_aff_dir="./tasks/datasets/poli_tweets/"

model="t5-large"
model_type="large"
seed=42
lr=0.0008 #0.0002
eval_output_dir="./eval_logs/"


'''
for train_year in {2017..2020}
do
    python -m slurm_jobs.example_run_finetune_t5_sweep --experiment baseline \
        --model $model --model_type $model_type --seed $seed \
        --num-nodes 1 --num-gpus-per-node 1  --do_train \
        --eval_output_dir $eval_output_dir \
        --train_file ${wmt_dir}train_json/year_${train_year}_200000000_bytes \
        --identifier ${model}_lm_${train_year}_wmt --lm \
        --lr $lr
done

for train_file in "combined_years_2015-2020" "combined_years_2015-2020_with_year_flag" "combined_years_2015-2020_with_month_flag"
do
    python -m slurm_jobs.example_run_finetune_t5_sweep --experiment baseline \
        --model $model --model_type $model_type --seed $seed \
        --num-nodes 1 --num-gpus-per-node 1  --do_train --do_eval \
        --eval_output_dir $eval_output_dir \
        --valid_file ${twitter_dir}test/combined_years_2015-2020 \
        --train_file ${twitter_dir}train/${train_file} \
        --identifier ${model}_lm_${train_file}_twitter --lm \
        --lr $lr
done


for train_file in "combined_years_2007-2020" "combined_years_2007-2020_with_year_flag"
do
    python -m slurm_jobs.example_run_finetune_t5_sweep --experiment baseline \
        --model $model --model_type $model_type --seed $seed \
        --num-nodes 1 --num-gpus-per-node 1  --do_train --do_eval \
        --eval_output_dir $eval_output_dir \
        --valid_file ${arxiv_dir}test/combined_years_2007-2020 \
        --train_file ${arxiv_dir}train/${train_file} \
        --identifier ${model}_lm_${train_file}_arxiv --lm \
        --lr $lr
done
'''

#for year in {2015..2020}
#do
#    for month in {0..11}
#    do
#        train_file="${twitter_dir}train/year_${year}_month_${month}_30000000_bytes"
        # wmt lm
#        if [ -f $train_file ];
#        then
#            python -m slurm_jobs.example_run_finetune_t5_sweep --experiment baseline \
#                --model $model --model_type $model_type --seed $seed \
#                --num-nodes 1 --num-gpus-per-node 1  --do_train --do_eval \
#                --eval_output_dir $eval_output_dir \
#                --valid_file ${twitter_dir}test/year_${year}_month_${month}_3000000_bytes \
#                --train_file $train_file \
#                --identifier ${model}_lm_${year}_${month}_twitter --lm \
#                --lr $lr
#        fi
#    done
#done


# Finetune news domain lm and tasks
for file in "combined_years_2012-2016" "combined_years_2012-2016_with_year_flag" "combined_years_2012-2016_with_month_flag"
do
#    for month in "4"
#    do
#        train_file="${wmt_dir}train_json/year_${year}_month_${month}_30000000_bytes"
        # wmt lm
#        if [ -f $train_file ];
#        then
    train_file="${wmt_dir}train_json/${file}"
    eval_file="${wmt_dir}test_json/combined_years_2012-2016"
    python -m slurm_jobs.example_run_finetune_t5_sweep --experiment baseline \
                --model $model --model_type $model_type --seed $seed \
                --num-nodes 1 --num-gpus-per-node 1  --do_train --do_eval \
                --eval_output_dir $eval_output_dir \
                --valid_file $eval_file \
                --train_file $train_file \
                --identifier ${model}_wmt_lm_${file} \
                --lm --dataset "lm" --dataset_config "plain_text" \
                --lr $lr \
                --lora
               #--train_file ${wmt_dir}train_json/year_${year}_200000000_bytes \
#        fi
#    done

    # news sum task
    #python -m slurm_jobs.example_run_summarization_sweep --experiment baseline \
    #     --model $model --seed $seed \
    #    --num-nodes 1 --num-gpus-per-node 1 --do_train --do_eval \
    #    --eval_output_dir $eval_output_dir \
    #    --train_file ${sum_dir}train/${year}.jsonl \
    #    --valid_file ${sum_dir}dev/${year}.jsonl \
    #    --identifier ${model}_news_sum_${year} \
    #    --lora
    
    # news source classification task
    #python -m slurm_jobs.example_run_finetune_t5_sweep --experiment baseline \
    #    --model $model --model_type $model_type --seed $seed \
    #    --num-nodes 1 --num-gpus-per-node 1 --do_train --do_eval \
    #    --eval_output_dir $eval_output_dir \
    #    --train_file ${news_cls_dir}train/${year}.jsonl \
    #    --valid_file ${news_cls_dir}dev/${year}.jsonl \
    #    --identifier ${model}_news_cls_${year} \
    #    --dataset yelp_polarity --dataset_config plain_text \
    #    --lr $lr \
    #    --lora 
done


# Finetune twitter domain lm and tasks
#for year in "2015" "2016" "2017" "2018" "2019" "2020" #"combined_years"
for file in "combined_years_2015-2020" "combined_years_2015-2020_with_year_flag" "combined_years_2015-2020_with_month_flag"
do
    # twitter lm
    #if [ $year != "combined_years" ];
    #then
    train_file="${twitter_dir}train/${file}"
    eval_file="${twitter_dir}test/combined_years_2015-2020"
    python -m slurm_jobs.example_run_finetune_t5_sweep --experiment baseline \
            --model $model --model_type $model_type --seed $seed \
            --num-nodes 1 --num-gpus-per-node 1  --do_train --do_eval \
            --eval_output_dir $eval_output_dir \
            --train_file $train_file \
            --valid_file $eval_file \
            --identifier ${model}_twitter_lm_${file} \
            --lm --dataset "lm" --dataset_config "plain_text" \
            --lr $lr \
            --lora 
            #--train_file ${twitter_dir}train/year_${year}_200000000_bytes \
    #fi

#    for month in {0..11}
#    do
#train_file="${poli_aff_dir}train/indivs/combined_years_in_order_with_month_flag.jsonl"
# poli aff task
#if [ -f $train_file ];
#then
#    python -m slurm_jobs.example_run_finetune_t5_sweep --experiment baseline \
#        --model $model --model_type $model_type --seed $seed \
#        --num-nodes 1 --num-gpus-per-node 1 --do_train --do_eval \
#        --eval_output_dir $eval_output_dir \
#        --train_file $train_file \
#        --valid_file ${poli_aff_dir}dev/combined_years.jsonl \
#        --identifier ${model}_poli_aff_combined_years_in_order_with_month_flag \
#        --dataset yelp_polarity --dataset_config plain_text \
#        --lr $lr \
        #--lora 
#fi
#    done
done


# Finetune science domain lm and tasks
#for year in "combined_years_with_flag" #"2006-2008" "2009-2011" "2012-2014" "2015-2017" "2018-2020" #"combined_years"
for file in "combined_years_2007-2020" "combined_years_2007-2020_with_year_flag" "combined_years_2007-2020_with_month_flag"
do
    # arxiv lm
    #if [ $year != "combined_years" ];
    #then
    train_file="${arxiv_dir}train/${file}"
    eval_file="${arxiv_dir}test/combined_years_2007-2020"
    python -m slurm_jobs.example_run_finetune_t5_sweep --experiment baseline \
            --model $model --model_type $model_type --seed $seed \
            --num-nodes 1 --num-gpus-per-node 1 --do_train --do_eval \
            --eval_output_dir $eval_output_dir \
            --train_file $train_file \
            --valid_file $eval_file \
            --identifier ${model}_arxiv_lm_${file} \
            --lm --dataset "lm" --dataset_config "plain_text" \
            --lr $lr \
            --lora 
    #fi

    # ai source classification task
    #python -m slurm_jobs.example_run_finetune_t5_sweep --experiment baseline \
    #    --model $model --model_type $model_type --seed $seed \
    #    --num-nodes 1 --num-gpus-per-node 1 --do_train --do_eval \
    #    --eval_output_dir $eval_output_dir \
    #    --train_file ${aic_dir}train/${year}.jsonl \
    #    --valid_file ${aic_dir}dev/${year}.jsonl \
    #    --identifier ${model}_aic_${year}_seed_${seed} \
    #    --dataset yelp_polarity --dataset_config plain_text \
    #    --lr $lr \
    #    --lora
done

