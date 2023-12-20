#!/bin/bash

wmt_dir="./finetuning-data/WMTdata/en/"
sum_dir="./tasks/datasets/newsroom/summarization/"
news_cls_dir="./tasks/datasets/newsroom/newsroom_source_classification/"

arxiv_dir="./finetuning-data/arxiv_data/"
aic_dir="./tasks/datasets/aic/"

twitter_dir="./finetuning-data/twitter_data/"
poli_aff_dir="./tasks/datasets/poli_tweets/"
#train_size=200000000
#valid_size=10000000

#seed=100
#for finetune_time in "2006-2008" "2009-2011" "2012-2014" "2015-2017" "2018-2020"
#do
#    model_path="./models/t5-small_${finetune_time}_aic_seed_${seed}"
    #model_path="./models/t5-small_combined_years_aic"

#    for eval_time in "2006-2008" "2009-2011" "2012-2014" "2015-2017" "2018-2020"
#    do
#        if [ $finetune_time != $eval_time ];
#        then
#            python -m slurm_jobs.example_run_finetune_t5_sweep --experiment baseline \
#                    --model $model_path --do_eval \
#                    --num-nodes 1 --num-gpus-per-node 1 \
#                    --eval_output_dir ./evals --seed $seed \
#                    --valid_file ${aic_dir}dev/${eval_time}.jsonl \
#                    --identifier t5-small_aic_${finetune_time}_eval_${eval_time}_seed_${seed} \
#                    --dataset aic --dataset_config plain_text
#        fi
#    done
#done


for finetune_year in {2015..2020}
do
    model_path="./models/t5-small_${finetune_year}_poli_aff"
    #eval_year=$finetune_year

    for eval_year in {2015..2020}
    do
        #predicted_eval_file="./predicted_year_sum_task/t5-small_pred_data/predicted_${eval_year}_examples.jsonl"
        if [ $finetune_year != $eval_year ];
        then
            python -m slurm_jobs.example_run_finetune_t5_sweep \
                    --model $model_path --do_eval \
                    --experiment baseline --num-nodes 1 --num-gpus-per-node 1 \
                    --eval_output_dir ./eval_logs/ --seed 42 \
                    --identifier t5-small_poli_aff_${finetune_year}_eval_${eval_year} \
                    --valid_file ${poli_aff_dir}dev/${eval_year}.jsonl \
                    --dataset yelp_polarity --dataset_config plain_text
                   #--valid_file $predicted_eval_file \
        fi
    done
done

#for eval_year in {2012..2016}
#    do
        # vanilla t5 evaluation on summarization tasks
        #python -m slurm_jobs.example_run_summarization_sweep \
        #       --model t5-large \
        #       --experiment baseline --num-nodes 1 --num-gpus-per-node 1 \
        #       --eval_output_dir ./eval_logs/ --seed 42 \
        #       --valid_file ${sum_dir}dev/${eval_year}.jsonl \
        #       --identifier t5-large_wmt_sum_vanilla_eval_${eval_year} --do_eval

        #model_path="./models/t5-small_news_sum_combined_years"

#        for missing_year in {2012..2016}
#        do
#            model_path="./models/t5-small_news_sum_missing_${missing_year}"
#            python -m slurm_jobs.example_run_summarization_sweep \
#                --model $model_path \
#                --experiment baseline --num-nodes 1 --num-gpus-per-node 1 \
#                --eval_output_dir ./eval_logs/ --seed 42 \
#                --valid_file ${sum_dir}dev/${eval_year}.jsonl \
#                --identifier t5_wmt_sum_missing_year_${missing_year}_eval_${eval_year} --do_eval
#        done
#    done