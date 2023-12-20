#!/bin/bash

wmt_dir="./finetuning-data/WMTdata/en/"
news_sum_dir="./tasks/datasets/newsroom/summarization/"
news_cls_dir="./tasks/datasets/newsroom/newsroom_source_classification/"
arxiv_dir="./finetuning-data/arxiv_data/"
aic_dir="./tasks/datasets/aic/"
twitter_dir="./finetuning-data/twitter_data/"
poli_aff_dir="./tasks/datasets/poli_tweets/"

model="t5-3b"
for target_year in "2015" "2016"
do
       identifier="${model}_news_cls_news_sum_2012_eval_${target_year}"
       python -m slurm_jobs.example_run_task_analogy_sweep \
              --model $model --experiment baseline \
              --num-nodes 1 --num-gpus-per-node 1 \
              --eval_output_dir ./search_outputs/${identifier} \
              --seed 42 --do_eval \
              --a_vec_dir ./${model}_vecs/news_sum/2012 \
              --b_vec_dir ./${model}_vecs/news_sum/${target_year} \
              --c_vec_dir ./${model}_vecs/news_cls/2012 \
              --eval_file ${news_cls_dir}dev/${target_year}.jsonl \
              --identifier $identifier
              #--lm --dataset "lm" --dataset_config "plain_text" 
done

#for source_year in "2006-2008" #"2006-2008" "2009-2011" "2012-2014" "2015-2017" "2018-2020"
#for source_year in 2012 #{2018..2020}
#do
       #for target_year in "2006-2008" "2009-2011" "2012-2014" "2015-2017" "2018-2020"
#       for target_year in 2016 #{2015..2019}
#       do
              #if [ $source_year != $target_year ];
              #then
#              python -m slurm_jobs.example_run_task_analogy_sweep \
#                            --model $model --experiment baseline \
#                            --num-nodes 1 --num-gpus-per-node 1 \
#                            --eval_output_dir ./search_outputs/${model}_modified_poli_aff_twitter_lm_${source_year}_${target_year} \
#                            --seed 42 --do_eval \
#                            --source_year $source_year --target_year $target_year \
#                            --task_vec_dir ./${model}_vecs/poli_aff \
#                            --lm_vec_dir ./${model}_vecs/twitter_lm \
#                            --eval_file ${poli_aff_dir}dev/${target_year}.jsonl \
              #              --lora
              #fi
#       done
#done
