#!/bin/bash

wmt_dir="./finetuning-data/WMTdata/en/"
news_sum_dir="./tasks/datasets/newsroom/summarization/"
news_cls_dir="./tasks/datasets/newsroom/newsroom_source_classification/"
arxiv_dir="./finetuning-data/arxiv_data/"
aic_dir="./tasks/datasets/aic/"
twitter_dir="./finetuning-data/twitter_data/"
poli_aff_dir="./tasks/datasets/poli_tweets/"

model="t5-3b"

#for source_year in "2012-2014" "2015-2017" "2018-2020" #"2006-2008" "2009-2011"
for source_year in 2015
do
       #for target_year in "2006-2008" "2009-2011" "2012-2014" "2015-2017" "2018-2020"
       for target_year in {2015..2020}
       do
              #if [ $source_year != $target_year ];
              #then
              python -m slurm_jobs.example_run_task_addition_sweep \
                            --model $model --experiment baseline \
                            --num-nodes 1 --num-gpus-per-node 1 \
                            --eval_output_dir ./search_outputs/${model}_poli_aff_twitter_lm_addition_${source_year}_${target_year} \
                            --seed 42 --do_eval \
                            --source_year $source_year --target_year $target_year \
                            --task_vec_dir ./${model}_vecs/poli_aff \
                            --lm_vec_dir ./${model}_vecs/twitter_lm \
                            --eval_file ${poli_aff_dir}dev/${target_year}.jsonl
                            #--lora
              #fi
       done
done