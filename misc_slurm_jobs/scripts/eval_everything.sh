#!/bin/bash

wmt_dir="./finetuning-data/WMTdata/en/"
news_sum_dir="./tasks/datasets/newsroom/summarization/"
news_cls_dir="./tasks/datasets/newsroom/newsroom_source_classification/"

arxiv_dir="./finetuning-data/arxiv_data/"
aic_dir="./tasks/datasets/aic/"

twitter_dir="./finetuning-data/twitter_data/"
poli_aff_dir="./tasks/datasets/poli_tweets/"

#model="t5-3b"
model="t5-60M"
seed=42
eval_output_dir="./eval_logs/"

'''for reg_type in "ind_emb_random_forest_16"
do
    #twitter_predicted_model_path="./predicted_models/${reg_type}_poli_aff_twitter_2020"
    #python -m slurm_jobs.example_run_finetune_t5_sweep --experiment baseline \
    #        --model $twitter_predicted_model_path --seed $seed \
    #        --num-nodes 1 --num-gpus-per-node 1 --do_eval \
    #        --eval_output_dir $eval_output_dir \
    #        --valid_file ${poli_aff_dir}dev/2020.jsonl  \
    #        --dataset yelp_polarity --dataset_config plain_text \
    #        --identifier ${model}_${reg_type}_pred_poli_aff_twitter_eval_2020 

    wmt_predicted_model_path="./predicted_models/${reg_type}_news_sum_wmt_2016"
    python -m slurm_jobs.example_run_summarization_sweep --experiment baseline \
            --model $wmt_predicted_model_path --seed $seed \
            --num-nodes 1 --num-gpus-per-node 1 --do_eval \
            --eval_output_dir $eval_output_dir \
            --valid_file ${sum_dir}dev/2016.jsonl \
            --identifier ${model}_pred_news_sum_wmt_2016
done


#"combo_linreg" "combo_sgd" "combo_random_forest_2" "combo_random_forest_4" "combo_random_forest_8" "combo_random_forest_16" "combo_random_forest_32" "combo_random_forest_64" "combo_random_forest_128"
for reg_type in "random_forest_16"
do
    for eval_month in {0..11}
    do
        predicted_model_path="./predicted_models/${reg_type}_wmt_lm_2014_${eval_month}"
        python -m slurm_jobs.example_run_finetune_t5_sweep --experiment baseline \
                --model $predicted_model_path --seed $seed \
                --num-nodes 1 --num-gpus-per-node 1 --do_eval \
                --eval_output_dir $eval_output_dir \
                --valid_file ${wmt_dir}test_json/year_2014_month_${eval_month}_3000000_bytes \
                --identifier ${model}_${reg_type}_pred_lm_wmt_eval_2014_${eval_month} \
                --lm

                #--dataset yelp_polarity --dataset_config plain_text \
                #--valid_file ${poli_aff_dir}dev/2020_${eval_month}.jsonl \
                #--identifier ${model}_${reg_type}_pred_poli_aff_eval_2020_${eval_month} \
    done
done
'''


# Finetune news domain lm and tasks
for finetune_year in "combined_years" #"2012" "2013" "2014" "2015" "2016" #"combined_years"
#for wmt_model_path in $model
do
    #wmt_model_path="./models/${model}_lm_${finetune_year}_wmt"
    #wmt_model_path="KaiNylund/${model}-lm-wmt-${finetune_year}"
    wmt_model_path="KaiNylund/t5-60M-lm-wmt-2012_to_2016"
    news_sum_model_path="KaiNylund/${model}-news_sum-${finetune_year}"
    #news_sum_model_path="./models/${model}_news_sum_${finetune_year}"
    #news_cls_model_path="./models/${model}_news_cls_${finetune_year}"

    for eval_year in "combined_years" #"2012" "2013" "2014" "2015" "2016"
    do
    #if [ $finetune_year != $eval_year ];
    #then
        # wmt lm
        #python -m slurm_jobs.example_run_finetune_t5_sweep --experiment baseline \
        #        --model $wmt_model_path --seed $seed \
        #        --num-nodes 1 --num-gpus-per-node 1  --do_eval \
        #        --eval_output_dir $eval_output_dir \
        #        --valid_file ${wmt_dir}test_json/${eval_year}_2012-2016 \
        #        --identifier ${model}_lm_wmt_${finetune_year}_eval_${eval_year}  \
        #        --dataset lm --dataset_config plain_text \
        #        --lm \
        #        --lora
                #--valid_file ${wmt_dir}test_json/year_${eval_year}_10000000_bytes \
                #--identifier ${model}_lm_wmt_${finetune_year}_eval_${eval_year} \

        # news sum task
        python -m slurm_jobs.example_run_summarization_sweep --experiment baseline \
                --model $news_sum_model_path --seed $seed \
                --num-nodes 1 --num-gpus-per-node 1 --do_eval \
                --eval_output_dir $eval_output_dir \
                --valid_file ${sum_dir}dev/${eval_year}.jsonl \
                --identifier ${model}_${finetune_year}_news_sum_eval_${eval_year}
                #--lora
                #--valid_file ${sum_dir}dev/${eval_year}.jsonl \
        
        # news source classification task
        #python -m slurm_jobs.example_run_finetune_t5_sweep --experiment baseline \
        #        --model $news_cls_model_path --seed $seed \
        #        --num-nodes 1 --num-gpus-per-node 1 --do_eval \
        #        --eval_output_dir $eval_output_dir \
        #        --train_file ${news_cls_dir}train/${eval_year}.jsonl \
        #        --valid_file ${news_cls_dir}dev/${eval_year}.jsonl \
        #        --identifier ${model}_${finetune_year}_news_cls_eval_${eval_year}  \
        #        --dataset yelp_polarity --dataset_config plain_text \
        #        --lora
        #fi
    done
done

'''
# Finetune twitter domain lm and tasks
for finetune_year in "combined_years" #"2015" "2016" "2017" "2018" "2019" "2020"
#for twitter_model_path in $model
do
    #twitter_model_path="./models/${model}_lm_${finetune_year}_twitter"
    #twitter_model_path="KaiNylund/${model}-lm-twitter-${finetune_year}"
    poli_aff_model_path="KaiNylund/${model}-poli_aff-${finetune_year}"
    #poli_aff_model_path="./models/${model}_poli_aff_${finetune_year}"

    for eval_year in "combined_years" #"2015" "2016" "2017" "2018" "2019" "2020"
    do
        #if [ $finetune_year != $eval_year ];
        #then
            # twitter lm
        #python -m slurm_jobs.example_run_finetune_t5_sweep --experiment baseline \
        #        --model $twitter_model_path --seed $seed \
        #        --num-nodes 1 --num-gpus-per-node 1  --do_eval \
        #        --eval_output_dir $eval_output_dir \
        #        --valid_file ${twitter_dir}test/year_${eval_year}_10000000_bytes \
        #        --identifier ${model}_lm_${finetune_year}_twitter_eval_${eval_year} \
        #        --lm \
                #--lora
                #--identifier ${model}_lm_${finetune_year}_twitter_eval_${eval_year} \

        # poli aff task
        python -m slurm_jobs.example_run_finetune_t5_sweep --experiment baseline \
            --model $poli_aff_model_path --seed $seed \
            --num-nodes 1 --num-gpus-per-node 1 --do_eval \
            --eval_output_dir $eval_output_dir \
            --valid_file ${poli_aff_dir}dev/${eval_year}.jsonl \
            --identifier ${model}_${finetune_year}_poli_aff_eval_${eval_year} \
            --dataset yelp_polarity --dataset_config plain_text
        #    --lora
        #fi
    done
done


# Finetune science domain lm and tasks
for finetune_year in "2006-2008" "2009-2011" "2012-2014" "2015-2017" "2018-2020" #"combined_years"
#for arxiv_model_path in $model
do
    #arxiv_model_path="./models/${model}_lm_${finetune_year}_arxiv_seed_42"
    arxiv_model_path="KaiNylund/${model}-lm-arxiv-${finetune_year}"
    #aic_model_path="./models/${model}_aic_${finetune_year}_seed_${seed}"

    for eval_year in "2006-2008" "2009-2011" "2012-2014" "2015-2017" "2018-2020"
    do
        if [ $finetune_year == $eval_year ];
        then
            # arxiv lm
            python -m slurm_jobs.example_run_finetune_t5_sweep --experiment baseline \
                --model $arxiv_model_path --seed $seed \
                --num-nodes 1 --num-gpus-per-node 1  --do_eval \
                --eval_output_dir $eval_output_dir \
                --valid_file ${arxiv_dir}test/${eval_year}_15000000_bytes \
                --identifier ${model}_lm_arxiv_${finetune_year}_eval_${eval_year}_seed_${seed} \
                --lm \
                --lora
                #--identifier ${model}_lm_arxiv_${finetune_year}_eval_${eval_year}_seed_${seed} \

            # ai source classification task
            #python -m slurm_jobs.example_run_finetune_t5_sweep --experiment baseline \
            #    --model $aic_model_path --seed $seed \
            #    --num-nodes 1 --num-gpus-per-node 1 --do_eval \
            #    --eval_output_dir $eval_output_dir \
            #    --valid_file ${aic_dir}dev/${eval_year}.jsonl \
            #    --identifier ${model}_${finetune_year}_aic_eval_${eval_year} \
            #    --dataset yelp_polarity --dataset_config plain_text \
            #    --lora
        fi
    done
done
'''