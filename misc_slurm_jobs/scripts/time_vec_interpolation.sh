#!/bin/bash
#SBATCH --job-name=eval-t5-task-vecs-interps
#SBATCH --account=ark
#SBATCH --partition=gpu-titan
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
# -C 'a40|a100'

wmt_dir="./finetuning-data/WMTdata/en/"
news_sum_dir="./tasks/datasets/newsroom/summarization/"
news_cls_dir="./tasks/datasets/newsroom/newsroom_source_classification/"

arxiv_dir="./finetuning-data/arxiv_data/"
aic_dir="./tasks/datasets/aic/"

twitter_dir="./finetuning-data/twitter_data/"
poli_aff_dir="./tasks/datasets/poli_tweets/"


TASK="wmt_lm"
SOURCE_TIME1="2015_0"
SOURCE_TIME2="2015_11"
SEED=42

ALPHA1S=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
ALPHA2S=(1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.0)

for MODEL in "t5-small" #"t5-small" "t5-large" "t5-3b"
do
    if [ $MODEL == "t5-3b" ]; then
        LR=0.0002
        LORA_PHRASE="--lora"
    elif [ $MODEL == "t5-large" ]; then
        LR=0.0008
        LORA_PHRASE="--lora"
    else 
        LR=0.0008
        LORA_PHRASE=""
    fi

    for i in {0..10}
    do
        ALPHA1=${ALPHA1S[$i]}
        ALPHA2=${ALPHA2S[$i]}
        python -u ./task_vectors/multi_task.py \
            --path_to_source_model ${MODEL} \
            --task_vectors ./${MODEL}_vecs/${TASK}/${SOURCE_TIME1}/ \
                        ./${MODEL}_vecs/${TASK}/${SOURCE_TIME2}/ \
            --lambdas $ALPHA1 $ALPHA2 \
            --output_dir ./interp_outputs/${MODEL}_${TASK}_interp_${SOURCE_TIME1}_${SOURCE_TIME2}_${ALPHA1}_${ALPHA2}

        for eval_year in 2015 #"2006-2008" "2009-2011" "2012-2014" "2015-2017" "2018-2020"
        do
            #eval_file="${sum_dir}dev/${eval_year}.jsonl"
            #python -m slurm_jobs.example_run_summarization_sweep --experiment baseline \
            #        --model ./interp_outputs/${MODEL}_${TASK}_interp_${SOURCE_TIME1}_${SOURCE_TIME2}_${ALPHA1}_${ALPHA2} \
            #        --seed $SEED \
            #        --num-nodes 1 --num-gpus-per-node 1 --do_eval \
            #        --eval_output_dir ./interp_outputs/${MODEL}_${TASK}_interp_${SOURCE_TIME1}_${SOURCE_TIME2}/eval_{$LABEL}_ALPHA1_${ALPHA1}_ALPHA2_${ALPHA2}_SEED=$SEED \
            #        --valid_file $eval_file \
            #        --identifier ${MODEL}_${TASK}_interp_${SOURCE_TIME1}_${SOURCE_TIME2}_${ALPHA1}_${ALPHA2}_eval_{$eval_year}
            
            #python -u ./scripts/run_summarization.py \
            #    --model_name_or_path ./interp_outputs/${MODEL}_${TASK}_interp_${SOURCE_TIME1}_${SOURCE_TIME2}_${ALPHA1}_${ALPHA2} \
            #    --validation_file $EVAL_FILE \
            #    --do_eval \
            #    --source_prefix 'summarize:' \
            #    --output_dir ./interp_outputs/${MODEL}_${TASK}_interp_${SOURCE_TIME1}_${SOURCE_TIME2}/eval_{$LABEL}_ALPHA1_${ALPHA1}_ALPHA2_${ALPHA2}_SEED=$SEED \
            #    --seed $SEED \
            #    --learning_rate $LR \
            #    --save_strategy no \
            #    --num_train_epochs 3  \
            #    --ddp_find_unused_parameters False \
            #    --per_device_train_batch_size 2 \
            #    --per_device_eval_batch_size 2 \
            #    --predict_with_generate \
            #    --overwrite_output_dir \
            #    $LORA_PHRASE
            
            #eval_file=${twitter_dir}test/year_${eval_year}_10000000_bytes
            #eval_file=${arxiv_dir}test/${eval_year}_15000000_bytes
            #eval_file="${wmt_dir}test_json/year_${eval_year}_10000000_bytes"
            #python -m slurm_jobs.example_run_finetune_t5_sweep  \
            #    --experiment baseline \
            #    --model ./interp_outputs/${MODEL}_${TASK}_interp_${SOURCE_TIME1}_${SOURCE_TIME2}_${ALPHA1}_${ALPHA2} \
            #    --seed $SEED \
            #    --num-nodes 1 --num-gpus-per-node 1  --do_eval \
            #    --eval_output_dir ./interp_outputs/${MODEL}_${TASK}_interp_${SOURCE_TIME1}_${SOURCE_TIME2}/eval_{$eval_year}_ALPHA1_${ALPHA1}_ALPHA2_${ALPHA2}_SEED=$SEED \
            #    --valid_file $eval_file \
            #    --identifier ${MODEL}_${TASK}_interp_${SOURCE_TIME1}_${SOURCE_TIME2}_${ALPHA1}_${ALPHA2}_eval_{$eval_year} \
            #    --lm --dataset "lm" --dataset_config "plain_text"

            for eval_month in {0..11}
            do
                eval_file="${wmt_dir}test_json/year_${eval_year}_month_${eval_month}_3000000_bytes"
                python -m slurm_jobs.example_run_finetune_t5_sweep  \
                    --experiment baseline \
                    --model ./interp_outputs/${MODEL}_${TASK}_interp_${SOURCE_TIME1}_${SOURCE_TIME2}_${ALPHA1}_${ALPHA2} \
                    --seed $SEED \
                    --num-nodes 1 --num-gpus-per-node 1  --do_eval \
                    --eval_output_dir ./interp_outputs/${MODEL}_${TASK}_interp_${SOURCE_TIME1}_${SOURCE_TIME2}/eval_{$eval_year}_${eval_month}_ALPHA1_${ALPHA1}_ALPHA2_${ALPHA2}_SEED=$SEED \
                    --valid_file $eval_file \
                    --identifier ${MODEL}_${TASK}_interp_${SOURCE_TIME1}_${SOURCE_TIME2}_${ALPHA1}_${ALPHA2}_eval_{$eval_year}_${eval_month} \
                    --lm --dataset "lm" --dataset_config "plain_text"
            done
        done

        #rm -rf ./interp_outputs/${MODEL}_${TASK}_interp_${SOURCE_TIME1}_${SOURCE_TIME2}_${ALPHA1}_${ALPHA2}
    done
done
