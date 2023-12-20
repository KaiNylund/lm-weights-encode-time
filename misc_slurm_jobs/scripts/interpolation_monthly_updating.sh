#!/bin/bash
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --time=120:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#-C 'a40|a100|titan'

wmt_dir="./finetuning-data/WMTdata/en/"
news_sum_dir="./tasks/datasets/newsroom/summarization/"
news_cls_dir="./tasks/datasets/newsroom/newsroom_source_classification/"

arxiv_dir="./finetuning-data/arxiv_data/"
aic_dir="./tasks/datasets/aic/"

twitter_dir="./finetuning-data/twitter_data/"
poli_aff_dir="./tasks/datasets/poli_tweets/"

TASK="news_sum"
MODEL="t5-small"
SEED=42

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

eval_out_dir="./search_outputs/${MODEL}_${TASK}_interp_fixed_monthly_updating/"
tv_out_dir="./interp_outputs/${MODEL}_${TASK}_interp_fixed_monthly_updating"
out_dir="./models/${MODEL}_${TASK}_interp_fixed_monthly_updating/"
mkdir -p $out_dir

n_pct_pts=1
prev_ckpt=$MODEL
for train_year in {2012..2016}
do
    for train_month in {0..11}
    do
        #train_file="${poli_aff_dir}train/indivs/${train_year}_${train_month}.jsonl"
        #train_file="${twitter_dir}train/year_${train_year}_month_${train_month}_30000000_bytes"
        #train_file="${wmt_dir}train_json/year_${train_year}_month_${train_month}_30000000_bytes"
        train_file="${news_sum_dir}train/${train_year}_${train_month}.jsonl"

        if [ ! -f $train_file ]; then
            echo "Missing train file for ${train_year}_${train_month}!"
            echo $train_file
        else
            echo "Training on ${train_year}_${train_month}!"

            cur_out_dir="${out_dir}up_to_${train_year}_${train_month}_interp"
            #python -u ./scripts/run_finetune_t5.py \
            #    --model_name_or_path $prev_ckpt \
            #    --train_file $train_file \
            #    --dataset_name "yelp_polarity" \
            #    --dataset_config "plain_text" \
            #    --do_train \
            #    --input_column_1 "text" \
            #    --output_dir $cur_out_dir \
            #    --seed $SEED \
            #    --save_steps 200 \
            #    --save_strategy no \
            #    --source_prefix_1 "lm:" \
            #    --target_label label \
            #    --learning_rate $LR \
            #    --max_predict_samples 1000 \
            #    --max_source_length 128 \
            #    --max_target_length 128 \
            #    --gradient_accumulation_steps 8 \
            #    --ddp_find_unused_parameters False \
            #    --per_device_train_batch_size 2 \
            #    --per_device_eval_batch_size 2 \
            #    --predict_with_generate \
            #    --patience 3 \
            #    --overwrite_output_dir \
            #    $LORA_PHRASE
            
            # Train with summarization objetive
            python -u ./scripts/run_summarization.py \
            --model_name_or_path $prev_ckpt \
            --train_file $train_file \
            --do_train \
            --text_column "text" \
            --summary_column "summary" \
            --output_dir $cur_out_dir \
            --seed $SEED \
            --save_steps 200 \
            --save_strategy no \
            --learning_rate $LR \
            --gradient_accumulation_steps 8 \
            --ddp_find_unused_parameters False \
            --per_device_train_batch_size 2 \
            --per_device_eval_batch_size 2 \
            --predict_with_generate \
            --overwrite_output_dir \
            $LORA_PHRASE


            # Train with langauge modeling objetive
            #python -u ./scripts/run_finetune_t5.py \
            #    --model_name_or_path $prev_ckpt \
            #    --train_file $train_file \
            #    --do_train \
            #    --input_column_1 "text" \
            #    --input_column_2 "text" \
            #    --output_dir $cur_out_dir \
            #    --seed $SEED \
            #    --save_steps 1000 \
            #    --save_strategy no \
            #    --learning_rate $LR \
            #    --save_total_limit 1 \
            #    --max_predict_samples 1000 \
            #    --max_source_length 128 \
            #    --max_target_length 128 \
            #    --gradient_accumulation_steps 8 \
            #    --ddp_find_unused_parameters False \
            #    --per_device_train_batch_size 2 \
            #    --per_device_eval_batch_size 2 \
            #    --predict_with_generate \
            #    --patience 3 \
            #    --overwrite_output_dir \
            #    --num_train_epochs 1 \
            #    --lm \
            #    $LORA_PHRASE

            '''
            #if [ "$prev_ckpt" != "$MODEL" ]; then
            python -m task_vectors.get_task_vector \
                --path_to_pretrained_model $MODEL \
                --path_to_finetuned_model $prev_ckpt \
                --alpha 1.0 \
                --output_dir "${tv_out_dir}_tv1" \
                $LORA_PHRASE

            python -m task_vectors.get_task_vector \
                --path_to_pretrained_model $MODEL \
                --path_to_finetuned_model $cur_out_dir \
                --alpha 1.0 \
                --output_dir "${tv_out_dir}_tv2" \
                $LORA_PHRASE

            ALPHA1=0.5
            ALPHA2=0.5
            interp_output_dir="${cur_out_dir}_ALPHA1_${ALPHA1}_ALPHA2_${ALPHA2}"
            #interp_output_dir="${cur_out_dir}_m${n_pct_pts}"
            python -u ./task_vectors/multi_task.py \
                --path_to_source_model ${MODEL} \
                --task_vectors ${tv_out_dir}_tv1/ \
                               ${tv_out_dir}_tv2/ \
                --output_dir $interp_output_dir \
                --lambdas $ALPHA1 $ALPHA2
                #--inv_decay
                #--n_pct_pts $n_pct_pts
            '''

            interp_output_dir=$cur_out_dir
            for eval_year in {2012..2016}
            do
                echo "Evaluating on ${eval_year}!"
                #eval_file=${poli_aff_dir}dev/${eval_year}.jsonl
                #eval_file=${wmt_dir}test_json/year_${eval_year}_10000000_bytes
                #python -m slurm_jobs.example_run_finetune_t5_sweep --experiment baseline \
                #        --model $interp_output_dir --seed $SEED \
                #        --num-nodes 1 --num-gpus-per-node 1 --do_eval \
                #        --eval_output_dir ${eval_out_dir}up_to_${train_year}_${train_month}_eval_${eval_year} \
                #        --valid_file $eval_file \
                #        --identifier ${MODEL}_${TASK}_up_to_${train_year}_${train_month}_eval_${eval_year} \
                #        --lm --dataset "lm" --dataset_config "plain_text" \
                #        $LORA_PHRASE
                        #--dataset yelp_polarity --dataset_config plain_text \

                eval_file="${news_sum_dir}dev/${eval_year}_${eval_month}.jsonl"
                python -m slurm_jobs.example_run_summarization_sweep --experiment baseline \
                    --model $interp_output_dir --seed $SEED \
                    --num-nodes 1 --num-gpus-per-node 1 --do_eval \
                    --eval_output_dir ${eval_out_dir}up_to_${train_year}_${train_month}_eval_${eval_year} \
                    --valid_file $eval_file \
                    --identifier ${MODEL}_${TASK}_up_to_${train_year}_${train_month}_eval_${eval_year}
                    $LORA_PHRASE
            done

            #rm -rf ${tv_out_dir}_tv1/
            #rm -rf ${tv_out_dir}_tv2/
            #rm -rf $cur_out_dir

            prev_ckpt=$interp_output_dir
            #n_pct_pts=$(($n_pct_pts + 1))
        fi
    done
done
