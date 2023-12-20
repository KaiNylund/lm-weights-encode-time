#!/bin/bash
#SBATCH --job-name=eval-t5-task-vecs-interps
#SBATCH --account=ark
#SBATCH --partition=gpu-titan
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1

wmt_dir="./finetuning-data/WMTdata/en/"
news_sum_dir="./tasks/datasets/newsroom/summarization/"
news_cls_dir="./tasks/datasets/newsroom/newsroom_source_classification/"

arxiv_dir="./finetuning-data/arxiv_data/"
aic_dir="./tasks/datasets/aic/"

twitter_dir="./finetuning-data/twitter_data/"
poli_aff_dir="./tasks/datasets/poli_tweets/"

MODEL="t5-small"
_MODEL="t5-60M"
SEED=42
eval_output_dir="./swapped_emb_models/"

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


#for eval_year in {2013..2016}
#do
    # WMT lm
    #swapped_wmt_path="${eval_output_dir}wmt_2012_swapped_${eval_year}"
    #python -u ./task_vectors/swap_embeddings.py \
    #    --model_to_swap "KaiNylund/${_MODEL}-lm-wmt-${eval_year}" \
    #    --embedding_model "KaiNylund/${_MODEL}-lm-wmt-2012" \
    #    --output_dir $swapped_wmt_path \
    #    --swap_mode "embeddings" \
    #    $LORA_PHRASE

    #python -u ./scripts/run_finetune_t5.py  \
    #    --model_name_or_path $swapped_wmt_path --do_eval \
    #    --validation_file ${wmt_dir}test_json/year_${eval_year}_10000000_bytes \
    #    --output_dir ${eval_output_dir}2012_wmt_lm_target_${eval_year}_inv \
    #    --seed $SEED \
    #    --num_train_epochs 1 --learning_rate 0.0008 \
    #    --input_column_1 "text" --input_column_2 "text" \
    #    --save_steps 1000 --save_strategy "steps" --save_total_limit 1 \
    #    --max_source_length 128 --max_target_length 128 \
    #    --gradient_accumulation_steps 8 --per_device_eval_batch_size 2 \
    #    --lm \
    #    $LORA_PHRASE


    # news sum task
    #swapped_news_sum_path="${eval_output_dir}news_sum_2012_swapped_${eval_year}"
    #python -u ./task_vectors/swap_embeddings.py \
    #    --model_to_swap "KaiNylund/${_MODEL}-news_sum-2012" \
    #    --embedding_model "KaiNylund/${_MODEL}-news_sum-${eval_year}" \
    #    --output_dir $swapped_news_sum_path \
    #    --swap_mode "norm" \
    #    $LORA_PHRASE

    #python -u ./scripts/run_summarization.py \
    #    --model_name_or_path $swapped_news_sum_path --do_eval \
    #    --validation_file ${news_sum_dir}dev/${eval_year}.jsonl \
    #    --output_dir ${eval_output_dir}2012_news_sum_target_${eval_year}_norm \
    #    --seed $SEED \
    #    --learning_rate $LR --num_train_epochs 3  \
    #    --save_strategy no --source_prefix 'summarize:' --ddp_find_unused_parameters False \
    #    --per_device_train_batch_size 2 --per_device_eval_batch_size 2 \
    #    --predict_with_generate --overwrite_output_dir \
    #    $LORA_PHRASE
    

    # news source classification task
    #swapped_news_cls_path="${eval_output_dir}news_cls_2012_swapped_${eval_year}"
    #python -u ./task_vectors/swap_embeddings.py \
    #    --model_to_swap "KaiNylund/${_MODEL}-news_cls-2012" \
    #    --embedding_model "KaiNylund/${_MODEL}-news_cls-${eval_year}" \
    #    --output_dir $swapped_news_cls_path \
    #    --swap_mode "norm" \
    #    $LORA_PHRASE

    #python -u ./scripts/run_finetune_t5.py  \
    #    --model_name_or_path $swapped_news_cls_path --do_eval \
    #    --validation_file ${news_cls_dir}dev/${eval_year}.jsonl \
    #    --output_dir ${eval_output_dir}2012_news_cls_target_${eval_year}_norm \
    #    --seed $SEED --dataset_name "yelp_polarity" --dataset_config "plain_text" --input_column_1 "text" \
    #    --learning_rate $LR --max_predict_samples 1000 --max_source_length 128 --max_target_length 128 \
    #    --save_steps 200 --save_strategy no --source_prefix_1 "lm:" --target_label label \
    #    --gradient_accumulation_steps 8 --ddp_find_unused_parameters False \
    #    --per_device_train_batch_size 2 --per_device_eval_batch_size 2 \
    #    --predict_with_generate --patience 3 \
    #    $LORA_PHRASE


    #rm -r $swapped_wmt_path
    #rm -r $swapped_news_sum_path
    #rm -r $swapped_news_cls_path
#done


for eval_year in {2019..2020} #{2016..2020}
do
    # Twitter lm
    swapped_twitter_path="${eval_output_dir}twitter_2015_swapped_${eval_year}"
    python -u ./task_vectors/swap_embeddings.py \
        --model_to_swap "KaiNylund/${_MODEL}-lm-twitter-${eval_year}" \
        --embedding_model "KaiNylund/${_MODEL}-lm-twitter-2015" \
        --output_dir $swapped_twitter_path \
        --swap_mode "embeddings" \
        $LORA_PHRASE

    python -u ./scripts/run_finetune_t5.py  \
        --model_name_or_path $swapped_twitter_path --do_eval \
        --validation_file ${twitter_dir}test/year_${eval_year}_10000000_bytes \
        --output_dir ${eval_output_dir}2015_twitter_lm_target_${eval_year}_inv \
        --seed $SEED \
        --num_train_epochs 1 --learning_rate 0.0008 \
        --input_column_1 "text" --input_column_2 "text" \
        --save_steps 1000 --save_strategy "steps" --save_total_limit 1 \
        --max_source_length 128 --max_target_length 128 \
        --gradient_accumulation_steps 8 --per_device_eval_batch_size 2 \
        --lm \
        $LORA_PHRASE
    

    # poli aff task
    #swapped_poli_aff_path="${eval_output_dir}poli_aff_2015_swapped_${eval_year}"
    #python -u ./task_vectors/swap_embeddings.py \
    #    --model_to_swap "KaiNylund/${_MODEL}-poli_aff-2015" \
    #    --embedding_model "KaiNylund/${_MODEL}-poli_aff-${eval_year}" \
    #    --output_dir $swapped_poli_aff_path \
    #    --swap_mode "norm" \
    #    $LORA_PHRASE

    #python -u ./scripts/run_finetune_t5.py  \
    #    --model_name_or_path $swapped_poli_aff_path --do_eval \
    #    --validation_file ${poli_aff_dir}dev/${eval_year}.jsonl \
    #    --output_dir ${eval_output_dir}2015_poli_aff_twitter_target_${eval_year}_norm \
    #    --seed $SEED --dataset_name "yelp_polarity" --dataset_config "plain_text" --input_column_1 "text" \
    #    --learning_rate $LR --max_predict_samples 1000 --max_source_length 128 --max_target_length 128 \
    #    --save_steps 200 --save_strategy no --source_prefix_1 "lm:" --target_label label \
    #    --gradient_accumulation_steps 8 --ddp_find_unused_parameters False \
    #    --per_device_train_batch_size 2 --per_device_eval_batch_size 2 \
    #    --predict_with_generate --patience 3 \
    #    $LORA_PHRASE


    rm -r $swapped_twitter_path
    #rm -r $swapped_poli_aff_path
done


#for eval_year in "2018-2020" #"2009-2011" "2012-2014" "2015-2017" "2018-2020"
#do
    # Arxiv lm
    #swapped_arxiv_path="${eval_output_dir}arxiv_2006-2008_swapped_${eval_year}"
    #python -u ./task_vectors/swap_embeddings.py \
    #    --model_to_swap "KaiNylund/${_MODEL}-lm-arxiv-2006-2008" \
    #    --embedding_model "KaiNylund/${_MODEL}-lm-arxiv-${eval_year}" \
    #    --output_dir $swapped_arxiv_path \
    #    --swap_mode "attn" \
    #    $LORA_PHRASE

    #python -u ./scripts/run_finetune_t5.py  \
    #    --model_name_or_path $swapped_arxiv_path --do_eval \
    #    --validation_file ${arxiv_dir}test/${eval_year}_15000000_bytes \
    #    --output_dir ${eval_output_dir}2006-2008_arxiv_lm_target_${eval_year}_attn \
    #    --seed $SEED \
    #    --num_train_epochs 1 --learning_rate 0.0008 \
    #    --input_column_1 "text" --input_column_2 "text" \
    #    --save_steps 1000 --save_strategy "steps" --save_total_limit 1 \
    #    --max_source_length 128 --max_target_length 128 \
    #    --gradient_accumulation_steps 8 --per_device_eval_batch_size 2 \
    #    --lm \
    #    $LORA_PHRASE
    

    # AIC task
    #swapped_aic_path="${eval_output_dir}aic_2006-2008_swapped_${eval_year}"
    #python -u ./task_vectors/swap_embeddings.py \
    #    --model_to_swap "KaiNylund/${_MODEL}-aic-2006-2008" \
    #    --embedding_model "KaiNylund/${_MODEL}-aic-${eval_year}" \
    #    --output_dir $swapped_aic_path \
    #    --swap_mode "norm" \
    #    $LORA_PHRASE

    #python -u ./scripts/run_finetune_t5.py  \
    #    --model_name_or_path $swapped_aic_path --do_eval \
    #    --validation_file ${aic_dir}dev/${eval_year}.jsonl \
    #    --output_dir ${eval_output_dir}2006-2008_aic_arxiv_target_${eval_year}_norm \
    #    --seed $SEED --dataset_name "yelp_polarity" --dataset_config "plain_text" --input_column_1 "text" \
    #    --learning_rate $LR --max_predict_samples 1000 --max_source_length 128 --max_target_length 128 \
    #    --save_steps 200 --save_strategy no --source_prefix_1 "lm:" --target_label label \
    #    --gradient_accumulation_steps 8 --ddp_find_unused_parameters False \
    #    --per_device_train_batch_size 2 --per_device_eval_batch_size 2 \
    #    --predict_with_generate --patience 3 \
    #    $LORA_PHRASE


    #rm -r $swapped_arxiv_path
    #rm -r $swapped_aic_path
#done