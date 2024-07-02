#!/bin/bash
#SBATCH --job-name=eval-t5-task-vecs-analogies
#SBATCH --account=ark
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1

# TODO: Modify these and possibly thethe evaluation file paths below to run
# wmt + news_sum and twitter + poli_aff time vector analogy sweeps.
news_sum_eval_dir=""
poli_aff_eval_dir=""

SEED=42
PRETRAINED_MODEL=$1
ALPHA1S=(0.1 0.2 0.3 0.4 0.5 0.6)
ALPHA2S=(0.1 0.2 0.3 0.4 0.5 0.6)
ALPHA3S=(0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0 2.2)


eval_out_dir="${PRETRAINED_MODEL}_analogy_evals/"
vec_out_dir="${PRETRAINED_MODEL}_vecs/"
if [[ $PRETRAINED_MODEL == "t5-3b" ]]; then
    LR=0.0002
    LORA_PHRASE="--lora"
    HF_MODEL="t5-3b"
elif [[ $PRETRAINED_MODEL == "t5-large" ]]; then
    LR=0.0008
    LORA_PHRASE="--lora"
    HF_MODEL="t5-770M"
else
    LR=0.0008
    LORA_PHRASE=""
    HF_MODEL="t5-60M"
fi

# ------------------------------------------------------------------------------------------------
# NewsSum + WMT LM time vector analogy alpha sweeps
# ------------------------------------------------------------------------------------------------

if [[ news_sum_eval_dir != "" ]];
then
    start_year_downstream_model="KaiNylund/${HF_MODEL}-news_sum-2012"
    start_year_lm_model="KaiNylund/${HF_MODEL}-lm-wmt-2012"

    python -u ../task_vectors/get_task_vector.py \
        --path_to_pretrained_model $PRETRAINED_MODEL \
        --path_to_finetuned_model $start_year_downstream_model \
        --alpha 1.0 \
        --output_dir "${vec_out_dir}news_sum_2012_vec" \
        $LORA_PHRASE

    python -u ../task_vectors/get_task_vector.py \
        --path_to_pretrained_model $PRETRAINED_MODEL \
        --path_to_finetuned_model $start_year_lm_model \
        --alpha 1.0 \
        --output_dir "${vec_out_dir}wmt_lm_2012_vec" \
        $LORA_PHRASE

    for target_year in {2013..2016}
    do
        target_year_lm_model="KaiNylund/${HF_MODEL}-lm-wmt-${target_year}"

        python -u ../task_vectors/get_task_vector.py \
            --path_to_pretrained_model $PRETRAINED_MODEL \
            --path_to_finetuned_model $target_year_lm_model \
            --alpha 1.0 \
            --output_dir "${vec_out_dir}wmt_lm_${target_year}_vec" \
            $LORA_PHRASE

        for i in "${!ALPHA1S[@]}"
        do
            ALPHA1=${ALPHA1S[$i]}
            for j in "${!ALPHA2S[@]}"
            do
                ALPHA2=${ALPHA2S[$j]}
                for j in "${!ALPHA3S[@]}"
                do
                    ALPHA3=${ALPHA3S[$k]}
                    echo $ALPHA1
                    echo $ALPHA2
                    echo $ALPHA3
                    # Do task analogy with time vectors
                    python -u ../task_vectors/task_analogy.py \
                        --path_to_source_model $PRETRAINED_MODEL \
                        --task_vector_A "${vec_out_dir}wmt_lm_2012_vec" \
                        --task_vector_B "${vec_out_dir}wmt_lm_${target_year}_vec" \
                        --task_vector_C "${vec_out_dir}news_sum_2012_vec" \
                        --lambdaA $ALPHA1 \
                        --lambdaB $ALPHA2 \
                        --lambdaC $ALPHA3 \
                        --output_dir ${vec_out_dir}${PRETRAINED_MODEL}_2012_news_sum_wmt_analogy_${ALPHA1}_${ALPHA2}_${ALPHA3} \
                        $LORA_PHRASE

                    # Run NewsSum evaluation
                    # TODO: update if needed
                    eval_file="${news_sum_eval_dir}${target_year}.jsonl"
                    python -u ../finetuning_scripts/finetune_t5_summarization.py \
                            --model_name_or_path ${vec_out_dir}${PRETRAINED_MODEL}_2012_news_sum_wmt_analogy_${ALPHA1}_${ALPHA2}_${ALPHA3} \
                            --do_eval \
                            --validation_file $eval_file \
                            --text_column "text" \
                            --summary_column "summary" \
                            --output_dir ${eval_out_dir}${PRETRAINED_MODEL}_2012_news_sum_wmt_analogy_${ALPHA1}_${ALPHA2}_${ALPHA3}_target_${target_year} \
                            --seed $SEED \
                            --save_steps 200 \
                            --save_strategy no \
                            --learning_rate $LR \
                            --source_prefix 'summarize: ' \
                            --gradient_accumulation_steps 8 \
                            --ddp_find_unused_parameters False \
                            --per_device_train_batch_size 2 \
                            --per_device_eval_batch_size 2 \
                            --predict_with_generate \
                            --overwrite_output_dir \
                            $LORA_PHRASE

                    rm -rf "${vec_out_dir}${PRETRAINED_MODEL}_2012_news_sum_wmt_analogy_${ALPHA1}_${ALPHA2}_${ALPHA3}"
                done
            done
        done
    done
fi


# ------------------------------------------------------------------------------------------------
# PoliAff + Twitter LM time vector analogy alpha sweeps
# ------------------------------------------------------------------------------------------------

if [[ poli_aff_eval_dir != "" ]];
then
    start_year_downstream_model="KaiNylund/${HF_MODEL}-poli_aff-2015"
    start_year_lm_model="KaiNylund/${HF_MODEL}-lm-twitter-2015"

    python -u ../task_vectors/get_task_vector.py \
        --path_to_pretrained_model $PRETRAINED_MODEL \
        --path_to_finetuned_model $start_year_downstream_model \
        --alpha 1.0 \
        --output_dir "${vec_out_dir}poli_aff_2015_vec" \
        $LORA_PHRASE

    python -u ../task_vectors/get_task_vector.py \
        --path_to_pretrained_model $PRETRAINED_MODEL \
        --path_to_finetuned_model $start_year_lm_model \
        --alpha 1.0 \
        --output_dir "${vec_out_dir}twitter_lm_2015_vec" \
        $LORA_PHRASE

    for target_year in {2016..2020}
    do
        target_year_lm_model="KaiNylund/${HF_MODEL}-lm-twitter-${target_year}"

        python -u ../task_vectors/get_task_vector.py \
            --path_to_pretrained_model $PRETRAINED_MODEL \
            --path_to_finetuned_model $target_year_lm_model \
            --alpha 1.0 \
            --output_dir "${vec_out_dir}twitter_lm_${target_year}_vec" \
            $LORA_PHRASE

        for i in "${!ALPHA1S[@]}"
        do
            ALPHA1=${ALPHA1S[$i]}
            for j in "${!ALPHA2S[@]}"
            do
                ALPHA2=${ALPHA2S[$j]}
                for j in "${!ALPHA3S[@]}"
                do
                    ALPHA3=${ALPHA3S[$k]}
                    # Do task analogy with time vectors
                    python -u ../task_vectors/task_analogy.py \
                        --path_to_source_model $PRETRAINED_MODEL \
                        --task_vector_A "${vec_out_dir}twitter_lm_2015_vec" \
                        --task_vector_B "${vec_out_dir}twitter_lm_${target_year}_vec" \
                        --task_vector_C "${vec_out_dir}poli_aff_2015_vec" \
                        --lambdaA $ALPHA1 \
                        --lambdaB $ALPHA2 \
                        --lambdaC $ALPHA3 \
                        --output_dir ${vec_out_dir}${PRETRAINED_MODEL}_2015_poli_aff_twitter_analogy_${ALPHA1}_${ALPHA2}_${ALPHA3} \
                        $LORA_PHRASE

                    # Run PoliAff evaluation
                    # TODO: update if needed
                    eval_file="${poli_aff_eval_dir}${target_year}.jsonl"
                    python -u ../finetuning_scripts/finetune_t5.py  \
                        --model_name_or_path ${vec_out_dir}${PRETRAINED_MODEL}_2015_poli_aff_twitter_analogy_${ALPHA1}_${ALPHA2}_${ALPHA3} \
                        --dataset_name "yelp_polarity" \
                        --dataset_config "plain_text" \
                        --validation_file $eval_file \
                        --do_eval \
                        --input_column_1 "text" \
                        --output_dir ${eval_out_dir}${PRETRAINED_MODEL}_2015_poli_aff_twitter_analogy_${ALPHA1}_${ALPHA2}_${ALPHA3}_target_${target_year} \
                        --seed $SEED \
                        --save_steps 200 \
                        --save_strategy no \
                        --source_prefix_1 "text:" \
                        --target_label label \
                        --learning_rate $LR \
                        --max_predict_samples 1000 \
                        --max_source_length 128 \
                        --max_target_length 128 \
                        --gradient_accumulation_steps 8 \
                        --ddp_find_unused_parameters False \
                        --per_device_train_batch_size 2 \
                        --per_device_eval_batch_size 2 \
                        --predict_with_generate \
                        --patience 3 \
                        $LORA_PHRASE

                    rm -rf "${vec_out_dir}${PRETRAINED_MODEL}_2015_poli_aff_twitter_analogy_${ALPHA1}_${ALPHA2}_${ALPHA3}"
                done
            done
        done
    done
fi
