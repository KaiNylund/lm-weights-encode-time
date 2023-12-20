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


INTERP_ID="wmt_2016_full_sub_test"
SOURCE_MODEL1="KaiNylund/t5-60M-lm-wmt-2012_to_2016" #_ALPHA1_0.5_ALPHA2_0.5"" #"KaiNylund/t5-60M-lm-wmt-2015"
SOURCE_MODEL2="KaiNylund/t5-60M-lm-wmt-2016" #"KaiNylund/t5-60M-lm-wmt-2016"
EVAL_FILES=(${wmt_dir}test_json/year_2012_10000000_bytes 
            ${wmt_dir}test_json/year_2013_10000000_bytes 
            ${wmt_dir}test_json/year_2014_10000000_bytes
            ${wmt_dir}test_json/year_2015_10000000_bytes
            ${wmt_dir}test_json/year_2016_10000000_bytes)
EVAL_LABELS=("2012" "2013" "2014" "2015" "2016")
SEED=42

#ALPHA1S=(0.5 0.75 1.5 1.25)
#ALPHA2S=(0.5 0.25 -0.5 -0.25)

ALPHA1S=(0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1  0.11 0.12 0.13
 0.14 0.15 0.16 0.17 0.18 0.19 0.2  0.21 0.22 0.23 0.24 0.25 0.26 0.27 
 0.28 0.29 0.3  0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4  0.41 
 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5  0.51 0.52 0.53 0.54 0.55 
 0.56 0.57 0.58 0.59 0.6  0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 
 0.7  0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8  0.81 0.82 0.83 
 0.84 0.85 0.86 0.87 0.88 0.89 0.9  0.91 0.92 0.93 0.94 0.95 0.96 0.97 
 0.98 0.99 1.0)
ALPHA2S=(1.0 0.99 0.98 0.97 0.96 0.95 0.94 0.93 0.92 0.91 0.9  0.89 0.88 0.87
 0.86 0.85 0.84 0.83 0.82 0.81 0.8  0.79 0.78 0.77 0.76 0.75 0.74 0.73
 0.72 0.71 0.7  0.69 0.68 0.67 0.66 0.65 0.64 0.63 0.62 0.61 0.6  0.59
 0.58 0.57 0.56 0.55 0.54 0.53 0.52 0.51 0.5  0.49 0.48 0.47 0.46 0.45
 0.44 0.43 0.42 0.41 0.4  0.39 0.38 0.37 0.36 0.35 0.34 0.33 0.32 0.31
 0.3  0.29 0.28 0.27 0.26 0.25 0.24 0.23 0.22 0.21 0.2  0.19 0.18 0.17
 0.16 0.15 0.14 0.13 0.12 0.11 0.1  0.09 0.08 0.07 0.06 0.05 0.04 0.03
 0.02 0.01 0.0)
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

    vec_out_dir="./${MODEL}_vecs/misc/"
    python -m task_vectors.get_task_vector \
        --path_to_pretrained_model $MODEL \
        --path_to_finetuned_model $SOURCE_MODEL1 \
        --alpha 1.0 \
        --output_dir "${vec_out_dir}tv1" \
        $LORA_PHRASE

    python -m task_vectors.get_task_vector \
        --path_to_pretrained_model $MODEL \
        --path_to_finetuned_model $SOURCE_MODEL2 \
        --alpha 1.0 \
        --output_dir "${vec_out_dir}tv2" \
        $LORA_PHRASE

    for i in "${!ALPHA1S[@]}"
    do
        #ALPHA1=${ALPHA1S[$i]}
        ALPHA1=$((1.0 + ${ALPHA2S[$i]}))
        ALPHA2="-${ALPHA2S[$i]}"
        python -u ./multi_task.py \
            --path_to_source_model ${MODEL} \
            --task_vectors "${vec_out_dir}tv1" \
                           "${vec_out_dir}tv2" \
            --lambdas $ALPHA1 $ALPHA2 \
            --output_dir ./interp_outputs/${INTERP_ID}_interp_${ALPHA1}_${ALPHA2}

        for i in "${!EVAL_FILES[@]}"
        do
            eval_file=${EVAL_FILES[$i]}
            eval_label=${EVAL_LABELS[$i]}
            python -m slurm_jobs.example_run_finetune_t5_sweep  \
                --experiment baseline \
                --model ./interp_outputs/${INTERP_ID}_interp_${ALPHA1}_${ALPHA2} \
                --seed $SEED \
                --num-nodes 1 --num-gpus-per-node 1 --do_eval \
                --eval_output_dir ./interp_outputs/${MODEL}_${INTERP_ID}_interp/eval_${eval_label}_ALPHA1_${ALPHA1}_ALPHA2_${ALPHA2} \
                --valid_file $eval_file \
                --identifier ${MODEL}_${INTERP_ID}_interp_${ALPHA1}_${ALPHA2}_eval_${eval_label} \
                --lm --dataset "lm" --dataset_config "plain_text"
                #--dataset "yelp_polarity" --dataset_config "plain_text"
        done
        #rm -rf "${vec_out_dir}_tv1"
        #rm -rf "${vec_out_dir}_tv2"
    done
done
