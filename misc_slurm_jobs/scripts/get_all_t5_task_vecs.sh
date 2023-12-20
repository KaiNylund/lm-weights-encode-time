#!/bin/bash
#SBATCH --partition=ckpt
## make sure we don't clobber log files if jobs get restarted
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --time=24:00:00
## make sure we are told about preempts, and jobs running out of time, 5 min beforehand
#SBATCH --signal=USR1@60
## number of cpus *per task*. Highly recommend this to be 10.
#SBATCH --cpus-per-task=1
## srun forks ntasks_per_node times on each node
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100G
#SBATCH --gpus-per-node=1
#SBATCH -C 'a40|a100'


pretrained_model="t5-large"
models_dir="./models/"

task_dest="./${pretrained_model}_vecs/wmt_lm/"
#"2006-2008" "2009-2011" "2012-2014" "2015-2017" "2018-2020"
#for time in {2012..2016}
#for time in {2012..2016}
for year in {2017..2020}
do
    #for month in {0..11}
    #do
    #finetuned_model_path="./models/t5-small_lm_combined_years_2012-2016_wmt"
    finetuned_model_path="KaiNylund/t5-770M-lm-wmt-${year}"
    python -m task_vectors.get_task_vector \
        --path_to_pretrained_model $pretrained_model \
        --path_to_finetuned_model $finetuned_model_path \
        --alpha 1.0 \
        --output_dir "${task_dest}${year}" \
        --lora
        #--output_dir "${task_dest}combined_years_2012-2016" \
        #--output_dir "${task_dest}${year}_${month}" \
        #--lora #--save_lora
        #"${models_dir}${pretrained_model}_aic_${time}_seed_42" \
    #done
done
