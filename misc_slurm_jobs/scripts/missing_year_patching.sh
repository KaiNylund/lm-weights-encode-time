#!/bin/bash

for missing_year in 2020 #{2015..2020}
#for missing_year in "2006-2008" "2009-2011" "2012-2014" "2015-2017" "2018-2020"
do
    python -m slurm_jobs.example_run_missing_year_sweep \
        --model t5-small --experiment baseline \
        --num-nodes 1 --num-gpus-per-node 1 \
        --eval_output_dir ./search_outputs/t5-small_poli_aff_missing_year_multitask_${missing_year} \
        --seed 42 --do_eval \
        --missing_year $missing_year
done