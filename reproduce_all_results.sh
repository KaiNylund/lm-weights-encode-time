#!/bin/bash

# Finetune all year and month-specific models. By default only finetunes on WMT LM data.
# To train on other datasets in the paper, download them from the instructions in
# README.md and then modify the corresponding paths in finetune_year_models.sh and
# finetune_month_models.sh
for model in "t5-small" "t5-large" "t5-3b"
do
    /bin/bash ./experiment_scripts/finetune_year_models.sh $model
done
/bin/bash ./experiment_scripts/finetune_month_models.sh

# Evaluate all year and month-specific models on all corresponding test splits.
# Also need to update TODOs for other datasets in evaluation scripts if training from
# scratch or using other datasets or training from scratch.
# NOTE: these will take a ~very~ long time to run on a single GPU,
# particularly the 3364 WMT LM month evaluations
for model in "t5-small" "t5-large" "t5-3b"
do
    /bin/bash ./experiment_scripts/eval_year_models.sh $model
done
/bin/bash ./experiment_scripts/eval_month_models.sh $model

# Intervening year interpolation experiments with WMT LM, NewsSum and PoliAff
/bin/bash ./experiment_scripts/yearly_time_vec_interpolations.sh t5-3b

# Intervening month interpolation experiments with WMT LM
/bin/bash ./experiment_scripts/monthly_time_vec_interpolations.sh

# NewsSum + WMT LM and PoliAff + Twitter LM time vector analogies.
# Need to specify paths for NewsSum and PoliAff data in the script to run.
# Could also take a very long time without paralellization since each sweep
# tests 364 alpha combinations
/bin/bash ./experiment_scripts/time_vec_analogies.sh t5-small
#/bin/bash ./experiment_scripts/time_vec_analogies.sh t5-large
#/bin/bash ./experiment_scripts/time_vec_analogies.sh t5-3b


# Uniform time soup setup
# arguments are:
# pretrained model (t5-small)
# alphas 1-6 (the sixth is 0 so it is not used)
# paths to each corresponding time vector
# path to the evaluation file (TODO)
# output path
# whether to eval as a language modeling task
# whether to eval as a summarization task
#/bin/bash ./experiment_scripts/test_time_vec_combo.sh t5-small \
#          0.2 0.2 0.2 0.2 0.2 0.0 \
#          ./t5-small_interp_evals/wmt_lm_2012_vec \
#          ./t5-small_interp_evals/wmt_lm_2013_vec \
#          ./t5-small_interp_evals/wmt_lm_2014_vec \
#          ./t5-small_interp_evals/wmt_lm_2015_vec \
#          ./t5-small_interp_evals/wmt_lm_2016_vec \
#          ./t5-small_interp_evals/wmt_lm_2016_vec \
#          ./wmt_lm_combined_years_eval_file \
#          ./wmt_lm_uniform_time_soup_combined_years_eval \
#          True \
#          False
# ... other tasks

# Greedy time soup setup. Will need to modify the time vector and eval file paths
# in greedy_time_soup.py based on how you store the datasets for each task
python -u ./experiment_scripts/greedy_time_soup.py --task "wmt_lm"
#python -u ./experiment_scripts/greedy_time_soup.py --task "news_sum"
#python -u ./experiment_scripts/greedy_time_soup.py --task "poli_aff"