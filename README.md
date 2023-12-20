# Time is Encoded in the Weights of Finetuned Language Models

We release three language modeling datasets, over 500 time-specific models, and scripts for reproducing the main paper results.

## Data

We provide processed [WMT (2012-2020)](https://huggingface.co/datasets/KaiNylund/WMT-year-splits), [Twitter (2015-2020)](https://huggingface.co/datasets/KaiNylund/twitter-year-splits), and [arXiv (2006-2020)](https://huggingface.co/datasets/KaiNylund/arxiv-year-splits) yearly language modeling splits, and monthly [WMT splits (Jan.2012-Dec.2020)](https://huggingface.co/datasets/KaiNylund/WMT-month-splits) on Huggingface.

We use the following processed downstream tasks from ["Time Waits for No One!" (Luu et al., 2022)](https://arxiv.org/pdf/2111.07408.pdf):

- Newsroom Summarization (NewsSum)
- Newsroom Source Classification (NewsCls)
- Tweet Political Affiliation Classification (PoliAff)
- AI Publisher Classification (AIC)

To use the NewsSum and NewsCls tasks, first download the [Newsroom dataset](https://lil.nlp.cornell.edu/newsroom/index.html), then process with the script from the ["Time Waits for No One!" repo](https://github.com/Temporal-Misalignment/time-waits-for-no-one/blob/main/data/newsroom/newsroom_to_tempdrift.py).

PoliAff text is omitted due to the Twitter License Agreement, but Luu et al. provide labels and tweet IDs in [their repo](https://github.com/Temporal-Misalignment/time-waits-for-no-one/tree/main).

AIC splits are also available at [Luu et al.'s repo](https://github.com/Temporal-Misalignment/time-waits-for-no-one/tree/main).

## Models

We release all T5 models finetuned with time-specific data on [Huggingface](https://huggingface.co/KaiNylund).

Yearly downstream models are labeled as ``KaiNylund/t5-{t5-size}-{task}-{time}``. For example, to load the T5-large model finetuned on 2018 PoliAff data, run:

```
from transformers import AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained("KaiNylund/t5-770M-poli_aff-2018")
```

For language modeling tasks, the format is ``KaiNylund/t5-{t5-size}-lm-{dataset}-{time}``. For instance, T5-small finetuned on October 2016 WMT language modeling is at KaiNylund/t5-60M-lm-wmt-2016-9

We also provide GPT2-small and XGLM-564M finetuned on yearly and monthly english WMT data (and german WMT data for XGLM), although we do not cover the finetuning process for these models in our paper.

## Reproducing Experiments

We provide scripts to reproduce individual experiments from the paper in [experiment_scripts](./experiment_scripts/) on a single GPU, and [a file with usage examples](./reproduce_all_results.sh).Unfortunately, because we do not directly release the downstream task datasets, running most fils will require downloading an external dataset and then updating paths to training or evaluation files.

For example, to reproduce the T5-small task analogy experiments for NewsSum + WMT LM:
1. Install the conda environment with ``conda env create -f environment.yml``
2. Download and process the newsroom summarization dataset into yearly json evaluation splits with "text" and "summary" fields
3. Update the lines ``news_sum_eval_dir = ""`` and ``eval_file="${news_sum_eval_dir}${eval_year}"`` in [time_vec_analogies.sh](./experiment_scripts/time_vec_analogies.sh)
4. Run ``bash ./experiment_scripts/time_vec_analogies.sh``

Due to the large number of evaluations (particularly for the monthly decay heatmap and time vector analogy alpha sweeps), we reccomend running experiments in paralell. As a starting point, we provide our unorganized slurm scripts in [misc_slurm_jobs](./misc_slurm_jobs/), although using these will require updating the file structure and slurm account information in [slurm_constants.py](./misc_slurm_jobs/slurm_constants.py).
