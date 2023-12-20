from slurm_jobs.slurm_job import run_grid
import argparse
from slurm_jobs.slurm_constants import CONSTANTS
import os
import numpy as np
from pathlib import Path



HPS = {
    "small": {
        "num_train_epochs":  1,
        "learning_rate": 8e-4,
        "gradient_accumulation": 8,
        "batch_size": 2,
        "patience": 3
    },
    "base": {
        "num_train_epochs":  1,
        "learning_rate":8e-4,
        "gradient_accumulation": 8,
        "batch_size": 2,
        "patience": 3
    },
    "large": {
        "num_train_epochs":  1,
        "learning_rate": 5e-5,
        "gradient_accumulation": 8,
        "batch_size": 2,
        "patience": 3
    },
    "3b": {
        "num_train_epochs":  1,
        "learning_rate": 5e-5,
        "gradient_accumulation": 8,
        "batch_size": 2,
        "patience": 3
    },
    "11b": {
        "num_train_epochs":  1,
        "learning_rate": 5e-5,
        "gradient_accumulation": 8,
        "batch_size": 2,
        "patience": 3
    },
}

TASKS = {
    "wmt_year_cls": {
        "plain_text": {
            "input_column_1": "text",
            "input_column_2": None,
            "source_prefix_1": "lm:",
            "source_prefix_2": None,
            "max_source_length": 128,
            "max_target_length": 128,
            "target_label": "label",
            "num_train_epochs": 3
        }
    },
    "aic": {
        "plain_text": {
            "input_column_1": "text",
            "input_column_2": None,
            "source_prefix_1": "lm:",
            "source_prefix_2": None,
            "max_source_length": 128,
            "max_target_length": 128,
            "target_label": "label",
            "num_train_epochs": 3
        }
    },
    "xnli": {
        "en":{
            "max_source_length": 130,
            "max_target_length": 128,
            "input_column_1": "premise",
            "input_column_2": "hypothesis",
            "source_prefix_1": "premise:",
            "source_prefix_2": "hypothesis:",
        },
        "fr": {
            "max_source_length": 130,
            "max_target_length": 128,
            "input_column_1": "premise",
            "input_column_2": "hypothesis",
            "source_prefix_1": "premise:",
            "source_prefix_2": "hypothesis:",
        }
    },
    "paws-x": {
        "en": {
            "input_column_1": "sentence1",
            "input_column_2": "sentence2",
            "source_prefix_1": "premise:",
            "source_prefix_2": "hypothesis:",
            "max_source_length": 130,
            "max_target_length": 128
        },
        "fr":  {
            "input_column_1": "sentence1",
            "input_column_2": "sentence2",
            "source_prefix_1": "premise:",
            "source_prefix_2": "hypothesis:",
            "max_source_length": 130,
            "max_target_length": 128
        }
    },
    "imdb": {
        "plain_text": {
            "input_column_1": "text",
            "input_column_2": None,
            "source_prefix_1": "sentiment:",
            "source_prefix_2": None,
            "max_source_length": 130,
            "max_target_length": 128,
            "target_label": "label",
            # "num_train_epochs":  10,
            # "learning_rate": 0.0008,
            # "gradient_accumulation": 16,
            # "batch_size": 8,
            # "patience": 3
        }
    },
    "yelp_polarity": {
        "plain_text": {
            "input_column_1": "text",
            "input_column_2": None,
            "source_prefix_1": "lm:",
            "source_prefix_2": None,
            "max_source_length": 128,
            "max_target_length": 128,
            "target_label": "label",
            "num_train_epochs":  3,
            # "learning_rate": 0.0008,
            "gradient_accumulation": 8,
            "batch_size": 2,
            "patience": 3
        }
    },
    "amazon_polarity": {
        "amazon_polarity": {
            "input_column_1": "content",
            "input_column_2": None,
            "source_prefix_1": "lm:",
            "source_prefix_2": None,
            "max_source_length": 128,
            "target_label": "label",
            "max_target_length": 128,
            # "num_train_epochs":  1,
            # "learning_rate": 0.0008,
            "gradient_accumulation": 8,
            "batch_size": 2,
            "patience": 3
        }
    },
    "sst2": {
        "plain_text": {
            "input_column_1": "sentence",
            "input_column_2": None,
            "source_prefix_1": "lm:",
            "source_prefix_2": None,
            "max_source_length": 128,
            "target_label": "label",
            "max_target_length": 128,
            # "num_train_epochs":  1,
            # "learning_rate": 0.0008,
            "gradient_accumulation": 8,
            "batch_size": 2,
            "patience": 3
        }
    },

    "scitldr": {
        "FullText": {
            "input_column_1": "source",
            "input_column_2": None,
            "target_label": "target",
            "source_prefix_1": "lm:",
            "source_prefix_2": None,
            "target_label": "target",
            "max_source_length": 128,
            "max_target_length": 128,
            # "num_train_epochs":  1,
            # "learning_rate": 0.0008,
            "gradient_accumulation": 8,
            "batch_size": 2,
            "patience": 3
        }
    },
    "newsroom": {
        "plain_text": {
            "input_column_1": "text",
            "input_column_2": "text",
            "target_label": "summary",
            "source_prefix_1": "text:",
            "source_prefix_2": "summary:",
            "max_source_length": 128,
            "max_target_length": 128,
            # "num_train_epochs":  1,
            # "learning_rate": 0.0008,
            "gradient_accumulation": 8,
            "batch_size": 2,
            "patience": 3
        }
    },
    "lm" : {
        "plain_text": {
            "source_prefix_1": None,
            "source_prefix_2": None,
            "num_train_epochs": 1,
            "input_column_1": "text",
            "input_column_2": "text",
            "target_label": None,
            "learning_rate": 0.0008,
            "save_steps": 1000,
            "save_strategy": "steps",
            "save_total_limit": 1,
            "max_source_length": 128,
            "max_target_length": 128,
            "gradient_accumulation": 8,
            "batch_size": 2,
            "patience": 3
        }
    }
}

def main(model='facebook/opt-125m', model_type='small', experiment="baseline",
         dataset='yelp_polarity', dataset_config='plain_text', seed=[0],
         num_gpus_per_node=2, num_nodes=1, do_train=True, do_eval=False,
         do_predict=False, eval_output_dir=None, slurm=False, disable_wandb=False,
         debug=False, lm=False, dry_mode=False, train_file=None, valid_file=None,
        predict_file=None, identifier=None, lora=False, lr=None, num_epochs=None):
    DEBUG_MODE = debug
    DRY_MODE = dry_mode
    MODEL = model
    username = ""
    RUN_CONSTANTS = CONSTANTS.get(username)
    if RUN_CONSTANTS is None:
        raise OSError("username isn't defined in slurm_constants file")
    ROOT_FOLDER = RUN_CONSTANTS['ROOT_FOLDER'] 
    NUM_NODES = num_nodes
    NUM_GPUS_PER_NODE = num_gpus_per_node
    identifier = f"IDENTIFIER={identifier}_EXPERIMENT={experiment}_DATASET={dataset}_EPOCHS={num_epochs}_LM={lm}"
    SWEEP_NAME = f"sweep_{identifier}"
    name_keys = ["IDENTIFIER","EXPERIMENT", "DATASET", "DATASET_CONFIG", "MODEL", "SEED", "LM"]

    if num_epochs != None:
        epochs = num_epochs
    elif (dataset != None) and ("num_train_epochs" in TASKS[dataset][dataset_config]):
        epochs = TASKS[dataset][dataset_config]["num_train_epochs"]
    else:
        epochs = 1


    if do_eval and not do_train:
        output_dir = eval_output_dir
        jobtime = '03:00:00'
    else:
        output_dir = ROOT_FOLDER + "models" + f"/{identifier}"
        jobtime = '96:00:00'

    grids = {
        SWEEP_NAME: {
            'fixed_args': '',
            'positional_args': {
                "EXPERIMENT": [experiment],
                "MODEL_": [MODEL],
                "DATASET": [dataset] if dataset != "newsroom" else [None],
                "DATASET_CONFIG": [dataset_config] if dataset != "newsroom" else [None],
                "MODEL": [MODEL.replace('/', '-')],
                "NUM_GPUS_PER_NODE": [NUM_GPUS_PER_NODE],
                "OUTPUT_DIR": [output_dir],
                "DO_TRAIN": [do_train],
                "DO_EVAL": [do_eval],
                "DO_PREDICT": [do_predict],
                "SLURM": [slurm],
                "WANDB_DISABLED": [disable_wandb],
                "SEED": seed,
                "INPUT_COLUMN_1": [TASKS[dataset][dataset_config]['input_column_1']] if dataset else ["text"],
                "INPUT_COLUMN_2": [TASKS[dataset][dataset_config]['input_column_2']] if dataset else ["text"],
                "SOURCE_PREFIX_1": [TASKS[dataset][dataset_config]['source_prefix_1']] if dataset else [None],
                "SOURCE_PREFIX_2": [TASKS[dataset][dataset_config]['source_prefix_2']] if dataset else [None],
                "MAX_SOURCE_LENGTH": [TASKS[dataset][dataset_config]['max_source_length']] if dataset else [128],
                "MAX_TARGET_LENGTH": [TASKS[dataset][dataset_config]['max_target_length']] if dataset else [128],
                "NUM_EPOCHS": [epochs],
                "LR": [lr] if lr != None else [HPS[model_type]['learning_rate']],
                "GRAD_ACC": [TASKS[dataset][dataset_config]['gradient_accumulation']],
                "BATCH_SIZE": [TASKS[dataset][dataset_config]['batch_size']],
                "PATIENCE": [TASKS[dataset][dataset_config]['patience']],
                "LM": [lm],
                "TARGET_LABEL": [TASKS[dataset][dataset_config]['target_label']] if dataset else [None],
                "TRAIN_FILE": [train_file],
                "VALID_FILE": [valid_file],
                "PREDICT_FILE": [predict_file],
                "LORA": [lora],
                "IDENTIFIER": [identifier],

            },
            'named_args': {},
        },
    }

    for sweep_name, grid in grids.items():
        run_grid(
            grid,
            name_keys,
            sweep_name,
            user=os.environ['USER'],
            prefix=f'bash {ROOT_FOLDER}/scripts/run_finetune_t5.sh',
            gpus=NUM_GPUS_PER_NODE,
            cpus=1,
            nodes=NUM_NODES,
            data_parallel=True,
            #TODO change these
            account=RUN_CONSTANTS['SLURM_ACCOUNT'],
            partition=RUN_CONSTANTS['SLURM_PARTITION'],
            jobtime=jobtime,
            mem_gb=64,
            job_id_start=1,
            volta32=False,
            debug_mode=DEBUG_MODE,
            dry_mode=DRY_MODE,
            add_name='end',
            DIR_PATH=ROOT_FOLDER,
            conda_env_name="merging_comp",
            constraints=["a40|a100"]
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--dry-mode', action='store_true')

    parser.add_argument('--model', type=str)
    parser.add_argument('--model_type', type=str, default='small')

    parser.add_argument('--train_file', type=str)
    parser.add_argument('--valid_file', type=str)
    parser.add_argument('--predict_file', type=str)
    parser.add_argument('--experiment', type=str, choices=['baseline', 'seed', 'lofi'])
    parser.add_argument('--num-nodes', type=int)
    parser.add_argument('--num-gpus-per-node', type=int)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--do_predict', action='store_true')

    parser.add_argument('--eval_output_dir', type=str)
    parser.add_argument('--seed', nargs="+", type=int)
    parser.add_argument('--slurm', action='store_true')
    parser.add_argument('--disable_wandb', action='store_true')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--dataset_config', type=str, default=None)
    parser.add_argument('--num_epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--lm', action='store_true')
    parser.add_argument('--lora', action='store_true')
    parser.add_argument('--identifier', type=str, default=None)
    args = parser.parse_args()
    kwargs = vars(args)
    print(kwargs)
    main(**kwargs)
