from slurm_jobs.slurm_job import run_grid
import argparse
from slurm_jobs.slurm_constants import CONSTANTS
import os
import numpy as np
from pathlib import Path



TASKS = {
    "xnli": {
        "en":{
            # "max_source_length": 1024,
            # "max_target_length": 128,
            # "num_train_epochs":  10,
            # "learning_rate": 1e-3,
            # "gradient_accumulation": 2,
            # "batch_size": 8,

            "max_source_length": 130,
            "max_target_length": 128,
            "num_train_epochs":  10,
            "learning_rate": 0.0008,
            "gradient_accumulation": 16,
            "batch_size": 8,

            "input_column_1": "premise",
            "input_column_2": "hypothesis",
            "source_prefix_1": "premise:",
            "source_prefix_2": "hypothesis:",
            "patience": 10
        },
        "fr": {
            "max_source_length": 130,
            "max_target_length": 128,
            "num_train_epochs":  10,
            "learning_rate": 0.0008,
            "gradient_accumulation": 16,
            "batch_size": 8,
            "input_column_1": "premise",
            "input_column_2": "hypothesis",
            "source_prefix_1": "premise:",
            "source_prefix_2": "hypothesis:",
            "patience": 10
        }
    },
    "paws-x": {
        "en": {
            "input_column_1": "sentence1",
            "input_column_2": "sentence2",
            "source_prefix_1": "premise:",
            "source_prefix_2": "hypothesis:",
            "max_source_length": 130,
            "max_target_length": 128,
            "num_train_epochs":  10,
            "learning_rate": 0.0008,
            "gradient_accumulation": 16,
            "batch_size": 8,
            "patience": 3
        },
        "fr":  {
            "input_column_1": "sentence1",
            "input_column_2": "sentence2",
            "source_prefix_1": "premise:",
            "source_prefix_2": "hypothesis:",
            "max_source_length": 130,
            "max_target_length": 128,
            "num_train_epochs":  10,
            "learning_rate": 0.0008,
            "gradient_accumulation": 16,
            "batch_size": 8,
            "patience": 3
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
            "num_train_epochs":  1,
            "learning_rate": 0.0008,
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
            "max_target_length": 128,
            "num_train_epochs":  1,
            "learning_rate": 0.0008,
            "gradient_accumulation": 8,
            "batch_size": 2,
            "patience": 3
        }
    }
}

def main(model='facebook/opt-125m', experiment="dense", dataset='c4', dataset_config='en.noblocklist', seed=[0], num_gpus_per_node=2, num_nodes=1, do_train=True, do_eval=False, do_predict=False, eval_output_dir=None, slurm=False, disable_wandb=False, debug=False, dry_mode=False, missing_year=2012, alpha1=[], alpha2=[]):
    DEBUG_MODE = debug
    DRY_MODE = dry_mode
    MODEL = model
    
    username = ""
    RUN_CONSTANTS = CONSTANTS.get(username)
    if RUN_CONSTANTS is None:
        raise OSError("username isn't defined in slurm_constants file")
    ROOT_FOLDER = "."
    NUM_NODES = num_nodes
    NUM_GPUS_PER_NODE = num_gpus_per_node
    identifier = f"PATCHING_YEAR={missing_year}_MODEL={MODEL.replace('/', '-')}_GPUS={NUM_GPUS_PER_NODE * NUM_NODES}_NODES={NUM_NODES}"
    SWEEP_NAME = f"sweep_{identifier}_new"
    name_keys = ["IDENTIFIER", "EXPERIMENT", "DATASET", "DATASET_CONFIG", "MODEL", "SEED", "ALPHA1", "ALPHA2", "ALPHA3"]


    # if do_eval and not do_train:
    output_dir = eval_output_dir + f"/{identifier}"
    # else:
    #     output_dir = ROOT_FOLDER + "models" + f"/{identifier}"

    print(TASKS.keys())
    print(dataset)
    print(dataset_config)

    grids = {
        SWEEP_NAME: {
            'fixed_args': '',
            'positional_args': {
                "ALPHA_1": alpha1 or [0, 0.1, 0.2, 0.3, 0.4 ,0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "ALPHA_2": alpha2 or [0, 0.1, 0.2, 0.3, 0.4 ,0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "EXPERIMENT": [experiment],
                "MODEL_": [MODEL],
                "DATASET": [dataset],
                "DATASET_CONFIG": [dataset_config],
                "MODEL": [MODEL.replace('/', '-')],
                "NUM_GPUS_PER_NODE": [NUM_GPUS_PER_NODE],
                "OUTPUT_DIR": [output_dir],
                "DO_TRAIN": [do_train],
                "DO_EVAL": [do_eval],
                "DO_PREDICT": [do_predict],
                "SLURM": [slurm],
                "WANDB_DISABLED": [disable_wandb],
                "SEED": seed,
                "INPUT_COLUMN_1": [TASKS[dataset][dataset_config]['input_column_1']],
                "INPUT_COLUMN_2": [TASKS[dataset][dataset_config]['input_column_2']],
                "SOURCE_PREFIX_1": [TASKS[dataset][dataset_config]['source_prefix_1']],
                "SOURCE_PREFIX_2": [TASKS[dataset][dataset_config]['source_prefix_2']],
                "MAX_SOURCE_LENGTH": [TASKS[dataset][dataset_config]['max_source_length']],
                "MAX_TARGET_LENGTH": [TASKS[dataset][dataset_config]['max_target_length']],
                "NUM_EPOCHS": [TASKS[dataset][dataset_config]['num_train_epochs']],
                "LR": [TASKS[dataset][dataset_config]['learning_rate']],
                "GRAD_ACC": [TASKS[dataset][dataset_config]['gradient_accumulation']],
                "BATCH_SIZE": [TASKS[dataset][dataset_config]['batch_size']],
                "PATIENCE": [TASKS[dataset][dataset_config]['patience']],
                "MISSING_YEAR": [missing_year],
                "IDENTIFIER": [f"{missing_year}_missing_year_sweep"],
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
            prefix=f'bash {ROOT_FOLDER}/scripts/run_missing_year.sh',
            gpus=NUM_GPUS_PER_NODE,
            cpus=2,
            nodes=NUM_NODES,
            data_parallel=True,
            #TODO change these
            account=RUN_CONSTANTS['SLURM_ACCOUNT'],
            partition=RUN_CONSTANTS['SLURM_PARTITION'],
            jobtime='04:30:00',
            mem_gb=32,
            job_id_start=1,
            volta32=False,
            debug_mode=DEBUG_MODE,
            dry_mode=DRY_MODE,
            add_name='end',
            DIR_PATH=ROOT_FOLDER,
            conda_env_name="merging_comp",
            #constraints=["a40|a100"]
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--dry-mode', action='store_true')

    parser.add_argument('--model', type=str)

    parser.add_argument('--experiment', type=str, choices=['baseline', 'seed', 'lofi'])
    parser.add_argument('--num-nodes', type=int)
    parser.add_argument('--num-gpus-per-node', type=int)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--do_predict', action='store_true')

    parser.add_argument('--eval_output_dir', type=str)
    parser.add_argument('--missing_year', type=str)
    parser.add_argument('--alpha1', nargs="+", type=float)
    parser.add_argument('--alpha2', nargs="+", type=float)
    parser.add_argument('--seed', nargs="+", type=int)
    parser.add_argument('--slurm', action='store_true')
    parser.add_argument('--disable_wandb', action='store_true')
    parser.add_argument('--dataset', type=str, default="yelp_polarity")
    parser.add_argument('--dataset_config', type=str, default="plain_text")

    args = parser.parse_args()
    kwargs = vars(args)
    print(kwargs)
    main(**kwargs)
