from slurm_jobs.slurm_job import run_grid
import argparse
from slurm_jobs.slurm_constants import CONSTANTS
import os
import numpy as np
from pathlib import Path


def main(model='facebook/opt-125m', experiment="dense", dataset='c4', dataset_config='en.noblocklist', seed=[0], num_gpus_per_node=2, num_nodes=1, do_train=True, do_eval=True, eval_output_dir=None, slurm=False, disable_wandb=False, debug=False, dry_mode=False, train_file=None, valid_file=None, test_file=None, do_predict=False, identifier=None, lora=False):
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
    identifier = f"IDENTIFIER={identifier}_EXPERIMENT={experiment}_DATASET={dataset}_DATASETCONFIG={dataset_config}_MODEL={MODEL.replace('/', '-')}_GPUS={NUM_GPUS_PER_NODE * NUM_NODES}_NODES={NUM_NODES}"
    SWEEP_NAME = f"sweep_{identifier}"
    name_keys = ["EXPERIMENT", "DATASET", "DATASET_CONFIG", "MODEL", "SEED"]


    if do_eval and not do_train:
        output_dir = eval_output_dir
        jobtime = '03:00:00'
    else:
        output_dir = ROOT_FOLDER + "models" + f"/{identifier}"
        jobtime = '24:00:00'
    grids = {
        SWEEP_NAME: {
            'fixed_args': '',
            'positional_args': {
                "EXPERIMENT": [experiment],
                "MODEL_": [MODEL],
                "DATASET": [dataset],
                "DATASET_CONFIG": [dataset_config],
                "MODEL": [MODEL.replace('/', '-')],
                "NUM_GPUS_PER_NODE": [NUM_GPUS_PER_NODE],
                "OUTPUT_DIR": [output_dir],
                "DO_TRAIN": [do_train],
                "DO_EVAL": [do_eval],
                "SLURM": [slurm],
                "WANDB_DISABLED": [disable_wandb],
                "SEED": seed,
                "TRAIN_FILE": [train_file],
                "VALID_FILE": [valid_file],
                "TEST_FILE": [test_file],
                "DO_PREDICT": [do_predict],
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
            prefix=f'bash {ROOT_FOLDER}/scripts/run_summarization.sh',
            gpus=NUM_GPUS_PER_NODE,
            cpus=2,
            nodes=NUM_NODES,
            data_parallel=True,
            #TODO change these
            account=RUN_CONSTANTS['SLURM_ACCOUNT'],
            partition=RUN_CONSTANTS['SLURM_PARTITION'],
            jobtime=jobtime,
            mem_gb=64,
            job_id_start=1,
            conda_env_name="merging_comp",
            volta32=False,
            debug_mode=DEBUG_MODE,
            dry_mode=DRY_MODE,
            add_name='end',
            DIR_PATH=ROOT_FOLDER,
            #constraints=["a40|a100|titan"]
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
    parser.add_argument('--seed', nargs="+", type=int)

    parser.add_argument('--train_file', type=str)
    parser.add_argument('--valid_file', type=str)
    parser.add_argument('--test_file', type=str)
    parser.add_argument('--lora', action='store_true')

    parser.add_argument('--slurm', action='store_true')
    parser.add_argument('--disable_wandb', action='store_true')
    parser.add_argument('--dataset', type=str, choices=['multi_news', 'wiki_summary'], default=None)
    parser.add_argument('--dataset_config', type=str, default=None)
    parser.add_argument('--identifier', type=str, default=None)

    args = parser.parse_args()
    kwargs = vars(args)
    print(kwargs)
    main(**kwargs)
