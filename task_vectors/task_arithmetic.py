import sys
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from copy import deepcopy
import torch
from pathlib import Path
import argparse


def _task_op(source_model, task_vector, op='add', alpha=1.0):
    pre_sd, tv_sd = source_model.state_dict(), task_vector.state_dict()
    with torch.no_grad():
        merged = {}
        for key in tv_sd:
            assert tv_sd[key].shape == pre_sd[key].shape, f"{key} is not the same shape between models"
            if pre_sd[key].dtype == torch.int64 or pre_sd[key].dtype == torch.uint8:
                merged[key] = pre_sd[key]
            if op == 'add':
                merged[key] = pre_sd[key] + alpha * tv_sd[key]
            else:
                merged[key] = pre_sd[key] - alpha * tv_sd[key]
        return merged

def task_op(source_model, task_vector, op='add', alpha=1.0):
    res = deepcopy(source_model)
    new_model = _task_op(source_model,task_vector, op=op, alpha=alpha)
    res.load_state_dict(new_model)
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_source_model")
    parser.add_argument("--path_to_task_vector")
    parser.add_argument("--op", choices=['add', 'forget'])
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument("--output_dir")
    args = parser.parse_args()


    print('loading models...')
    source_model = AutoModelForSeq2SeqLM.from_pretrained(args.path_to_source_model).eval()

    task_vector = AutoModelForSeq2SeqLM.from_pretrained(args.path_to_task_vector).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.path_to_source_model)

    new_model = task_op(source_model, task_vector, op=args.op, alpha=args.alpha)

    if not Path(args.output_dir).is_dir():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    new_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


