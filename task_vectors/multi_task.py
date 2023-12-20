import sys
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from copy import deepcopy
import torch
from pathlib import Path
import argparse

def _scale_vector(task_vector,  alpha=1.0):
    tv_sd =  task_vector.state_dict()
    with torch.no_grad():
        merged = {}
        for key in tv_sd:
            if tv_sd[key].dtype == torch.int64 or tv_sd[key].dtype == torch.uint8:
                merged[key] = tv_sd[key]
            else:
                merged[key] = alpha * tv_sd[key]
        return merged

def scale_vector(task_vector, alpha=1.0):
    res = deepcopy(task_vector)
    new_model = _scale_vector(task_vector, alpha=alpha)
    res.load_state_dict(new_model)
    return res


def _task_op(source_model, task_vector, op='add', alpha=1.0):
    pre_sd, tv_sd = source_model.state_dict(), task_vector.state_dict()
    return _task_op_state_dict(pre_sd, tv_sd)
    
def _task_op_state_dict(pre_sd, tv_sd, op='add', alpha=1.0):
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
    parser.add_argument("--task_vectors", nargs='+')
    parser.add_argument("--lambdas", nargs='+')
    parser.add_argument("--n_pct_pts", type=int, default=None)
    parser.add_argument("--output_dir")
    parser.add_argument("--inv_decay", action="store_true")
    args = parser.parse_args()


    print('loading models...')
    print(args.task_vectors)

    if args.n_pct_pts != None:
        if args.n_pct_pts == 1:
            lambdas = [0.0, 1.0]
        elif args.inv_decay:
            lambdas = [1.0 / int(args.n_pct_pts), 1.0 - (1.0 / int(args.n_pct_pts))]
        else:
            lambdas = [1.0 - (1.0 / int(args.n_pct_pts)), 1.0 / int(args.n_pct_pts)]
    else:
        lambdas = args.lambdas

    print("lambdas: " + str(lambdas))

    print("B + A, num vectors: " + str(len(args.task_vectors)))
    model = AutoModelForSeq2SeqLM.from_pretrained(args.task_vectors[0]).eval()
    task_vec_dict = _scale_vector(model, alpha=float(lambdas[0]))
    del model
    torch.cuda.empty_cache()
    for i in range(1, len(args.task_vectors)):
        cur_model = AutoModelForSeq2SeqLM.from_pretrained(args.task_vectors[i]).eval()
        cur_task_vec_dict = _scale_vector(cur_model, alpha=float(lambdas[i]))
        del cur_model
        torch.cuda.empty_cache()
        task_vec_dict = _task_op_state_dict(task_vec_dict, cur_task_vec_dict, op="add", alpha=1.0)
        del cur_task_vec_dict
        torch.cuda.empty_cache()


    print("t5 + B + A")
    source_model = AutoModelForSeq2SeqLM.from_pretrained(args.path_to_source_model).eval()
    new_model_dict = _task_op_state_dict(source_model.state_dict(), task_vec_dict, op="add", alpha=1.0)
    del task_vec_dict
    torch.cuda.empty_cache()
    new_model = source_model
    new_model.load_state_dict(new_model_dict)

    tokenizer = AutoTokenizer.from_pretrained(args.path_to_source_model)

    if not Path(args.output_dir).is_dir():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    new_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


