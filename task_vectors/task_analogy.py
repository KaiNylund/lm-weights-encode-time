import os
import sys
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from copy import deepcopy
import torch
from pathlib import Path
from peft import PeftConfig, PeftModel, get_peft_model, LoraConfig, TaskType
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
    parser.add_argument("--path_to_source_model", type=str)
    parser.add_argument("--task_vector_A", type=str)
    parser.add_argument("--task_vector_B", type=str)
    parser.add_argument("--task_vector_C", type=str)
    parser.add_argument("--lambdaA", type=float, nargs="+")
    parser.add_argument("--lambdaB", type=float, nargs="+")
    parser.add_argument("--lambdaC", type=float, nargs="+")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--lora", action='store_true')
    args = parser.parse_args()


    print('loading models...')
    torch.cuda.empty_cache()
    if args.lora and os.path.exists(args.task_vector_A + "/adapter_config.json"):
        print("using pretrained lora vectors")
        source_model = AutoModelForSeq2SeqLM.from_pretrained(args.path_to_source_model).eval()
        # Doing task vetor arithmetic in a funky order to minimize the number
        # of models we need to load into ram
        config = PeftConfig.from_pretrained(args.task_vector_A)
        print("Scaling A...")
        task_vector_A = PeftModel.from_pretrained(source_model, args.task_vector_A)
        task_vector_A = task_vector_A.merge_and_unload()
        task_vector_A.load_state_dict(_scale_vector(task_vector_A, alpha=args.lambdaA))

        print("Scaling B...")
        task_vector_B = PeftModel.from_pretrained(source_model, args.task_vector_B)
        task_vector_B = task_vector_B.merge_and_unload()
        task_vector_B.load_state_dict(_scale_vector(task_vector_B, alpha=args.lambdaB))
        del source_model
        torch.cuda.empty_cache()

        print("B - A")
        task_vectorAB_dict = _task_op_state_dict(task_vector_B.state_dict(), task_vector_A.state_dict(), op="forget", alpha=1.0)
        del task_vector_A
        del task_vector_B
        torch.cuda.empty_cache()

        print("Scaling C...")
        source_model = AutoModelForSeq2SeqLM.from_pretrained(args.path_to_source_model).eval()
        task_vector_C = PeftModel.from_pretrained(source_model, args.task_vector_C)
        task_vector_C = task_vector_C.merge_and_unload()
        task_vector_C.load_state_dict(_scale_vector(task_vector_C, alpha=args.lambdaC))

        task_vectorAB = deepcopy(source_model)
        task_vectorAB.load_state_dict(task_vectorAB_dict)
        del source_model
        del task_vectorAB_dict
        torch.cuda.empty_cache()

    else:
        print("Scaling A...")
        task_vector_A = AutoModelForSeq2SeqLM.from_pretrained(args.task_vector_A).eval()
        task_vector_A_dict = _scale_vector(task_vector_A, alpha=args.lambdaA)
        del task_vector_A
        torch.cuda.empty_cache()

        print("Scaling B...")
        task_vector_B = AutoModelForSeq2SeqLM.from_pretrained(args.task_vector_B).eval()
        task_vector_B_dict = _scale_vector(task_vector_B, alpha=args.lambdaB)
        del task_vector_B
        torch.cuda.empty_cache()

        print("B - A")
        task_vectorAB_dict = _task_op_state_dict(task_vector_B_dict, task_vector_A_dict, op="forget", alpha=1.0)
        del task_vector_A_dict
        del task_vector_B_dict
        torch.cuda.empty_cache()

        print("Scaling C...")
        task_vector_C = AutoModelForSeq2SeqLM.from_pretrained(args.task_vector_C).eval()
        task_vector_C_dict = _scale_vector(task_vector_C, alpha=args.lambdaC)
        del task_vector_C
        torch.cuda.empty_cache()


    print("C + B - A")
    task_vectorCBA_dict = _task_op_state_dict(task_vector_C_dict, task_vectorAB_dict, op="add", alpha=1.0)
    del task_vector_C_dict
    del task_vectorAB_dict
    torch.cuda.empty_cache()

    print("t5 + C + B - A")
    source_model = AutoModelForSeq2SeqLM.from_pretrained(args.path_to_source_model).eval()
    new_model_dict = _task_op_state_dict(source_model.state_dict(), task_vectorCBA_dict, op="add", alpha=1.0)
    del task_vectorCBA_dict
    del source_model
    torch.cuda.empty_cache()

    new_model = AutoModelForSeq2SeqLM.from_pretrained(args.path_to_source_model).eval()
    new_model.load_state_dict(new_model_dict)
    tokenizer = AutoTokenizer.from_pretrained(args.path_to_source_model)

    if not Path(args.output_dir).is_dir():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    #if args.lora:
    #    lora_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
    #    new_model = get_peft_model(new_model, lora_config)
    new_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


