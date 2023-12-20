import sys
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from copy import deepcopy
from peft import PeftConfig, PeftModel, LoraConfig, TaskType, get_peft_model
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

def _get_task_vector(pretrained_model, finetuned_model, alpha=1.0):
    pre_sd, ft_sd = pretrained_model.state_dict(), finetuned_model.state_dict()
    with torch.no_grad():
        merged = {}
        for key in ft_sd:
            if ft_sd[key].shape != pre_sd[key].shape:
                import pdb; pdb.set_trace()
                # f"{key} is not the same shape between models"
            if pre_sd[key].dtype == torch.int64 or pre_sd[key].dtype == torch.uint8:
                merged[key] = pre_sd[key]
            merged[key] = alpha * (ft_sd[key] - pre_sd[key])
        return merged

def is_same_model(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.isclose(key_item_1[1], key_item_2[1]).all().item():
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                from fairseq import pdb; pdb.set_trace()
                print('Mismatch found at', key_item_1[0])
    if models_differ == 0:
        return True
    else:
        return False

def get_task_vector(pretrained_model, finetuned_model, alpha=None):
    res = deepcopy(pretrained_model)
    task_vec = _get_task_vector(pretrained_model,finetuned_model, alpha=alpha)
    res.load_state_dict(task_vec)
    #candidate_ft = task_op(pretrained_model, res, op='add', alpha=alpha)
    # assert is_same_model(candidate_ft, finetuned_model)
    
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_pretrained_model")
    parser.add_argument("--path_to_finetuned_model")
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument("--output_dir")
    parser.add_argument("--lora", action='store_true')
    parser.add_argument("--save_lora", action='store_true')
    args = parser.parse_args()


    print('loading models...')
    pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(args.path_to_pretrained_model).eval()
    if args.lora:
        finetuned_config = PeftConfig.from_pretrained(args.path_to_finetuned_model)
        finetuned_model = AutoModelForSeq2SeqLM.from_pretrained(finetuned_config.base_model_name_or_path)
        finetuned_model = PeftModel.from_pretrained(finetuned_model, args.path_to_finetuned_model)
        finetuned_model = finetuned_model.merge_and_unload()
    else:
        finetuned_model = AutoModelForSeq2SeqLM.from_pretrained(args.path_to_finetuned_model).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.path_to_pretrained_model, model_max_length=1024)

    print('getting task vector...')
    task_vec_dict = _get_task_vector(pretrained_model, finetuned_model, alpha=args.alpha)
    if not Path(args.output_dir).is_dir():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    del pretrained_model
    del finetuned_model
    torch.cuda.empty_cache()

    task_vec = AutoModelForSeq2SeqLM.from_pretrained(args.path_to_pretrained_model).eval()
    task_vec.load_state_dict(task_vec_dict)

    if args.save_lora:
        lora_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
        task_vec = get_peft_model(task_vec, lora_config)

    task_vec.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


