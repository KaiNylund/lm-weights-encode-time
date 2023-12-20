from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftConfig, PeftModel, LoraConfig, TaskType, get_peft_model
from pathlib import Path
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_to_swap")
    parser.add_argument("--embedding_model")
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument("--output_dir")
    parser.add_argument("--lora", action='store_true')
    parser.add_argument("--swap_mode", type=str, default="embeddings", choices=["embeddings", "ff_layers", "attn", "norm"])
    args = parser.parse_args()
    

    print('loading models...')
    if args.lora:
        lora_config = PeftConfig.from_pretrained(args.model_to_swap)
        lora_base_model = AutoModelForSeq2SeqLM.from_pretrained(lora_config.base_model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(lora_config.base_model_name_or_path)
        model_to_swap = PeftModel.from_pretrained(lora_base_model, args.model_to_swap)
        model_to_swap = model_to_swap.merge_and_unload().eval()
        embedding_model = PeftModel.from_pretrained(lora_base_model, args.embedding_model)
        embedding_model = embedding_model.merge_and_unload().eval()
        del lora_base_model
        del lora_config
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_to_swap)
        model_to_swap = AutoModelForSeq2SeqLM.from_pretrained(args.model_to_swap).eval()
        embedding_model = AutoModelForSeq2SeqLM.from_pretrained(args.embedding_model).eval()

    model_to_swap_sd = model_to_swap.state_dict()

    if args.swap_mode == "embeddings":
        new_embeddings = (args.alpha * embedding_model.state_dict()["shared.weight"]) + \
                        ((1.0 - args.alpha) * model_to_swap_sd["shared.weight"]) 
        model_to_swap_sd["encoder.embed_tokens.weight"] = new_embeddings
        model_to_swap_sd["lm_head.weight"] = new_embeddings
        model_to_swap_sd["shared.weight"] = new_embeddings

    elif args.swap_mode == "ff_layers":
        emb_sd = embedding_model.state_dict()
        for param_name in model_to_swap_sd.keys():
            if "DenseReluDense" in param_name:
                model_to_swap_sd[param_name] = args.alpha * emb_sd[param_name]

    elif args.swap_mode == "attn":
        emb_sd = embedding_model.state_dict()
        for param_name in model_to_swap_sd.keys():
            if "SelfAttention" in param_name:
                model_to_swap_sd[param_name] = args.alpha * emb_sd[param_name]
    
    elif args.swap_mode == "norm":
        emb_sd = embedding_model.state_dict()
        for param_name in model_to_swap_sd.keys():
            if "norm" in param_name.lower():
                model_to_swap_sd[param_name] = args.alpha * emb_sd[param_name]


    model_to_swap.load_state_dict(model_to_swap_sd)

    if not Path(args.output_dir).is_dir():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.lora:
        lora_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
        model_to_swap = get_peft_model(model_to_swap, lora_config)
        
    model_to_swap.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


