import numpy as np
import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from collections import defaultdict
from tqdm import tqdm
from peft import PeftConfig, PeftModel

umap_reducer = umap.UMAP(min_dist=0.4, n_neighbors=10, metric="cosine")
tsne_reducer = TSNE(n_components=2, learning_rate='auto', init='random')
pca_reducer = PCA(n_components=2)

aic_times = ["2006-2008", "2009-2011", "2012-2014", "2015-2017", "2018-2020"]

tasks = ["news_sum", "news_cls", "poli_aff", "aic", "wmt_lm", "twitter_lm", "arxiv_lm"]
times = [list(range(2012, 2017)), list(range(2012, 2017)), list(range(2015, 2021)), 
        aic_times, list(range(2012, 2021)), list(range(2015, 2021)), aic_times]
model_names = ["t5-small"]#, "t5-large", "t5-3b"]

model_to_proj_params = {}
for model_name in model_names:
    if model_name == "t5-small":
        num_layers = 6
        #combine_params = ["decoder.final_layer_norm.weight", "decoder.block.5.layer.1.EncDecAttention.v.weight"]
        param_names = ["encoder.final_layer_norm.weight", "decoder.final_layer_norm.weight",
                    "encoder.block.5.layer.0.SelfAttention.v.weight",
                    "decoder.block.5.layer.1.EncDecAttention.v.weight",
                    "encoder.block.5.layer.1.DenseReluDense.wi.weight",
                    "decoder.block.5.layer.2.DenseReluDense.wi.weight",
                    "encoder.block.5.layer.1.DenseReluDense.wo.weight",
                    "decoder.block.5.layer.2.DenseReluDense.wo.weight"]
    elif model_name == "t5-large":
        num_layers = 24
        #combine_params = ["decoder.final_layer_norm.weight", "decoder.block.23.layer.1.EncDecAttention.v.weight"]
        param_names = ["encoder.final_layer_norm.weight", "decoder.final_layer_norm.weight",
                    "encoder.block.23.layer.0.SelfAttention.v.weight",
                    "decoder.block.23.layer.1.EncDecAttention.v.weight",
                    "encoder.block.23.layer.1.DenseReluDense.wi.weight",
                    "decoder.block.23.layer.2.DenseReluDense.wi.weight",
                    "encoder.block.23.layer.1.DenseReluDense.wo.weight",
                    "decoder.block.23.layer.2.DenseReluDense.wo.weight"]
    elif model_name == "t5-3b":
        num_layers = 24
        #combine_params = ["decoder.final_layer_norm.weight", "decoder.block.23.layer.1.EncDecAttention.v.weight"]
        param_names = ["encoder.final_layer_norm.weight", "decoder.final_layer_norm.weight",
                    "encoder.block.23.layer.0.SelfAttention.v.weight",
                    "decoder.block.23.layer.1.EncDecAttention.v.weight",
                    "encoder.block.23.layer.1.DenseReluDense.wi.weight",
                    "decoder.block.23.layer.2.DenseReluDense.wi.weight",
                    "encoder.block.23.layer.1.DenseReluDense.wo.weight",
                    "decoder.block.23.layer.2.DenseReluDense.wo.weight"]
        
    all_vecs_path = f"./{model_name}_vecs/"

    time_vec_proj_params = defaultdict(list)
    for i, task in enumerate(tasks):
        time_vec_path = all_vecs_path + task + "/" #PATH_TO_FOLDERS + f"{path_model_name}_{task}_{time}"

        for time in times[i]:
            model_path = time_vec_path + str(time)
            print(model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            #for name, param in model.named_parameters():
            #    print(name, param.size())
            model_params = dict(model.named_parameters())
            #proj_weights = np.concatenate((model_params[param_name1].detach().numpy().flatten(),
            #                               model_params[param_name2].detach().numpy().flatten()))
            for param in param_names:
                if str(num_layers - 1) in param:
                    #print("using all layers")
                    param_weights = []
                    for i in range(num_layers):
                        layer_param = param.replace(str(num_layers - 1), str(i))
                        layer_param_weights = model_params[layer_param].detach().numpy().flatten()
                        param_weights.append(layer_param_weights)
                    param_weights = np.mean(param_weights, axis=0)
                else:
                    param_weights = model_params[param].detach().numpy().flatten()
                
                time_vec_proj_params[param].append(param_weights)
            del model
            del model_params

    #time_vec_proj_params["combined"] = np.concatenate([time_vec_proj_params[p] for p in combine_params], axis=1)
    model_to_proj_params[model_name] = time_vec_proj_params

#model_to_proj_params["combined_models"] = {}
#model_to_proj_params["combined_models"]["combined_EncDecAttention.v.weight"] = \
#    np.concatenate([model_to_proj_params["t5-small"]["decoder.block.5.layer.1.EncDecAttention.v.weight"],
#                    model_to_proj_params["t5-large"]["decoder.block.23.layer.1.EncDecAttention.v.weight"],
#                    model_to_proj_params["t5-3b"]["decoder.block.23.layer.1.EncDecAttention.v.weight"]], axis=1)
#model_to_proj_params["combined_models"]["combined_decoder.final_layer_norm.weight"] = \
#    np.concatenate([model_to_proj_params[m]["decoder.final_layer_norm.weight"] for m in model_names], axis=1)


for model_name in model_to_proj_params.keys():
    param_vec_projections = {}
    param_to_projs = {}
    model_vec_proj_params = model_to_proj_params[model_name]
    for param in model_vec_proj_params.keys():
        param_vec_weights = np.array(model_vec_proj_params[param])
        print(param_vec_weights.shape)

        param_vec_projections[param + "_umap"] = umap_reducer.fit_transform(param_vec_weights)
        param_vec_projections[param + "_tsne"] = tsne_reducer.fit_transform(param_vec_weights)
        param_vec_projections[param + "_pca"] = pca_reducer.fit_transform(param_vec_weights)
        #print(param_vec_projections[param].shape)

        for proj_type in ["umap", "tsne", "pca"]:
            task_to_projs = {}
            pts_start = 0
            for i, task in enumerate(tasks):
                num_pts = len(times[i])
                task_to_projs[task] = param_vec_projections[param + "_" + proj_type][pts_start:(pts_start + num_pts)]
                pts_start += num_pts

            param_to_projs[param + "_" + proj_type] = task_to_projs

    np.save(f"./projections/{model_name}_time_vec_layer_projections", param_to_projs)