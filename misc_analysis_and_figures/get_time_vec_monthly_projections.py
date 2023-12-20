import numpy as np
import umap
import os
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from transformers import AutoModelForSeq2SeqLM
from collections import defaultdict
from tqdm import tqdm
from task_vectors.get_task_vector import get_task_vector


# Assumes both models have the same params
def get_model_flattened_weights(model):
    model_sd = model.state_dict()
    model_weights = []
    with torch.no_grad():
        for param_name in model_sd.keys():
            model_weights.append(model_sd[param_name].detach().numpy().flatten())

        return np.hstack(np.array(model_weights))


# Returns a dict of param_name -> list of vec param projection in same order as proj_model_paths
# reduces using the given reducer
def project_vec_params(out_dir, base_model_name, proj_model_paths, params, reducer, num_layers=6):
    pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name).eval()
    vec_proj_param_weights = defaultdict(list)
    # Build dict of all params to project
    for model_path in proj_model_paths:
        if "KaiNylund" not in model_path and not os.path.exists(model_path):
            print("missing " + model_path)
            continue
        
        print(model_path)
        proj_model = AutoModelForSeq2SeqLM.from_pretrained(model_path).eval()
        proj_vec = get_task_vector(pretrained_model, proj_model, alpha=1.0)
        #for name, param in month_vec.named_parameters():
        #    print(name, param.size())
        model_params = dict(proj_vec.named_parameters())
        #proj_weights = np.concatenate((model_params[param_name1].detach().numpy().flatten(),
        #                               model_params[param_name2].detach().numpy().flatten()))
        for param in params:
            if str(num_layers - 1) in param:
                param_weights = []
                for i in range(num_layers):
                    layer_param = param.replace(str(num_layers - 1), str(i))
                    layer_param_weights = model_params[layer_param].detach().numpy().flatten()
                    param_weights.append(layer_param_weights)
                param_weights = np.mean(param_weights, axis=0)
            elif param == "all":
                param_weights = get_model_flattened_weights(proj_vec)
            else:
                param_weights = model_params[param].detach().numpy().flatten()
            
            vec_proj_param_weights[param].append(param_weights)
        del proj_model
        del model_params

    # Actually do all the projecting with the given reducer
    vec_param_projections = {}
    for param, vec_weights in vec_proj_param_weights.items():
        vec_weights = np.array(vec_weights)
        print(vec_weights.shape)
        vec_param_projections[param] = reducer.fit_transform(vec_weights)

    np.save(out_dir, vec_param_projections)


model_name_to_num_layers = {
    "t5-small": 6,
    "t5-large": 24,
    "t5-3b": 24
}

model_name_to_proj_params = {
    "t5-small": ['shared.weight', "encoder.final_layer_norm.weight", 
                 "decoder.final_layer_norm.weight",
                 "encoder.block.5.layer.0.SelfAttention.v.weight",
                 "decoder.block.5.layer.1.EncDecAttention.v.weight",
                 "encoder.block.5.layer.1.DenseReluDense.wi.weight",
                 "decoder.block.5.layer.2.DenseReluDense.wi.weight",
                 "encoder.block.5.layer.1.DenseReluDense.wo.weight",
                 "decoder.block.5.layer.2.DenseReluDense.wo.weight"],
    "t5-large": ["encoder.final_layer_norm.weight", "decoder.final_layer_norm.weight",
                 "encoder.block.23.layer.0.SelfAttention.v.weight",
                 "decoder.block.23.layer.1.EncDecAttention.v.weight",
                 "encoder.block.23.layer.1.DenseReluDense.wi.weight",
                 "decoder.block.23.layer.2.DenseReluDense.wi.weight",
                 "encoder.block.23.layer.1.DenseReluDense.wo.weight",
                 "decoder.block.23.layer.2.DenseReluDense.wo.weight"],
    "t5-3b": ["encoder.final_layer_norm.weight", "decoder.final_layer_norm.weight",
              "encoder.block.23.layer.0.SelfAttention.v.weight",
              "decoder.block.23.layer.1.EncDecAttention.v.weight",
              "encoder.block.23.layer.1.DenseReluDense.wi.weight",
              "decoder.block.23.layer.2.DenseReluDense.wi.weight",
              "encoder.block.23.layer.1.DenseReluDense.wo.weight",
              "decoder.block.23.layer.2.DenseReluDense.wo.weight"]
}


if __name__ == "__main__":
    umap_reducer = umap.UMAP()
    tsne_reducer = TSNE(n_components=2, learning_rate='auto', init='random')
    pca_reducer = PCA(n_components=2)

    model_name = "t5-small"
    task = "poli_aff"
    proj_model_paths = []
    for year in range(2015, 2021):
        for month in range(12):
            proj_model_paths.append(f"./models/{model_name}_{task}_linear_monthly_updating/up_to_{year}_{month}_interp")

        proj_model_paths.append(f"KaiNylund/t5-60M-poli_aff-{year}")

    num_layers = model_name_to_num_layers[model_name]
    param_names = model_name_to_proj_params[model_name]
    out_dir = f"./projections/{model_name}_{task}_linear_monthly_updating_projections"
    project_vec_params(out_dir, model_name, proj_model_paths, param_names, umap_reducer, num_layers=num_layers)