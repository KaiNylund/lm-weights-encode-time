from transformers import AutoModelForSeq2SeqLM
from collections import defaultdict
from tqdm import tqdm
from task_vectors.get_task_vector import get_task_vector
import numpy as np
import torch
import os

task_to_years = {
    #"news_sum": list(range(2012, 2017)),
    #"news_cls": list(range(2012, 2017)),
    "wmt_lm": list(range(2012, 2017)),
    "poli_aff": list(range(2015, 2021)),
    #"twitter_lm": list(range(2015, 2021)),
    #"aic": ["2006-2008", "2009-2011", "2012-2014", "2015-2017", "2018-2020"],
    #"arxiv_lm": ["2006-2008", "2009-2011", "2012-2014", "2015-2017", "2018-2020"]
}


def l2_dist(A, B, axis=None):
  return np.linalg.norm(A - B, ord=2, axis=axis)**2

def cos_dist(A, B, axis=None):
    #print(A.shape, B.shape)
    return np.dot(A, B) / (np.linalg.norm(A, axis=axis) * np.linalg.norm(B, axis=axis))

def time_less(t1, t2, m1=None, m2=None):
    if isinstance(t1, str):
        t1 = int(t1.split("-")[0])
        t2 = int(t2.split("-")[0])
    if m1 == None or t1 != t2:
        return t1 < t2
    else:
        return m1 < m2


# Assumes both models have the same params
def get_model_dist(m1, m2, dist_func, param=None, axis=None):
    m1_sd = m1.state_dict()
    m2_sd = m2.state_dict()
    m1_weights = []
    m2_weights = []
    with torch.no_grad():
        if param == None:
            param_names = m1_sd.keys()
        elif param == "embeddings":
            param_names = ["shared.weight"]
        elif param == "ff_layers":
            param_names = [p for p in list(m1_sd.keys()) if "DenseReluDense" in p]
        elif param == "attn":
            param_names = [p for p in list(m1_sd.keys()) if "SelfAttention" in p]
        elif param == "norm":
            param_names = [p for p in list(m1_sd.keys()) if "norm" in p]
        else:
            param_names = [param]
        for param_name in param_names:
            m1_weights.append(m1_sd[param_name].detach().numpy().flatten())
            m2_weights.append(m2_sd[param_name].detach().numpy().flatten())
        vec_dist = dist_func(np.hstack(np.array(m1_weights)), np.hstack(np.array(m2_weights)), axis=axis)
        del m1_sd, m2_sd, m1_weights, m2_weights
        return vec_dist



# For getting monthly/yearly cos sim distances
'''
for model in ["t5-small"]: #["t5-large", "t5-3b"]:
    out_path = f"./model_distances/{model}_monthly_param_time_vec_distances"
    distances_dict = defaultdict(dict)
    for param_name in [None, "embeddings", "ff_layers", "attn", "norm"]:
        for task, years in tqdm(task_to_years.items()):
            N = len(years) * 12
            M = N
            time_cos_sims = np.ones((N, M))
            #time_l2_sims = np.ones((len(times), len(times)))
            for i, y1 in enumerate(years):
                for m1 in range(12):
                    t1_vec_path = f"{model}_vecs/{task}/{y1}_{m1}"
                    if not os.path.exists(t1_vec_path):
                        print("Missing model: " + t1_vec_path)
                        continue
                    t1_vec = AutoModelForSeq2SeqLM.from_pretrained(t1_vec_path)
                    for j, y2 in enumerate(years):
                        for m2 in range(12):
                            if not (m1 == m2 and y1 == y2) and time_less(y1, y2, m1, m2):
                                t2_vec_path = f"{model}_vecs/{task}/{y2}_{m2}"
                                if not os.path.exists(t2_vec_path):
                                    print("Missing model: " + t2_vec_path)
                                    continue
                                t2_vec = AutoModelForSeq2SeqLM.from_pretrained(t2_vec_path)
                                time_cos_dists = get_model_dist(t1_vec, t2_vec, cos_dist, param=param_name)
                                time_cos_sims[i * 12 + m1][j * 12 + m2] = time_cos_dists
                                time_cos_sims[j * 12 + m2][i * 12 + m1] = time_cos_dists
                                print(task, y1, y2, m1, m2, time_cos_sims[j * 12 + m2][i * 12 + m1])
                                del t2_vec
                    del t1_vec

            distances_dict[param_name][task + "_cos_dist"] = time_cos_sims
        #distances_dict[model][task + "_l2_dist"] = time_l2_sims
        np.save(out_path, distances_dict)
'''

# For getting distances between sequential updating checkpoints and year time vectors
sequential_ckpt_path_to_years = {
    "/models/t5-small_wmt_lm_linear_monthly_updating": list(range(2012, 2016)),
    "/models/t5-small_poli_aff_linear_monthly_updating": list(range(2015, 2021))
}


model = "t5-small"
out_path = f"/model_distances/{model}_linear_monthly_updating_ckpt_dists"
params = [None, "embeddings", "ff_layers", "attn", "norm"]
pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(model)
# Format output dictionary
distances_dict = defaultdict(dict)
for task, years in task_to_years.items():
    for param in params:
        distances_dict[task + "_cos_dist"][param] = defaultdict(list)

# Compute distances between every month ckpt and each of the eval year time vectors
for task, years in task_to_years.items():
    eval_time_vecs = {}
    ckpt_cos_sims = defaultdict(dict)
    for eval_year in years:
        eval_vec_path = f"./{model}_vecs/{task}/{eval_year}"
        eval_time_vecs[eval_year] = AutoModelForSeq2SeqLM.from_pretrained(eval_vec_path)

    for y1 in years:
        for m1 in range(12):
            t1_model_path = f"./models/{model}_{task}_linear_monthly_updating/up_to_{y1}_{m1}_interp"
            if not os.path.exists(t1_model_path):
                print(f"Missing model for: {y1}_{m1}")
                continue
            t1_model = AutoModelForSeq2SeqLM.from_pretrained(t1_model_path)
            t1_vec = get_task_vector(pretrained_model, t1_model, alpha=1.0)
            del t1_model

            for param_name in params:
                for eval_year in years:
                    ckpt_cos_dist = get_model_dist(t1_vec, eval_time_vecs[eval_year], cos_dist, param=param_name)
                    distances_dict[task + "_cos_dist"][param_name][eval_year].append(ckpt_cos_dist)
                    print(task, y1, m1, param_name, eval_year, ckpt_cos_dist)
            
            np.save(out_path, distances_dict)