import os
import torch
import cvxpy as cp
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
#from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, SGDRegressor
#from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

def cos_sim(A, B):
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))


def get_model_flattened_weights(model, param=None, mean=False):
    model_sd = model.state_dict()
    model_weights = []
    with torch.no_grad():
        if param != None:
            param_mat = model_sd[param].detach().numpy()
            if mean:
                model_weights.append(np.mean(param_mat, axis=0).flatten())
            else:
                model_weights.append(param_mat.flatten())
        else:
            for param_name in model_sd.keys():
                model_weights.append(model_sd[param_name].detach().numpy().flatten())

        return np.hstack(np.array(model_weights))


def get_indicator_embs(model, emb_idxs):
    indicator_embs = []
    model_sd = model.state_dict()
    lm_embs = model_sd["shared.weight"].detach().numpy()
    #indicator_embs.append(np.mean(lm_embs, axis=0))
    for i in emb_idxs:
        indicator_embs.append(lm_embs[i])
    return np.array(indicator_embs).flatten()

'''
vanilla_t5_small = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
task_vecs = []
lm_vecs = []
print("Loading task vectors...")
for year in tqdm(range(2012, 2017)):
    lm_vecs.append(AutoModelForSeq2SeqLM.from_pretrained(f"./t5-small_vecs/wmt_lm/{year}"))
    task_vecs.append(AutoModelForSeq2SeqLM.from_pretrained(f"./t5-small_vecs/news_sum/{year}"))

month_param_to_fit = defaultdict(dict)
param_names = list(vanilla_t5_small.state_dict().keys())
param_names.remove("lm_head.weight")
param_names.remove("shared.weight")
#param_names.remove("encoder.embed_tokens.weight")
key_param = None #"decoder.block.5.layer.2.DenseReluDense.wi.weight"

month_param_to_fit = {}
pred_params_path = "ind_emb_random_forest_16_news_sum_wmt"
print("Predicting params...")
for param in ["encoder.embed_tokens.weight"]: #tqdm(param_names):
    year_forecasts = []
    forecast_keys = []
    for i in range(2): #(list(range(24)) + list(range(36, 60))):
        year_forecasts.append(get_model_flattened_weights(task_vecs[i], param=param))
        #forecast_keys.append(get_model_flattened_weights(lm_vecs[i], param=param, mean=True))
        # (1) Use embeddings for _Trump, _2020, _2019, and _Bitcoin as indicators for year-to-year shifts
        # (2) Use embeddings for _2015, _2016, _2017, _2018, _2019, _2020 as indicators for year-to-year shifts
        # (3) _2012, _2013, _2014, _2015, _2016, _2017
        forecast_keys.append(get_indicator_embs(lm_vecs[i], [1673, 2038, 1412, 1230, 1420, 1233]))


    print(param, forecast_keys[0].shape)
    for pred_year in [2014, 2015, 2016]:
        pred_idx = pred_year - 2012

        year_target = get_model_flattened_weights(task_vecs[pred_idx], param=param)
        #target_key = get_model_flattened_weights(lm_vecs[-1], param=param, mean=True)
        target_key = get_indicator_embs(lm_vecs[pred_idx], [1673, 2038, 1412, 1230, 1420, 1233])
        prev_year_param = year_forecasts[-1]


        reg = RandomForestRegressor(n_estimators=16, n_jobs=8, random_state=123)
        reg.fit(forecast_keys, year_forecasts)

        year_pred_param = np.array(reg.predict([target_key])).flatten()
        print(pred_year,
              round(cos_sim(year_target, year_pred_param), 4),
              round(cos_sim(year_target, prev_year_param), 4))

        month_param_to_fit[str(param)] = year_pred_param
        np.save(f"./predicted_models/{pred_params_path}_2013_pred_${pred_year}.npy", month_param_to_fit)

print("Saving models...")
vanilla_t5_tokenizer = AutoTokenizer.from_pretrained("t5-small")
vanilla_t5 = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
vanilla_t5_small_sd = vanilla_t5.state_dict()

for pred_year in [2014, 2015, 2016]:
    #sd_path = f"ind_emb_random_forest_16_news_sum_wmt_2016"
    sd_path = f"{pred_params_path}_2013_pred_{pred_year}"

    prev_model = AutoModelForSeq2SeqLM.from_pretrained(f"KaiNylund/t5-60M-news_sum-2013")
    prev_model_sd = prev_model.state_dict()
    predicted_params = np.load("./predicted_models/" + sd_path + ".npy", allow_pickle=True).item()
    predicted_sd = deepcopy(prev_model_sd)
    predicted_params["lm_head.weight"] = predicted_params["encoder.embed_tokens.weight"]
    predicted_params["shared.weight"] = predicted_params["encoder.embed_tokens.weight"]
    for param_name in predicted_params.keys():
        predicted_params[param_name] = predicted_params[param_name].reshape(list(vanilla_t5_small_sd[param_name].size()))
        predicted_sd[param_name] = vanilla_t5_small_sd[param_name] + torch.from_numpy(predicted_params[param_name])

    predicted_model = deepcopy(prev_model)
    predicted_model.load_state_dict(predicted_sd)
    predicted_model.save_pretrained("./predicted_models/" + sd_path + "/")
    vanilla_t5_tokenizer.save_pretrained("./predicted_models/" + sd_path + "/")
'''

vanilla_t5_tokenizer = AutoTokenizer.from_pretrained("t5-small")
vanilla_t5 = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
vanilla_t5_small_sd = vanilla_t5.state_dict()
sd_path = f"ind_emb_random_forest_16_news_sum_wmt_2016"
predicted_params = np.load("./predicted_models/" + sd_path + ".npy", allow_pickle=True).item()
prev_task_vec = AutoModelForSeq2SeqLM.from_pretrained(f"./t5-small_vecs/news_sum/2015")
prev_task_sd = prev_task_vec.state_dict()
predicted_vec_sd = deepcopy(prev_task_sd)
predicted_params["lm_head.weight"] = predicted_params["encoder.embed_tokens.weight"]
predicted_params["shared.weight"] = predicted_params["encoder.embed_tokens.weight"]
for param in predicted_params.keys():
    predicted_vec_sd[param] = torch.from_numpy(predicted_params[param].reshape(list(vanilla_t5_small_sd[param].size())))

predicted_model = deepcopy(prev_task_vec)
predicted_model.load_state_dict(predicted_vec_sd)
predicted_model.save_pretrained("./t5-small_vecs/predicted_2016")
vanilla_t5_tokenizer.save_pretrained("./t5-small_vecs/predicted_2016")
