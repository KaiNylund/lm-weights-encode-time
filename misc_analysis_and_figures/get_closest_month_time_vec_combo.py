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


def get_model_flattened_weights(model, param=None):
    model_sd = model.state_dict()
    model_weights = []
    with torch.no_grad():
        if param != None:
            model_weights.append(model_sd[param].detach().numpy().flatten())
        else:
            for param_name in model_sd.keys():
                model_weights.append(model_sd[param_name].detach().numpy().flatten())
        return np.hstack(np.array(model_weights))


vanilla_t5_small = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
task_vecs = []
prev_added_model = None
print("Loading task vectors...")
for year in tqdm(range(2012, 2014)):
    for month in range(12):
        if os.path.exists(f"./t5-small_vecs/wmt_lm/{year}_{month}"):
            month_model = AutoModelForSeq2SeqLM.from_pretrained(f"./t5-small_vecs/wmt_lm/{year}_{month}")
            task_vecs.append(month_model)
            prev_added_model = month_model
        else:
            print(f"Missing {year}_{month} model, using duplicate prev model")
            task_vecs.append(prev_added_model)


month_param_to_fit = defaultdict(dict)
losses = []
params = []
param_names = list(vanilla_t5_small.state_dict().keys())
param_names.remove("lm_head.weight")
param_names.remove("shared.weight")
#param_names.remove("encoder.embed_tokens.weight")
key_param = None #"decoder.block.5.layer.2.DenseReluDense.wi.weight"
num_forecast_months = 12
num_target_months = 12

if key_param == None:
    forecast_keys = np.array(list(range(num_forecast_months))).reshape(-1, 1)
    target_keys = np.array(list(range(num_forecast_months, num_forecast_months + num_target_months))).reshape(-1, 1)
    #forecast_keys = np.array((list(range(24)) + list(range(36, 60)))).reshape(-1, 1)
    #target_keys = np.array(list(range(24, 36))).reshape(-1, 1)
#else:
#    forecast_keys = []
#    target_keys = []
#    for i in range(num_forecast_months):
#        forecast_keys.append(np.mean(task_vecs[i].state_dict()[key_param].detach().numpy(), axis=0).flatten())
#    for i in range(num_forecast_months, num_forecast_months + num_target_months):
#        target_keys.append(np.mean(task_vecs[i].state_dict()[key_param].detach().numpy(), axis=0).flatten())
#    forecast_keys = np.array(forecast_keys)
#    target_keys = np.array(target_keys)

print("Predicting params...")
for param in tqdm(param_names):
    #forecast_keys = []
    #forecast_targets = []
    month_forecasts = []
    month_targets = []
    for i in range(12): #(list(range(24)) + list(range(36, 60))):
        month_forecasts.append(get_model_flattened_weights(task_vecs[i], param=param))
    for i in range(12, 24):
        month_targets.append(get_model_flattened_weights(task_vecs[i], param=param))

    prev_month_param = month_forecasts[23]

    reg = RandomForestRegressor(n_estimators=16, n_jobs=8, random_state=123)
    reg.fit(forecast_keys, month_forecasts)

    print(param)
    for month in range(len(target_keys)):
        month_pred_param = np.array(reg.predict([target_keys[month]])).flatten()
        print(month, 
              round(cos_sim(month_targets[month], month_pred_param), 4),
              round(cos_sim(month_targets[month], prev_month_param), 4))

        pred_params_path = f"./predicted_models/ind_emb_random_forest_16_wmt_lm_2013_{month}.npy"
        if os.path.exists(pred_params_path):
            month_param_to_fit = np.load(pred_params_path, allow_pickle=True).item()
        else:
            month_param_to_fit = {}

        month_param_to_fit[str(param)] = month_pred_param
        np.save(pred_params_path, month_param_to_fit)



print("Saving models...")
vanilla_t5_tokenizer = AutoTokenizer.from_pretrained("t5-small")
vanilla_t5 = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
vanilla_t5_small_sd = vanilla_t5.state_dict()


pred_year = 2014
for pred_month in range(12):
    sd_path = f"ind_emb_random_forest_16_wmt_lm_2013_{pred_year}_{pred_month}"
    
    predicted_params = np.load("./predicted_models/" + sd_path + ".npy", allow_pickle=True).item()
    #reshaped_pred_embs = vanilla_t5_small_sd["shared.weight"] + torch.from_numpy(predicted_params["encoder.embed_tokens.weight"].reshape(list(vanilla_t5_small_sd["shared.weight"].size())))
    #print(reshaped_pred_embs.size())

    #prev_year_month_model = AutoModelForSeq2SeqLM.from_pretrained(f"KaiNylund/t5-60M-lm-wmt-2012-{pred_month}")

    predicted_sd = deepcopy(vanilla_t5_small_sd)
    predicted_params["lm_head.weight"] = predicted_params["encoder.embed_tokens.weight"]
    predicted_params["shared.weight"] = predicted_params["encoder.embed_tokens.weight"]
    for param_name in predicted_params.keys():
        predicted_params[param_name] = predicted_params[param_name].reshape(list(vanilla_t5_small_sd[param_name].size()))
        predicted_sd[param_name] += torch.from_numpy(predicted_params[param_name])

    #prev_year_month_model = AutoModelForSeq2SeqLM.from_pretrained(f"KaiNylund/t5-60M-lm-wmt-2015-{pred_month}")
    #predicted_model = deepcopy(prev_year_month_model)
    #predicted_sd = predicted_model.state_dict()
    #predicted_sd["encoder.embed_tokens.weight"] = reshaped_pred_embs
    #predicted_sd["lm_head.weight"] = reshaped_pred_embs
    #predicted_sd["shared.weight"] = reshaped_pred_embs
    predicted_model = deepcopy(prev_year_month_model)
    predicted_model.load_state_dict(predicted_sd)
    predicted_model.save_pretrained("./predicted_models/" + sd_path + "/")
    vanilla_t5_tokenizer.save_pretrained("./predicted_models/" + sd_path + "/")
