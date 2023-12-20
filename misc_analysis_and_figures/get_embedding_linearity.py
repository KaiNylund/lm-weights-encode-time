from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftConfig, PeftModel
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import torch
import os
from task_vectors.get_task_vector import get_task_vector

model_size_names = ["t5-3b"]#, "t5-60M", "t5-770M",]
model_names = ["t5-3b"]#, "t5-small", "t5-large",]

task_to_times = {
    "news_sum": list(range(2012, 2017)),
    "news_cls": list(range(2012, 2017)),
    "wmt_lm": list(range(2012, 2017)),
    "poli_aff": list(range(2015, 2021)),
    "twitter_lm": list(range(2015, 2021)),
    "aic": ["2006-2008", "2009-2011", "2012-2014", "2015-2017", "2018-2020"],
    "arxiv_lm": ["2006-2008", "2009-2011", "2012-2014", "2015-2017", "2018-2020"]
}

for m, model_name in enumerate(model_names):
    for task, task_times in task_to_times.items():
        pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        emb_idx_to_token = {v: k for k, v in tokenizer.get_vocab().items()}

        all_time_model_embeddings = []
        all_time_vec_embeddings = []
        for time in task_times: #range(2012, 2022):
            #for month in range(12):
                #model_path = f"./models/{model_name}_lm_{year}_{month}_wmt"
            if "lm" in task:
                #model_path = f"./models/{model_name}_lm_{time}_{task.split('_')[0]}"
                model_path = f"KaiNylund/{model_size_names[m]}-lm-{task.split('_')[0]}-{time}"
            else:
                #model_path = f"./models/{model_name}_{task}_{time}"
                model_path = f"KaiNylund/{model_size_names[m]}-{task}-{time}"

            try:
                print(model_path)
                if model_name == "t5-small":
                    time_model = AutoModelForSeq2SeqLM.from_pretrained(model_path, use_auth_token=True)
                else:
                    config = PeftConfig.from_pretrained(model_path)
                    time_model = PeftModel.from_pretrained(pretrained_model, model_path)
                    time_model = time_model.merge_and_unload()
                time_vec = get_task_vector(pretrained_model, time_model, alpha=1.0)

                all_time_model_embeddings.append(time_model.state_dict()['shared.weight'].detach().numpy())
                all_time_vec_embeddings.append(time_vec.state_dict()['shared.weight'].detach().numpy())
                del time_model, time_vec
            except Exception as e:
                print("Missing " + model_path)
                print(e)


        all_time_model_embeddings = np.array(all_time_model_embeddings)
        all_time_vec_embeddings = np.array(all_time_vec_embeddings)
        #print(all_time_model_embeddings.shape, all_time_vec_embeddings.shape)
        np.save(f"{model_name}_{task}_yearly_model_embeddings", all_time_model_embeddings)
        np.save(f"{model_name}_{task}_yearly_vec_embeddings", all_time_vec_embeddings)


        all_month_model_emb_correlations = []
        all_month_vec_emb_correlations = []
        all_month_model_max_corrs = []
        all_month_vec_max_corrs = []
        all_month_model_max_corr_dists = []
        all_month_vec_max_corr_dists = []
        diagonal_mask = np.eye(all_time_model_embeddings.shape[0], dtype=bool)
        corr_matrix_shape = (all_time_model_embeddings.shape[0], all_time_model_embeddings.shape[0])
        for i in tqdm(range(all_time_model_embeddings.shape[1])):
            model_corr_matrix = np.corrcoef(all_time_model_embeddings[:, i, :])
            vec_corr_matrix = np.corrcoef(all_time_vec_embeddings[:, i, :])
            all_month_model_emb_correlations.append(model_corr_matrix)
            all_month_vec_emb_correlations.append(vec_corr_matrix)
            model_corr_matrix[diagonal_mask] = 0.0
            vec_corr_matrix[diagonal_mask] = 0.0
            model_max_corr_idx = np.unravel_index(np.argmax(model_corr_matrix, axis=None), corr_matrix_shape)
            vec_max_corr_idx = np.unravel_index(np.argmax(vec_corr_matrix, axis=None), corr_matrix_shape)
            all_month_model_max_corrs.append(model_corr_matrix[model_max_corr_idx])
            all_month_vec_max_corrs.append(vec_corr_matrix[vec_max_corr_idx])
            all_month_model_max_corr_dists.append(abs(model_max_corr_idx[0] - model_max_corr_idx[1]))
            all_month_vec_max_corr_dists.append(abs(vec_max_corr_idx[0] - vec_max_corr_idx[1]))


        all_month_model_emb_correlations = np.array(all_month_model_emb_correlations)
        all_month_vec_emb_correlations = np.array(all_month_vec_emb_correlations)
        #print(all_month_model_emb_correlations.shape, all_month_vec_emb_correlations.shape)

        all_month_model_max_corrs = np.array(all_month_model_max_corrs)
        all_month_vec_max_corrs = np.array(all_month_vec_max_corrs)
        all_month_model_max_corr_dists = np.array(all_month_model_max_corr_dists)
        all_month_vec_max_corr_dists = np.array(all_month_vec_max_corr_dists)

        avg_month_model_emb_correlations = np.nanmean(all_month_model_emb_correlations, axis=(1, 2))
        avg_month_vec_emb_correlations = np.nanmean(all_month_vec_emb_correlations, axis=(1, 2))
        #print(avg_month_model_emb_correlations.shape, avg_month_vec_emb_correlations.shape)

        total_model_avg_linearity = np.nanmean(avg_month_model_emb_correlations.flatten())
        total_vec_avg_linearity = np.nanmean(avg_month_vec_emb_correlations.flatten())
        print(f"{model_name} {task} model total linearity: {total_model_avg_linearity}")
        print(f"{model_name} {task} vec total linearity: {total_vec_avg_linearity}")

        all_correlation_metrics = {
            "model_correlation_matrices": all_month_model_emb_correlations,
            "vec_correlation_matrices": all_month_vec_emb_correlations,
            "model_correlation_avgs": avg_month_model_emb_correlations,
            "vec_correlation_avgs": avg_month_vec_emb_correlations,
            "model_max_correlations": all_month_model_max_corrs,
            "vec_max_correlations": all_month_vec_max_corrs,
            "model_max_correlation_dists": all_month_model_max_corr_dists,
            "vec_max_correlation_dists": all_month_vec_max_corr_dists,
            "total_model_avg_linearity": total_model_avg_linearity,
            "total_vec_avg_linearity": total_vec_avg_linearity
            #"emb_idx_to_token": emb_idx_to_token
        }
        np.save(f"{model_name}_{task}_yearly_embedding_correlations", all_correlation_metrics)
