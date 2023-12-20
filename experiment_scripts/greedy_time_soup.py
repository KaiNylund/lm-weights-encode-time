import subprocess
import time
import json
import math
import sys
import os
import argparse

seed = 42
model = "t5-small"
task_vec_path = f"../{model}_vecs/"
summarization = False
lm = False

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=["poli_aff", "news_sum", "wmt_lm"])
    args = parser.parse_args()

    # TODO: update time_vec_paths and eval_file based on how you store the datasets and time vectors for each task
    if args.task == "poli_aff":
        # PoliAff setup
        task = "poli_aff"
        eval_file = "../datasets/poli_tweets/dev/combined_years.jsonl"
        metric = "eval_macro_f1"
        start_year = 2015
        time_vec_paths = [f"{task_vec_path}{task}/2015", f"{task_vec_path}{task}/2016", f"{task_vec_path}{task}/2017", f"{task_vec_path}{task}/2018", f"{task_vec_path}{task}/2019", f"{task_vec_path}{task}/2020"]
        initial_years = [2017, 2018, 2016, 2019, 2015, 2020] # sorted by avg. performance on combined_years
    elif args.task == "news_sum":
        # NewsSum setup (add --summarization flag to command)
        task = "news_sum"
        eval_file = "../datasets/newsroom/summarization/dev/combined_years.jsonl"
        metric = "eval_rougeL"
        start_year = 2012
        time_vec_paths = [f"{task_vec_path}{task}/2012", f"{task_vec_path}{task}/2013", f"{task_vec_path}{task}/2014", f"{task_vec_path}{task}/2015", f"{task_vec_path}{task}/2016", f"{task_vec_path}{task}/2016"]
        initial_years = [2013, 2012, 2014, 2015, 2016] # sorted by avg. performance on combined_years
        summarization = True
    elif args.task == "wmt_lm":
        # WMT LM setup (add --lm flag to command)
        task = "wmt_lm"
        eval_file = "../WMTdata/test_json/combined_years_2012-2016"
        metric = "eval_loss"
        start_year = 2012
        time_vec_paths = [f"{task_vec_path}{task}/2012", f"{task_vec_path}{task}/2013", f"{task_vec_path}{task}/2014", f"{task_vec_path}{task}/2015", f"{task_vec_path}{task}/2016", f"{task_vec_path}{task}/2016"]
        initial_years = [2014, 2015, 2013, 2016, 2012] # sorted by avg. performance on combined_years
        lm = True

    output_dir = f"../soup_outputs/{model}_{task}_greedy_soup_evals/"
    if not os.path.isdir("../soup_outputs/"):
        os.makedirs("../soup_outputs/")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    soup_ingredient_years = []
    years_to_add = list(initial_years)
    year_weights = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    best_weights = []
    if metric == "eval_loss":
        soup_performance = math.inf
    else:
        soup_performance = 0

    # run for k=20 iterations
    for i in range(20):
        cur_year = years_to_add.pop(0)
        cur_ingredients = list(soup_ingredient_years)
        cur_ingredients.append(cur_year)

        for year in initial_years:
            year_weights[year - start_year] = cur_ingredients.count(year) / len(cur_ingredients)

        eval_command = ["bash",
            "test_time_vec_combo.sh",
            model,
            f"{year_weights[0]}",
            f"{year_weights[1]}",
            f"{year_weights[2]}",
            f"{year_weights[3]}",
            f"{year_weights[4]}",
            f"{year_weights[5]}",
            f"{time_vec_paths[0]}",
            f"{time_vec_paths[1]}",
            f"{time_vec_paths[2]}",
            f"{time_vec_paths[3]}",
            f"{time_vec_paths[4]}",
            f"{time_vec_paths[5]}",
            eval_file,
            output_dir,
            lm,
            summarization]

        eval_command = " ".join(eval_command)
        p = subprocess.Popen(eval_command, stdout=subprocess.PIPE, shell=True)
        (output, err) = p.communicate()
        print("Waiting for evaluation to finish...")
        p_status = p.wait()
        #print(output)
        #print(err)

        eval_results_file = f"{output_dir}{model}_combo_{year_weights[0]}_${year_weights[1]}_${year_weights[2]}_${year_weights[3]}_${year_weights[4]}_${year_weights[5]}_eval/eval_results.json"


        # Wait until our eval file is generated and then gather results.
        # Time out and fail after 2 hrs.
        failed = False
        waiting_start = time.time()
        while True:
            try:
                with open(eval_results_file) as eval_json_file:
                    eval_json = json.load(eval_json_file)
                    cur_performance = eval_json[metric]
                    break
            except FileNotFoundError:
                # Timeout after 2 hrs
                if (time.time() - waiting_start) > 7200:
                    failed = True
                    break
                else:
                    time.sleep(10)
        if failed:
            print("Failed to open eval file: " + eval_results_file)
            break
        else:
            print("Eval time elapsed: " + str(round(time.time() - waiting_start, 3)))

        # If performance improved, then keep the updated weights and
        # requeue the same year to be incremented again
        if (metric == "eval_loss" and cur_performance <= soup_performance) or \
        (metric != "eval_loss" and cur_performance >= soup_performance):
            soup_performance = cur_performance
            soup_ingredient_years = cur_ingredients
            best_weights = list(year_weights)

        years_to_add.append(cur_year)
        print("cur performance: " + str(cur_performance) + \
            ", cur weights: " + str(year_weights) + \
            ", cur queue: " + str(years_to_add))

    print("Best greedy weights: " + str(best_weights) + ", best performance: " + str(soup_performance))