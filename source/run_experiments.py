import pandas as pd
import numpy as np
from experiment.runner import run_experiment
from experiment.config import DOME_BASIC_CONFIG, DRAW_CONFIG, ESC50_BASIC_CONFIG


CONFIG = ESC50_BASIC_CONFIG
RESULT_FILE = f"{CONFIG['dataset']}_results_new"

results = []

for embedding in CONFIG["embeddings"]:

    for rejection_method in CONFIG["rejection_method"]:

        for classifier in CONFIG["classifiers"]:
            

            for num_known in CONFIG["num_known_list"]:

                for seed in CONFIG["seeds"]:
                    print(rejection_method, classifier, num_known)
                    threshold, metrics = run_experiment(
                        embedding,
                        rejection_method,
                        classifier,
                        num_known,
                        seed,
                        CONFIG["test_fold"]
                    )

                    acc_known, acc_unknown, auroc, oscr = metrics

                    results.append({
                        "dataset": CONFIG["dataset"],
                        "embedding": embedding,
                        "rejection_method": rejection_method,
                        "classifier": classifier,
                        "num_known": num_known,
                        "num_unknown": 50 - num_known,
                        "seed": seed,
                        "threshold": threshold,
                        "ACC_known": acc_known,
                        "ACC_unknown": acc_unknown,
                        "AUROC": auroc,
                        "OSCR": oscr
                    })



df = pd.DataFrame(results)

df.to_csv(f"results/{RESULT_FILE}.csv", index=False)

print(f"Results saved to results/{RESULT_FILE}.csv")

