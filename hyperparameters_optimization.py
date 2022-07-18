import multiprocessing, psutil, os, time, sys, yaml
import numpy as np
import optuna

from main import episode_run

parameter_file = sys.argv[-1]

with open(parameter_file, 'r') as stream:
    try:
        conf = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


def objective_performance(trial):

    """
    Objective to maximize the probability of finding rewards in the last 5 trials.
    """
    
    conf['AC']['A_DA'] = trial.suggest_float('A_DA', 0.001, 0.1)
    conf['AC']['A_5HT'] = trial.suggest_float('A_5HT', 0.001, 0.1)
    conf['AC']['A_ACh'] = trial.suggest_float('A_ACh', 0.001, 0.1)
    conf['AC']['w_max'] = trial.suggest_int('w_max_ac', 1, 10)
    

    # Run all episodes
    pool = multiprocessing.Pool(os.cpu_count() - 1)

    results = []
    for episode in range(0, conf['num_agents']):

        results.append(pool.apply_async(episode_run,(episode, conf)))
        
        current_process = psutil.Process()
        children = current_process.children(recursive=True)

        while len(children) > os.cpu_count() - 1:
            time.sleep(0.1)
            current_process = psutil.Process()
            children = current_process.children(recursive=True)

    results = [result.get() for result in results]
    pool.close()
    pool.join()

    # Compute average number of successive agents at the last trial

    counter = 0
    success = np.zeros(5)
    for result in results:
        
        success += result['rewarding_trials'][-5:]
        counter += 1

    return 1 - (success/counter).mean()


study = optuna.create_study()
study.optimize(objective_performance, n_trials=100)

print(study.best_params)  # E.g. {'x': 2.002108042}

