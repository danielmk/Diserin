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



def exploration_performance(results):
    """
    Probability of finding the reward at least one time in the first 20 trials
    """

    num_agents = len(results)
    
    for result in results:
        
        if result['rewarding_trials'][:20].sum()>0:

            success += 1
    
    return 1 - success/num_agents

def learning_performance(results):
    """
    Probability of not finding the reward in the last 5 trials.
    """

    num_agents = len(results)
    success = np.zeros(5)
    for result in results:
        
        success += result['rewarding_trials'][-5:]

    return 1 - (success/num_agents).mean()


def objective(trial):
    
    
    conf['AC']['A_DA'] = trial.suggest_float('A_DA', 0.01, 0.5)
    conf['AC']['A_5HT'] = trial.suggest_float('A_5HT', 0.0001, 0.01)
    conf['AC']['A_ACh'] = trial.suggest_float('A_ACh', 0.001, 0.01)
    conf['AC']['w_max'] = trial.suggest_int('w_max_ac', 10, 100)
    

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

    obj1 = learning_performance(results)
    obj2 = exploration_performance(results)

    return obj2


study = optuna.create_study()
study.optimize(objective, n_trials=100)

print(study.best_params)  # E.g. {'x': 2.002108042}

