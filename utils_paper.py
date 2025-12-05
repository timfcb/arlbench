### THC score computation by Obando-Ceron (Consistency of hyperparameter)


from collections import namedtuple
import pickle

import numpy as np
import pandas as pd

from rliable import library as rly
from rliable import metrics

ATARI_100K_GAMES = [
            'Alien', 'Amidar', 'Assault', 'Asterix', 'BankHeist', 'BattleZone',
            'Boxing', 'Breakout', 'ChopperCommand', 'CrazyClimber', 'DemonAttack',
            'Freeway', 'Frostbite', 'Gopher', 'Hero', 'Jamesbond', 'Kangaroo',
            'Krull', 'KungFuMaster', 'MsPacman', 'Pong', 'PrivateEye', 'Qbert',
            'RoadRunner', 'Seaquest', 'UpNDown'
            ]
N_GAMES = len(ATARI_100K_GAMES)

def area_under_the_curve(data):
    return  data[:, :, 0]/2 + data[:, :, 1:-1].sum(axis=2) + data[:, :, -1]/2

Interval = namedtuple("Interval", "hparam lo hi")

def get_int_estimate(data):
    return [Interval(k, np.mean(v) - np.std(v), np.mean(v) + np.std(v)) for k, v in data.items()]

def get_iqm_estimate(data, nreps=5000):
    _, iqm_range = rly.get_interval_estimates(
        data, 
        lambda x: np.array([metrics.aggregate_iqm(x)]),
        reps=nreps
    )
    return [Interval(k.split('_')[-1], iqm_range[k][0], iqm_range[k][1]) for k in data.keys()]
 

def flatten_interval_dict(d):
    return [Interval(k, v[0, 0], v[1, 0]) for k, v in d.items()]

def flatten_iqm_dict(d):
    return [(k, v) for k, v in d.items()]

def interval_to_ranking(scores):
    rank_by_high = sorted(scores, key=lambda x: -x.hi)
    rank_by_high.append(Interval(-1, -100, -100)) # Sentinel
    
    ranks = np.ones((len(scores),))
    for idx, interval in enumerate(rank_by_high):
        begin_equal = 0
        for comp_idx in range(0, idx+1):
            if rank_by_high[comp_idx].lo <= interval.hi:
                begin_equal = comp_idx
                break
        
        for comp_idx in range(idx+1, len(rank_by_high)):
            if rank_by_high[comp_idx].hi < interval.lo: # comp_idx is smaller than interval
                ranks[idx] = (begin_equal + 1 + comp_idx)/2
                break


    return list(zip([mr.hparam for mr in rank_by_high], ranks))

def iqm_to_ranking(scores):
    rank_by_mean = sorted(scores, key=lambda x: -x[1])
    return list(zip([mr[0] for mr in rank_by_mean], range(len(rank_by_mean))))
    
def get_game_rankings(data):
    if len(list(data.values())[0].shape) == 3:
        data = {k: area_under_the_curve(v) for k, v in data.items()}
    transposed_data = {
        game: 
            {
                hp.split('_')[-1]: data[hp][:, idx].reshape(1, -1)
                for hp in data.keys()
            } 
            for idx, game in enumerate(ATARI_100K_GAMES)
        }
    game_intervals = {game: 
                      #flatten_interval_dict(
                        get_int_estimate(transposed_data[game])#) 
                        for game in transposed_data.keys()}
    return {game: interval_to_ranking(intervals)
                for game, intervals in game_intervals.items()}


def span(row):
    return np.ptp(row) + 1

def get_agent_rankings(data):
    if len(list(data.values())[0].shape) == 3:
        data = {k: area_under_the_curve(v) for k, v in data.items()}
    hp_intervals = get_iqm_estimate(data)
    return interval_to_ranking(hp_intervals)


def get_thc_metric(data):
    rankings = get_game_rankings(data)
    rankings = {game:
            { 
                tup[0]: tup[1]
                for tup in rankings[game]
            }
            for game in rankings
           }
    temp = pd.DataFrame(rankings).values
    long_form = temp.reshape(-1)
    hp_column = np.array(list(data.keys())).repeat(len(ATARI_100K_GAMES))
    df = pd.DataFrame([hp_column, long_form], 
                      index=["hyperparameter", "ratings"]).T
    df_deviations = df.groupby(by="hyperparameter").agg(np.ptp)["ratings"]
    df_deviations = (df_deviations/(len(df_deviations)-1))

    return (df_deviations.mean(), df_deviations.std())

def get_agent_metric(drq_eps_data, der_data):
    drq_eps_rankings = get_agent_rankings(drq_eps_data) 
    der_rankings = get_agent_rankings(der_data)
    df = pd.concat((pd.DataFrame(drq_eps_rankings, columns=["hyperparameter", "mean_rank"]),
                    pd.DataFrame(der_rankings, columns=["hyperparameter", "mean_rank"]))) 
    df_deviations = df.groupby(by="hyperparameter").agg(np.ptp)["mean_rank"]
    df_deviations = (df_deviations/(len(df_deviations)-1))
    return (df_deviations.mean(), df_deviations.std())


if __name__ == "__main__":

    data = {
        "agent_lr0.001": np.random.rand(5, 26),
        "agent_lr0.01": np.random.rand(5, 26),
    }

    #with open(f'data/40M_experiments/final_perf/widths.pickle', mode='rb') as f:
    #    data = pickle.load(f)
    #print(data['DrQ_eps_widths'].keys())
    # print(get_agent_metric(data['DrQ_eps_widths'], data['DER_widths']))
    print(get_thc_metric(data))
