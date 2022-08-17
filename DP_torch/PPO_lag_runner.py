"""Launch file for the LPPO algorithm with GAE

This script instantiate the Safety Gym environment, the agent, and start the training
"""

import argparse
import os

import gym
import yaml

from ppo_lag.ppo import LPPO
from ppo_lag.traker import Tracker

from packing.cell.cell_gym import CellEnv

with open('config.yml', 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    seed = cfg['setup']['seed']
    ymlfile.close()

if not cfg['setup']['use_gpu']:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

os.environ['PYTHONHASHSEED'] = str(seed)

parser = argparse.ArgumentParser()
parser.add_argument('-env', type=str, help='Gym env', default=cfg['train']['name'])

def main(params):
    config = vars(parser.parse_args())

    env = gym.make(config['env'])
    env.seed(seed)
   
    print(env.unwrapped.spec.id)
    
    agent = LPPO(env, cfg['agent'])

    tracker = Tracker(
        env.unwrapped.spec.id,
        params['tag'],
        seed,
        cfg['agent'], 
        ['Epoch', 'EpReward', 'EpCost', 'Lambda']
    )

    agent.train(
        tracker,
        cfg['agent']
    )

if __name__ == "__main__":
    main(cfg)
   