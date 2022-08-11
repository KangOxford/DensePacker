import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
import copy
import numpy as np
import gym
from utils import *
from gym import spaces
from gym.utils import seeding
from collections import deque

import sys
import profile
from utils import *
import datetime
import numpy as np


from packing.scenario import Scenario


from packing.cell.cell_gym import CellEnv
# from particle_gym import MultiParticleEnv

from model.pure_ddpg.ddpg import DDPGAgent
import matplotlib.pyplot as plt

def train_env(env_gym):

    
    scores, episodes = [], []
    EPISODES = 2000
    env_gym.agent.training = True
    # 此处获得颗粒的状态
    pass
    for e in range(EPISODES):
        done = False
        
        np.random.seed(0)
        # 初始化cell，1.方向保留，长度重算， 2.直接回到最原始的立方体晶格
        state = env_gym.reset()
        score = 0
        while not done:
            # fresh env
            env_gym.global_step += 1

            # RL choose action based on observation and go one step
            action = env_gym.get_agent_action(state)
            next_state, reward, done, new_pos, noise, current_pos_onehot, potential_pos_onehot = env_gym.agent_step(
                action)
            # action_vector = np.zeros((env_gym.agent_action_size,))
            # action_vector[action] = 1
            env_gym.memory([noise, current_pos_onehot, potential_pos_onehot], new_pos, reward)
            score += reward
            state = next_state

            if done and ((e + 1) % 10 == 0 or e == EPISODES-1):
                probs = env_gym.train_episodes()
                scores.append(score)
                episodes.append(e)
                print("episode:", e, "  score:", score, "  time_step:", env_gym.global_step)

        if np.max(probs) > 0.75:
            break
        if len(scores) > 3:
            if np.mean(scores[-3:]) <= -150:
                break

        if e % 100 == 0:
            pass
            env_gym.save_model("./models/transition_gradient.h5")

def main():
    
    scenario = Scenario()
    packing = scenario.build_packing()

    # Create environment
    agent_env = CellEnv(packing, scenario.reset_packing, scenario.reward, scenario.observation, scenario.done, method="rotation")
    state_dim = [len(agent_env.observation_space.low)]
    action_dim = [len(agent_env.action_space.low)]
    action_boundaries = [agent_env.action_space.low, agent_env.action_space.high]

    # Create agent and multi-particles
    agent = DDPGAgent(state_dim, action_dim, action_boundaries, actor_lr = 1e-4,
                     critic_lr = 1e-3, batch_size = 128, noise = 'ou', gamma = 0.99, rand_steps = 2,
                     buffer_size = int(1e6), tau = 0.001)

    np.random.seed(0)
    PD = []
    Penalty = []

    #training loop: call remember on predicted states and train the models
    episode = 0
    for i in range(3000):
        #get initial state
        state = agent_env.reset()
        if i == 500:
            filename = f'{packing.fraction:.3f}_zero.scr'
            scr(filename, 'sphere', packing.visable_particles, packing.agent.state.lattice)
        terminal = False
        score = 0
        step = 0
        #proceed until reaching an exit state
        while not terminal:
            #predict new action
            action = agent.get_action(state, episode)
            #perform the transition according to the predicted action
            state_new, reward, terminal, info = agent_env.step(action)
            #store the transaction in the memory
            agent.remember(state, state_new, action, reward, terminal)
            #adjust the weights according to the new transaction
            agent.learn()
            #iterate to the next state
            state = state_new
            score += reward
            step += 1
            #env.render()
        #if packing.cell_penalty == 0. and packing.fraction > 0.5:
            if i == 500:
                filename = f'{packing.fraction:.3f}.scr'
                scr(filename, 'sphere', packing.visable_particles, packing.agent.state.lattice)
        
        if i == 500:
                filename = f'{packing.fraction:.3f}.scr'
                scr(filename, 'sphere', packing.visable_particles, packing.agent.state.lattice)


        PD.append(packing.fraction)
        Penalty.append(packing.cell_penalty)
        print("Iteration {:d} --> step {:d} score {:.2f}. packing density {:.4f}".format( i, step, score, packing.fraction))
        print("penalty {:2f} elite_volume {:2f}".format(packing.cell_penalty, packing.agent.volume_elite))
        episode += 1

    plt.figure(figsize=[8,6])
    plt.plot(PD, 'black',linewidth = 3.0)
    plt.plot(Penalty,'black',ls='--',linewidth = 3.0)
    plt.legend(['PD' , 'Penalty'],fontsize=18)
    plt.xlabel('Episode' ,fontsize=16)
    plt.ylabel('Goal',fontsize=16)
    plt.title('Loss Curves',fontsize=16)
    plt.show()

if __name__ == "__main__":
    main()
