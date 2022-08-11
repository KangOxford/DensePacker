from model.pure_ddpg.ddpg import DDPGAgent
import gym
from gym import wrappers
import os
import numpy as np
import matplotlib.pyplot as plt

N_EPISODES = 10000

def main():
    #get simulation environment
    env = gym.make("Pendulum-v1")
    state_dims = [len(env.observation_space.low)]
    action_dims = [len(env.action_space.low)]
    action_boundaries = [env.action_space.low, env.action_space.high]
    print(action_boundaries)
    #create agent with environment parameters
    agent = DDPGAgent(state_dim = state_dims, action_dim = action_dims,
                action_boundaries = action_boundaries, actor_lr = 5 * 1e-3,
                critic_lr = 2*1e-2, batch_size = 128, gamma = 0.99, rand_steps = 2,
                buffer_size = int(1e6), tau = 0.001)
    np.random.seed(0)
    scores = []
    #training loop: call remember on predicted states and train the models
    episode = 0
    for i in range(N_EPISODES):
        #get initial state
        state = env.reset()
        terminal = False
        score = 0
        #proceed until reaching an exit state
        while not terminal:
            #predict new action
            action = agent.get_action(state, episode)
            #perform the transition according to the predicted action
            state_new, reward, terminal, info = env.step(action)
            #store the transaction in the memory
            agent.remember(state, state_new, action, reward, terminal)
            #adjust the weights according to the new transaction
            agent.learn()
            #iterate to the next state
            state = state_new
            score += reward
            env.render()
        scores.append(score)
        print("Iteration {:d} --> score {:.2f}. Running average {:.2f}".format( i, score, np.mean(scores)))
        episode += 1
    plt.plot(scores)
    plt.xlabel("Episode")
    plt.ylabel("Cumulate reward")
    plt.show()



if __name__ == "__main__":
    #tell tensorflow to train with GPU 0
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
