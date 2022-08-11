import gym
from gym import wrappers

from model.ppo.ppo import PPO


from packing.cell.cell_gym import CellEnv



if __name__ == "__main__":
    #Pendulum-v1
    env = gym.make("Cell-v0")
    env.seed(0)
#Cell-v0
    # Ensure action bound is symmetric
    #assert (env.action_space.high == -env.action_space.low)

    ppo = PPO(env)

    # ppo.load_model("basic_models/ppo_episode176.h5")
    ppo.train(max_epochs=1000, save_freq=50)
    reward = ppo.test()
    print("Total rewards: ", reward)