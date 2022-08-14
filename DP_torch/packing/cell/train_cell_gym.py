# import gym
# from stable_baselines3.common.vec_env import DummyVecEnv
# env_id = 'Cell-v0'
# env = DummyVecEnv([lambda: gym.make(env_id)])


from stable_baselines3 import PPO

from packing.scenario import Scenario
from packing.cell.cell_gym import CellEnv


scenario = Scenario()
    packing = scenario.build_packing()
    # Create environment
    env = CellEnv(packing, scenario.reset_packing, scenario.reward, scenario.observation, scenario.done)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=25000)
    model.save("ppo_densepacking")

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()