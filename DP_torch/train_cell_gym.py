# import gym
# from stable_baselines3.common.vec_env import DummyVecEnv
# env_id = 'Cell-v0'
# env = DummyVecEnv([lambda: gym.make(env_id)])

# import sys
# with open('/workspace/DensePacking/analysis_train_cell_gym.txt', 'w') as sys.stdout:

from stable_baselines3 import PPO
from packing.scenario import Scenario
from packing.cell.cell_gym import CellEnv

scenario = Scenario()
packing = scenario.build_packing()
# Create environment

env = CellEnv(packing, scenario.reset_packing, scenario.reward, scenario.observation, scenario.done)
# model = PPO.load("/workspace/DensePacking/ppo_densepacking.zip")
# model = PPO.load("ppo_densepacking")
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_densepacking_tensorboard-v10/")
# model.learn(total_timesteps=int(1e6), tb_log_name="first_run")
# model.learn(total_timesteps=int(1e3), tb_log_name="second_run",reset_num_timesteps=False)
# model.learn(total_timesteps=int(1e3), tb_log_name="third_run", reset_num_timesteps=False)
# model.learn(total_timesteps=int(1e3), tb_log_name="forth_run", reset_num_timesteps=False)
# model.learn(total_timesteps=int(1e3), tb_log_name="fifth_run", reset_num_timesteps=False)
# model.learn(total_timesteps=int(1e3), tb_log_name="sixth_run", reset_num_timesteps=False)
# model.learn(total_timesteps=int(1e3), tb_log_name="seventh_run", reset_num_timesteps=False)
# model.learn(total_timesteps=int(1e3), tb_log_name="eighth_run", reset_num_timesteps=False)
# model.learn(total_timesteps=int(1e3), tb_log_name="ninth_run", reset_num_timesteps=False)
# model.learn(total_timesteps=int(1e6), tb_log_name="subsequent_run", reset_num_timesteps=False)
model.learn(total_timesteps=int(1e30), tb_log_name="big20_penalty_run")
model.save("ppo_densepacking-v10")

obs = env.reset()
for i in range(int(1e2)):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()