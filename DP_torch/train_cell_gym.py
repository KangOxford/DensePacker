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

# v19: new done
env = CellEnv(packing, scenario.reset_packing, scenario.reward, scenario.observation, scenario.done)
# model = PPO.load("/workspace/DensePacking/ppo_densepacking.zip")
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="/mnt/Edisk/andrew/DensePacking-1/tensorboard/ppo_densepacking_tensorboard-v21/")
# model.learn(total_timesteps=int(1e3), tb_log_name="new_penalty_run")
for i in range(int(1e3)):
    model.learn(total_timesteps=int(1e5), tb_log_name="strain_tensor_run", reset_num_timesteps=False)
    model.save("ppo_densepacking-v21","ppo_densepacking-v21")

obs = env.reset()
for i in range(int(1e2)):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

# Todo: penalty design
# reward： * performance的提升 / per itself? which one???
# mountain car gym reward (itself or divide)
# gym.registered
# fixed h
# seals/walker2d-v0 fixed horizon

# tensorboard --logdir=./tensorboard/ppo_densepacking_tensorboard-v21/strain_tensor_run_0 --port 8123
# /mnt/Edisk/andrew/DensePacking-1/tensorboard/ppo_densepacking_tensorboard-v20/period_remove_run_0

