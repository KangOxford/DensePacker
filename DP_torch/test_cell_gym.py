import pandas as pd
from tqdm import tqdm
from stable_baselines3 import PPO
from packing.scenario import Scenario
from packing.cell.cell_gym import CellEnv

scenario = Scenario()
packing = scenario.build_packing()
# Create environment

env = CellEnv(packing, scenario.reset_packing, scenario.reward, scenario.observation, scenario.done)
model = PPO.load("/mnt/Edisk/andrew/DensePacking-1/DP_torch/a.zip")
# version 5 is the 17:52 17th Aug, last version.

info_list = []
obs = env.reset()
for i in tqdm(range(int(1e5))):
# for i in tqdm(range(int(1e5))):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    info_list.append(info)
    if done:
        env.reset()
 
pd.DataFrame(info_list).to_csv("/mnt/Edisk/andrew/DensePacking-1/outcomes/analysis_test_cell_gym-v18.6.csv")