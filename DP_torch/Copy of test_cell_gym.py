from stable_baselines3 import PPO
from packing.scenario import Scenario
from packing.cell.cell_gym import CellEnv
from tqdm import tqdm


scenario = Scenario()
packing = scenario.build_packing()
# Create environment

env = CellEnv(packing, scenario.reset_packing, scenario.reward, scenario.observation, scenario.done)
model = PPO.load("/content/drive/MyDrive/DensePacking/EliteDensePacker/ppo_densepacking-v18.zip")

obs = env.reset()
for i in tqdm(range(int(1e5))):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()