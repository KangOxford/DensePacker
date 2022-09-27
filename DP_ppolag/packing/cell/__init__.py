from packing.cell.cell_gym import CellEnv
from gym.envs import register

register(
    id='Cell-v0',
    entry_point='packing.cell:CellEnv'
    #max_episodes=cell_cfg['num_episodes']
)

