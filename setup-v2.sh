# ======================================================= #
# used for non dp python env



pip install torch
pip install keras
pip install gym
pip install pytorch3d
pip install tk
pip install scipy
pip install stable-baselines3
pip install ipython
pip install tensorboard

export PYTHONPATH=/workspace/DensePacking/DP_torch:$PYTHONPATH


# python -m cProfile -s cumulative /workspace/DensePacking/DP_torch/train_cell_gym.py > analysis_train_cell_gym.txt
# python /workspace/DensePacking/DP_torch/train_cell_gym.py > log_train_cell_gym-v10.txt
# python /workspace/DensePacking/DP_torch/train_cell_gym.py
# tensorboard --logdir /workspace/DensePacking/ppo_densepacking_tensorboard-v11/