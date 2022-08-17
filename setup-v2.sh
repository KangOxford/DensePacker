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
pip install tensorflow
pip install tensorboard
pip install tqdm

# export PYTHONPATH=/workspace/DensePacking/DP_torch:$PYTHONPATH
export PYTHONPATH=/content/drive/MyDrive/DensePacker/DP_torch:$PYTHONPATH


# python -m cProfile -s cumulative /workspace/DensePacking/DP_torch/train_cell_gym.py > analysis_train_cell_gym.txt
# python /workspace/DensePacking/DP_torch/train_cell_gym.py > log_train_cell_gym-v10.txt
# python /workspace/DensePacking/DP_torch/train_cell_gym.py

# python /content/drive/MyDrive/DensePacking/DP_torch/test_cell_gym.py > log_test_cell_gym-v18.txt