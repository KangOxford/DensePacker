# ======================================================= #
# used for dp python env
export PATH=/workspace/anaconda3/bin:$PATH
conda init bash

# reopen the terminal to let it take effect

export PATH=/workspace/DensePacking/DP_torch:$PATH
export PYTHONPATH=/workspace/DensePacking/DP_torch:$PYTHONPATH

# source ~/.bash_profile
# source ~/.bashrc


conda activate dp
ipython

##python version == 3.7.3
# pip install torch==1.11.0
# pip install ipython
# pip install keras
# pip install gym
# pip install pytorch3d
# pip install tk
# pip install scipy
# pip install stable-baselines3
