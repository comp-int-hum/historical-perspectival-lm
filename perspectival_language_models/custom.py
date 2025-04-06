# RUN settings
PROJECT_NAME = "song_lyrics"
DATA = "LOAD_CUSTOM_DATA"# "DATA_PREPARATION" or "LOAD_CUSTOM_DATA"
CUSTOM_DATA_DIRECTORY = "custom_data/song_lyrics"
RUN_PRETRAINING = True
RUN_FINETUNING = False
RUN_EVALUATION = False

# WandB
USE_WANDB = True

# Steamroller configuration, should be set to 'local' unless on a slurm cluster
STEAMROLLER_ENGINE = 'slurm' #"slurm" #"local" # "slurm"
GPU_COUNT = 1
MEMORY = "64GB"
GPU_ACCOUNT = "tlippin1_gpu"
CPU_ACCOUNT = "tlippin1"
GPU_QUEUE = "a100"
CPU_QUEUE = "parallel"
