# RUN settings
RUN_PRETRAINING = True
RUN_FINETUNING = False
RUN_EVALUATION = True

# Data settings
PROJECT_NAME = "test_set"
DATA = "LOAD_CUSTOM_DATA"# "DATA_PREPARATION" or "LOAD_CUSTOM_DATA"
CUSTOM_DATA_DIRECTORY = "custom_data/historical_test_data"

# Evaluation settings
EVALUATION_TASKS_LIST = ["cloze_task_topk"]

# WandB
USE_WANDB = True

# Steamroller configuration, should be set to 'local' unless on a slurm cluster
STEAMROLLER_ENGINE = 'local' # 'local' or 'slurm'
GPU_COUNT = 1
MEMORY = "64GB"
GPU_ACCOUNT = "your_gpu_account"
CPU_ACCOUNT = "your_cpu_account"
GPU_QUEUE = "your_gpu_queue"
CPU_QUEUE = "your_cpu_queue"
