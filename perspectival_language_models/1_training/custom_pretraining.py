import os
from perspectival_language_models.custom import *

# WandB settings
WANDB_PROJECT = "perspectival_language_models_pretraining"


# training configs
TRAINER_CONFIG_1 = "config/llama-smoll-345M.yaml"
TRAINER_CONFIG_2 = "config/llama-smoll-345M.yaml"
STUDENT_CONFIG = "config/llama-smoll-345M.yaml"
