import os
from custom import *

# Local paths (TO CHANGE)
DATA_ROOT = os.path.expanduser("~/corpora") 
GUTENBERG_PATH = os.path.join(DATA_ROOT, "gutenberg/")

WORK_DIR = f"0_data_preparation/work/{PROJECT_NAME}"
ORIGINAL_WORK_DIR = WORK_DIR

# Time slice ranges
TIME_SLICES = [(1750, 1820), (1820, 1850), (1850, 1880), (1880, 1910), (1910, 1940)]

# Author and work filtering
P1_THRESH = 90
P2_THRESH = 101
BD_THRESH = 5
OMIT_AUTHORS = []
MAX_WORKS = 20

# Model and prompt settings
USE_INFERENCE = True
WORK_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
USE_DATES_FILE = True
DATES_FILE ="0_data_preparation/data/gb_authors_dates_1950.jsonl"



# Data split settings
TRAIN_PORTION = 10**7
DEV_PORTION = 10**6
TEST_PORTION = 5*(10**6)
SPLIT_STYLE = "count" #percent or count
SPLIT_LEVEL = "paragraph" # can be sentence, paragraph, or chapter.