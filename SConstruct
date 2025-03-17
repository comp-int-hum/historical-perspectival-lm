import os
import os.path
import json

from steamroller import Environment
from collections import defaultdict

vars = Variables("custom.py")
vars.AddVariables(
    # Gutenberg data
    ("DATA_ROOT", "", os.path.expanduser("~/corpora")),
    ("GUTENBERG_PATH", "", "${DATA_ROOT}/gutenberg/"),
    ("PG_CATALOG", "", "data/pg_catalog.csv"),
    ("MODEL_PATH", "", "${DATA_ROOT}/models/llama3/Meta-Llama-3-8B-converted"),

    # SPARQL query
    ("SPARQL_QUERY","", "data/en_authors.txt"),
    
    # Filter settings
    ("P1_THRESH", "", 90), #similarity threshold for pass 1 of fuzzy matching, paried with bd_thresh
    ("P2_THRESH", "", 101), #similarity threshold for pass 2 of fuzzy matching, used alone
    ("BD_THRESH", "", 5), #allowed birthdate delta
    ("OMIT_AUTHORS","",[]), #temporary measure to omit a given author, uses WD authorname
    # TODO: max works is 1 to reduce token numbers for preliminary training
    ("MAX_WORKS","", 20), #maximum number of works per author for data balancing purposes
    ("FOLDS", "", 1),
    # Time settings
    ("TIME_SLICES", "", [(1750, 1820), (1820, 1850), (1850, 1880), (1880, 1910), (1910, 1940)]),
    #work date filter inference settings
    ("USE_INFERENCE", "", True), # use inference to filter works and determined work creation dates
    ("USE_DATES_FILE", "", True),
    ("DATES_FILE","", "data/gb_authors_dates_1950.jsonl"), #used if USE_DATES_FILE is True
    ("WORK_MODEL", "", "meta-llama/Llama-3.3-70B-Instruct"),
    ("WORK_PROMPT", "", "data/work_date_prompt.txt"),
    ("HISTORICAL_DATA", "", "data/minimal_pairs_new.csv"),

    # BLIMP settings
    ("MIN_OCCURRENCE_BLIMP", "", 2),
    ("MIN_OCCURRENCE_EVALUATION", "", 2),

    # SLURM settings
    ("CPU_QUEUE", "", "some_queue"),
    ("CPU_ACCOUNT", "", "some_account"),    
    ("GPU_QUEUE", "", "another_queue"),
    ("GPU_ACCOUNT", "", "another_account"),
    ("GPU_COUNT", "", 1),
    ("ORIGINAL_WORK_DIR", "", "work_times"),
    ("WORK_DIR", "", "work_times"),
    ("EVAL_DIR", "", "eval_dir"),

    # Data Split settings
    # TODO: change splits back to 0.7, 0.1, 0.2
    ("TRAIN_PORTION", "", 10**7),
    ("DEV_PORTION", "", 10**6),
    ("TEST_PORTION", "", 5*(10**6)),
    ("SPLIT_STYLE", "", "count"), #percent or count
    ("SPLIT_LEVEL", "", "paragraph"), # can be sentence, paragraph, or chapter.
    # Random Seed
    ("RANDOM_SEED", "", 42),

    # Wandb settings
    ("USE_WANDB", "", True),
    ("WANDB_PROJECT", "", "Dora-llama-3-8B-historical-llm"),

    # Training
    ("DORA_LLAMA_CONFIG", "", "config/dora-llama8B.yaml"),

    # STEAMROLLER settings
    ("STEAMROLLER_ENGINE", "", "slurm"),
    ("CPU_QUEUE", "", "parallel"),
    ("CPU_ACCOUNT", "", "tlippin1"),    
    ("GPU_QUEUE", "", "a100"),
    ("GPU_ACCOUNT", "", "tlippin1_gpu"),
    ("GPU_COUNT", "", 1),
    ("MEMORY", "", "64GB"),
    
    # Evaluation
    ("TASKS", "", ["blimp", "historical_cloze", "historical_mp_updated"]), # , "blimp"]),#  "historical_cloze"]),
    ("TASKS_TO_ACCUMULATE", "", ["historical_cloze"]),
    ("LOCAL_MODEL", "", False),
    ("TIMESLICE_PARAMS", "", 
        {
            "1750_1820": {
                "train_dev_test": [
                    "work_times/1750_1820/data.train",
                    "work_times/1750_1820/data.dev",
                    "work_times/1750_1820/data.test"
                ],
                "models": [
                    # {"model_name": "dora_llama", "model_path": "Hplm/dora_llama_model_1750_1820"},
                    # {"model_name": "baby_llama", "model_path": "Hplm/student_model_1750_1820"},
                    # {"model_name": "baby_llama_rand_init", "model_path": "Hplm/student_model_1750_1820_random_init"}
                    {"model_name": "baby_llama_2_baseline", "model_path": "JLTastet/baby-llama-2-345m"},
                    {"model_name": "llama_8B_baseline", "model_path": "${DATA_ROOT}/models/llama3/Meta-Llama-3-8B-converted"},
                ]
            },
            "1820_1850": {
                "train_dev_test": [
                    "work_times/1820_1850/data.train",
                    "work_times/1820_1850/data.dev",
                    "work_times/1820_1850/data.test"
                ],
                "models": [
                    # {"model_name": "dora_llama", "model_path": "Hplm/dora_llama_model_1820_1850"},
                    # {"model_name": "baby_llama", "model_path": "Hplm/student_model_1820_1850"},
                    # {"model_name": "baby_llama_rand_init", "model_path": "Hplm/student_model_1820_1850_random_init"},
                    {"model_name": "baby_llama_2_baseline", "model_path": "JLTastet/baby-llama-2-345m"},
                    {"model_name": "llama_8B_baseline", "model_path": "${DATA_ROOT}/models/llama3/Meta-Llama-3-8B-converted"},
                ]
            },
            "1850_1880": {
                "train_dev_test": [
                    "work_times/1850_1880/data.train",
                    "work_times/1850_1880/data.dev",
                    "work_times/1850_1880/data.test"
                ],
                "models": [
                    # {"model_name": "dora_llama", "model_path": "Hplm/dora_llama_model_1850_1880"},
                    # {"model_name": "baby_llama", "model_path": "Hplm/student_model_1850_1880"},
                    # {"model_name": "baby_llama_rand_init", "model_path": "Hplm/student_model_1850_1880_random_init"},
                    {"model_name": "baby_llama_2_baseline", "model_path": "JLTastet/baby-llama-2-345m"},
                    {"model_name": "llama_8B_baseline", "model_path": "${DATA_ROOT}/models/llama3/Meta-Llama-3-8B-converted"},
                ]
            },
            "1880_1910": {
                "train_dev_test": [
                    "work_times/1880_1910/data.train",
                    "work_times/1880_1910/data.dev",
                    "work_times/1880_1910/data.test"
                ],
                "models": [
                    # {"model_name": "dora_llama", "model_path": "Hplm/dora_llama_model_1880_1910"},
                    # {"model_name": "baby_llama", "model_path": "Hplm/student_model_1880_1910"},
                    # {"model_name": "baby_llama_rand_init", "model_path": "Hplm/student_model_1880_1910_random_init"},
                    {"model_name": "baby_llama_2_baseline", "model_path": "JLTastet/baby-llama-2-345m"},
                    {"model_name": "llama_8B_baseline", "model_path": "${DATA_ROOT}/models/llama3/Meta-Llama-3-8B-converted"},
                ]
            },
            "1910_1940": {
                "train_dev_test": [
                    "work_times/1910_1940/data.train",
                    "work_times/1910_1940/data.dev",
                    "work_times/1910_1940/data.test"
                ],
                "models": [
                    # {"model_name": "dora_llama", "model_path": "Hplm/dora_llama_model_1910_1940"},
                    # {"model_name": "baby_llama", "model_path": "Hplm/student_model_1910_1940"},
                    # {"model_name": "baby_llama_rand_init", "model_path": "Hplm/student_model_1910_1940_random_init"},
                    {"model_name": "baby_llama_2_baseline", "model_path": "JLTastet/baby-llama-2-345m"},
                    {"model_name": "llama_8B_baseline", "model_path": "${DATA_ROOT}/models/llama3/Meta-Llama-3-8B-converted"},
                ]
            }
            
            
        }
    ),
)

env = Environment(variables=vars)
env.Append(
    BUILDERS={
        "GenerateQuery" : Builder(
              action="python scripts/generate_query.py --output ${TARGET} --end_time ${MAX_CUTOFF}"
        ),
        "QueryWD" : Builder(
              action="python scripts/author_gather_metadata.py --sparql ${SOURCES} --output ${TARGETS}"
        ),
        "GBAuthorFuzzy": Builder(
          action="python scripts/author_gb_fuzzy.py "
                 "--input ${SOURCES} --output ${TARGETS} "
             "--pg_catalog ${PG_CATALOG} "
             "--author_omit ${OMIT_AUTHORS} "
             "--p1_thresh ${P1_THRESH} --p2_thresh ${P2_THRESH} --bd_thresh ${BD_THRESH} --random_state ${RANDOM_SEED}"
        ),
        "AttributeDates": Builder(
            action="python scripts/work_dates.py "
            "--input ${SOURCES} --output ${TARGETS} "
            "--model ${WORK_MODEL} --prompt ${WORK_PROMPT} --quant_4"
        ),
        "FilterAndSampleWorks": Builder(
            action="python scripts/sample_works.py "
            "--input ${SOURCES} --output ${TARGETS} "
            "--cutoff_start ${WORK_CUTOFF_START} "
            "--cutoff_end ${WORK_CUTOFF_END} --max_works ${MAX_WORKS} --filter_birth_death"
        ),

        "ExtractAuthorWorksFromPG" : Builder(
            action = (
                   "python scripts/extract_author_works_from_gutenberg.py "
                "--input ${SOURCES} "
                "--gutenberg_path ${GUTENBERG_PATH} "
                "--output ${TARGETS}"
            )
   
        ),
        "ExtractDocStructures" : Builder(
            action = (
                "python scripts/extract_doc_structures.py "
                "--input ${SOURCES} "
                "--output ${TARGETS}"
            )
        ),
        "TrainingSplit" : Builder(
            action = (
                "python scripts/train_test_val.py "
                "--input ${SOURCES} "
                "--output_train ${TARGETS[0]} "
                "--output_dev ${TARGETS[1]} "
                "--output_test ${TARGETS[2]} "
                "--train_portion ${TRAIN_PORTION} "
                "--dev_portion ${DEV_PORTION} "
                "--test_portion ${TEST_PORTION} "
                "--random_seed ${RANDOM_SEED} "
                "--split_level ${SPLIT_LEVEL} "
                "--split_style ${SPLIT_STYLE}"
            )
        ),
        "TrainTokenizer" : Builder(
            action = (
                "python scripts/train_tokenizer.py "
                "--input ${SOURCES} "
                "--output ${TARGETS}"
            )
        ),
        "GetTokenizer" : Builder(
            action = (
                "python scripts/get_tokenizer.py "
                "--input ${TOKENIZER_NAME} "
                "--output ${TARGETS}"
            )
        ),
        "TokenizeSplit" : Builder(
            action = (
                "python scripts/tokenize_split.py "
                "--input ${SOURCES[0]} "
                "--tokenizer ${SOURCES[1]} "
                "--output ${TARGETS}"
            )
        ),
        "DoraFinetune" : Builder(
            action = (
                "python scripts/dora_finetune.py "
                "--model_name ${MODEL_PATH} "
                "--tokenizer_path ${MODEL_PATH} "
                "--train_data ${SOURCES[0]} "
                "--eval_data ${SOURCES[1]} "
                "--config ${CONFIG} "
                "--random_seed ${RANDOM_SEED} "
                "--use_wandb ${USE_WANDB} "
                "--wandb_project ${WANDB_PROJECT} "
                "--wandb_name ${WANDB_NAME} "
                "--output_dir ${TARGET}"
            )
        ),
        # "Evaluate" : Builder(
        #     action = (
        #         "python scripts/evaluate.py "
        #         "--test_data ${SOURCES[0]} "
        #         "--tokenizer_path ${SOURCES[1]} "
        #         "--teacher_dir_1 ${SOURCES[2]} "
        #         "--teacher_dir_2 ${SOURCES[3]} "
        #         "--student_dir ${SOURCES[4]} "
        #         "--report ${TARGET}"
        #     )
        # ),
        "BLIMP_to_CSV" : Builder(
            action = (
                "python scripts/blimp_to_csv.py "
                "--blimp_directories ${SOURCES} "
                "--blimp_identifiers ${IDENTIFIERS} "
                "--output_directory ${TARGET}"
            )
        ),
        "START_EVALUATION" : Builder(
            action = (
                "python scripts/start_evaluation.py "
                "--model ${SOURCES[0]} "
                f"{'--local_model ' if env['LOCAL_MODEL'] else ''}"  
                "--tasks ${TASK} "
                "--output ${TARGET}"
            ),
            source_factory = env.Dir if env["LOCAL_MODEL"] else env.Value,
            target_factory=env.File
        ),
        "FILTER_EVALUATION" : Builder(
            action = (
                "python scripts/filter_evaluation.py "
                "--data ${SOURCES[1:]} "
                "--evaluation_result ${SOURCES[0]} "
                "--min_occurrence ${MIN_OCCURRENCE_EVALUATION} "
                "--output ${TARGET} "
            )
        ),
        "BLIMP" : Builder(
            action = (
                "python scripts/start_blimp.py "
                "--model_dir ${SOURCES[0]} "
                "--evaluation_result ${SOURCES[1]} "
                "--tasks blimp_filtered "
                "--output ${TARGET}"
            )
        ),
        "FILTER_BLIMP" : Builder(
            action = (
                "python scripts/filter_blimp.py "
                "--data ${SOURCES[1:]} "
                "--blimp_dir ${SOURCES[0]} "
                "--min_occurrence ${MIN_OCCURRENCE_BLIMP} "
                "--output ${TARGET} "
            )
        ),
        "EWOK" : Builder(
            action = (
                "python scripts/start_ewok.py "
                "--model_dir ${SOURCES} "
                "--output ${TARGET}"
            )
        ),
        "SuperGLUE" : Builder(
            action = (
                "python scripts/start_glue.py "
                "--model_dir ${SOURCES} "
                "--output ${TARGET}"
            )
        ),
        "CrossTimePerplexity" : Builder(
            action = (
                "python scripts/cross_time_perplexity.py "
                "--models ${MODELS} "
                "--test_sets ${TEST_SETS} "
                "--tokenizers ${TOKENIZERS} "
                "--time_data ${TIME_DATA} "
                "--output ${TARGET}"
            )
        ),
        "HistoricalMinimalPairs": Builder(
            action = (
                "python scripts/start_historical_mp.py "
                "--model_dir ${SOURCES} "
                "--output ${TARGET}"
            )
        ),
        "HistoricalClozeTask": Builder(
            action = (
                "python scripts/start_historical_cloze.py "
                "--model_dir ${SOURCES} "
                "--output ${TARGET}"
            )
        ),
        "HistoricalTest" : Builder(
            action = (
                "python scripts/evaluate_minimal_pairs.py "
                "--model_dir ${SOURCES[0]} "
                "--minimal_pairs ${HISTORICAL_DATA} "
                "--tokenizer ${SOURCES[1]} "
                "--report ${TARGET}"
            )
        ),
        "HISTORICAL_to_CSV" : Builder(
            action = (
                "python scripts/historical_to_csv.py "
                "--historical_directories ${SOURCES} "
                "--historical_identifiers ${IDENTIFIERS} "
                "--output_directory ${TARGET}"
            )
        ),
        "CombineOutputs" : Builder(
            action = (
                "python scripts/combine_outputs.py "
                "--data ${SOURCES} "
                "--model_names ${MODEL_NAMES} "
                "--output ${TARGET}"
            )
        ),
        "BucketOutputs": Builder(
            action = (
                "python scripts/bucket_outputs.py "
                "--data ${SOURCES} "
                "--time_boundaries ${TIME_BOUNDARIES} "
                "--output_file ${TARGET}"
            )
        ),
        "CalculateBucketStatistics": Builder(
            action = (
                "python scripts/calculate_bucket_statistics.py "
                "--bucketed_data ${SOURCES} "
                "--output_file ${TARGET}"	
            )
        )
        
    }
)

def cpu_task_config(name, time_required, memory_required=env["MEMORY"]):
    return {
        "STEAMROLLER_ACCOUNT": env["CPU_ACCOUNT"],
        "STEAMROLLER_QUEUE": env["CPU_QUEUE"],
        "STEAMROLLER_TIME": time_required,
        "STEAMROLLER_MEMORY": memory_required,
        "STEAMROLLER_NAME_PREFIX": f"{name}",
        "STEAMROLLER_ENGINE": env["STEAMROLLER_ENGINE"],
    }

def gpu_task_config(name, time_required, memory_required=env["MEMORY"]):
    return {
        "STEAMROLLER_ACCOUNT": env["GPU_ACCOUNT"],
        "STEAMROLLER_QUEUE": env["GPU_QUEUE"],
        "STEAMROLLER_TIME": time_required,
        "STEAMROLLER_MEMORY": memory_required,
        "STEAMROLLER_NAME_PREFIX": f"{name}",
        "STEAMROLLER_ENGINE": env["STEAMROLLER_ENGINE"],
        "STEAMROLLER_GPU_COUNT": env["GPU_COUNT"],
    }

# Tasks --> filters --> time-slices --> models
evaluation_results = {}

# Collect all training texts
all_training_texts = [slice_params["train_dev_test"][0] for slice_params in env["TIMESLICE_PARAMS"].values()]

# Iterate over tasks
for task_name in env.get("TASKS", []):
    
    task_by_filters = {"base": defaultdict(list), "filtered": defaultdict(list), "maximal_filtered": defaultdict(list)}
    eval_dir = f"{env['EVAL_DIR']}/{task_name}"
    
    # Iterate over time slices
    for time_slice, slice_params in env["TIMESLICE_PARAMS"].items():
        
        # Iterate available models (dora, baby, baby_rand_init)
        for model_params in slice_params["models"]:
            model_name = model_params["model_name"]
            
            task_result = env.START_EVALUATION(
                source = [model_params["model_path"]],
                target = f"{eval_dir}/base/{time_slice}/{model_name}/{task_name}_results.json",
                TASK = task_name,
                **gpu_task_config(f"{task_name}_{model_name}", "01:30:00", "24GB")
            )
            filtered_result = env.FILTER_EVALUATION(
                source = [task_result, slice_params["train_dev_test"][0]],
                target = f"{eval_dir}/filtered/{time_slice}/{model_name}/{task_name}_results.json",
                MIN_OCCURRENCE = env["MIN_OCCURRENCE_EVALUATION"],
                **cpu_task_config(f"FILTER_{task_name}_{model_name}", "00:30:00", "24GB")
            )
            maximal_filtered_result = env.FILTER_EVALUATION(
                source = [task_result, all_training_texts],
                target = f"{eval_dir}/maximal_filtered/{time_slice}/{model_name}/{task_name}_results.json",
                MIN_OCCURRENCE = env["MIN_OCCURRENCE_EVALUATION"],
                **cpu_task_config(f"MAX_FILTER_{task_name}_{model_name}", "00:30:00", "24GB")
            )
            
            # Update based on filters
            task_by_filters["base"][time_slice].append({"model_name": model_name, "result": task_result})
            task_by_filters["filtered"][time_slice].append({"model_name": model_name, "result": filtered_result})
            task_by_filters["maximal_filtered"][time_slice].append({"model_name": model_name, "result": maximal_filtered_result})
        
    evaluation_results[task_name] = task_by_filters

# Accumulate over models for each task and filters


# Tasks --> filters --> time-slices --> models
for task_name, task_outputs_by_filter in evaluation_results.items():
    if task_name in env["TASKS_TO_ACCUMULATE"]:
        for filter, filtered_outputs in task_outputs_by_filter.items():
            for time_slice, time_slice_outputs in filtered_outputs.items():
                
                model_names = [x["model_name"] for x in time_slice_outputs]
                source_files = [x["result"] for x in time_slice_outputs]
                
                combined_filtered_outputs = env.CombineOutputs(
                    source = source_files,
                    target = f"${{EVAL_DIR}}/accumulated_results/{task_name}/{filter}/{time_slice}/{task_name}_{filter}_results.json",
                    MODEL_NAMES = model_names,
                    **cpu_task_config("CombineOutputs", "2:30:00", "24GB")
                )
                
                bucketed_filtered_outputs = env.BucketOutputs(
                    source = combined_filtered_outputs,
                    target = f"${{EVAL_DIR}}/bucketed_results/{task_name}/{filter}/{time_slice}/{task_name}_{filter}_results.json",
                    TIME_BOUNDARIES = [1750, 1820, 1850, 1880, 1910, 1940],
                    **cpu_task_config("BucketOutputs", "2:30:00", "24GB")
                )
                
                bucketed_statistics = env.CalculateBucketStatistics(
                    source = bucketed_filtered_outputs,
                    target = f"${{EVAL_DIR}}/bucketed_statistics/{task_name}/{filter}/{time_slice}/{task_name}_{filter}_stats.json",
                    **cpu_task_config("CalculateBucketStatistics", "2:30:00", "24GB")
                )
# all_blimp = [x[0] for x in all_evaluation_results["blimp"]]
# all_identifiers = [x[1] for x in all_evaluation_results["blimp"]]

# env.BLIMP_to_CSV(
#         source = all_blimp,
#         target = "${ORIGINAL_WORK_DIR}/combined_BLIMP_filtered/blimp_results.csv",
#         IDENTIFIERS = all_identifiers,
#         **cpu_task_config("BLIMP_to_CSV", "00:30:00", "24GB")
#     )

# all_models = [slice_params["hf_model_name"] for slice_params in env["TIMESLICE_MODELS"]]
# all_test_sets = [slice_params["train_dev_test"][2] for slice_params in env["TIMESLICE_MODELS"]]
# all_time_data = [f'{slice_params["time_slice"][0]}_{slice_params["time_slice"][1]}' for slice_params in env["TIMESLICE_MODELS"]]

# all_models, all_test_sets, all_time_data = zip(
#     *[(slice_params["hf_model_name"], slice_params["train_dev_test"][2], slice)
#       for slice, slice_params in env["TIMESLICE_MODELS"].items()]
# )

# env.CrossTimePerplexity(
#     source =[], # [all_models, all_test_sets, all_tokenizers],
#     target = "${ORIGINAL_WORK_DIR}/cross_time_perplexity/results.json",
#     TIME_DATA = all_time_data,
#     MODELS = all_models,
#     TEST_SETS = all_test_sets,
#     TOKENIZERS = None,
#     **gpu_task_config("CrossTimePerplexity", "03:30:00", "48GB")
# )
"""
historical_data = []
for model, test_set, tokenizer, time_data in all_test_data:
    result = env.HistoricalTest(
        source = [model, tokenizer],
        target = f"{env['ORIGINAL_WORK_DIR']}/historical_test/{time_data}/results.json",
    )
    historical_data.append((result, time_data))

all_historical = [x[0] for x in historical_data]
all_historical_identifiers = [x[1] for x in historical_data]

csv = env.HISTORICAL_to_CSV(
    source = all_historical,
    target = "${ORIGINAL_WORK_DIR}/historical_test/results.csv",
    IDENTIFIERS = all_historical_identifiers,
    **cpu_task_config("HISTORICAL_to_CSV", "00:30:00", "24GB")
)
"""



