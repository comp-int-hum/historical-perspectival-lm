import os
import os.path

from steamroller import Environment


vars = Variables("custom.py")
vars.AddVariables(

    # Gutenberg data
    ("DATA_ROOT", "", os.path.expanduser("~/corpora")),
    ("GUTENBERG_PATH", "", "${DATA_ROOT}/gutenberg/"),
    ("PG_CATALOG", "", "data/pg_catalog.csv"),

    # SPARQL query
    ("SPARQL_QUERY","", "data/en_authors.txt"),
    
    # Filter settings
    ("WORK_CUTOFF_START", "", -9999), #set to None for no workdate attribution/filtering (leaving it only to authordate)
    ("WORK_CUTOFF_END", "", 1849), #set to None for no workdate attribution/filtering (leaving it only to authordate)
    ("P1_THRESH", "", 90), #similarity threshold for pass 1 of fuzzy matching, paried with bd_thresh
    ("P2_THRESH", "", 101), #similarity threshold for pass 2 of fuzzy matching, used alone
    ("BD_THRESH", "", 5), #allowed birthdate delta
    ("OMIT_AUTHORS","",["Herman Melville"]), #temporary measure to omit a given author, uses WD authorname
    # TODO: max works is 1 to reduce token numbers for preliminary training
    ("MAX_WORKS","", 3), #maximum number of works per author for data balancing purposes
    ("FOLDS", "", 1),
    #work date filter inference settings
    ("USE_INFERENCE", "", False), # use inference to filter works and determined work creation dates
    ("USE_DATES_FILE", "", True),
    ("DATES_FILE","", "data/gb_authors_dates.jsonl"), #used if USE_DATES_FILE is True
    ("WORK_MODEL", "", "meta-llama/Llama-3.3-70B-Instruct"),
    ("WORK_PROMPT", "", "data/work_date_prompt.txt"),

    # SLURM settings
    ("CPU_QUEUE", "", "some_queue"),
    ("CPU_ACCOUNT", "", "some_account"),    
    ("GPU_QUEUE", "", "another_queue"),
    ("GPU_ACCOUNT", "", "another_account"),
    ("GPU_COUNT", "", 1),
    ("WORK_DIR", "", "work_with_data"),

    # Data Split settings
    # TODO: change splits back to 0.7, 0.1, 0.2
    ("TRAIN_PORTION", "", 10000000),
    ("DEV_PORTION", "", 1000000),
    ("TEST_PORTION", "", 5000000),
    ("SPLIT_STYLE", "", "count"), #percent or count
    ("SPLIT_LEVEL", "", "paragraph"), # can be sentence, paragraph, or chapter.
    # Random Seed
    ("RANDOM_SEED", "", 42),

    # Wandb settings
    ("USE_WANDB", "", False),
    ("WANDB_PROJECT", "", "BabyLlama_2"),

    # Training
    ("TRAINER_CONFIG_1", "", "config/llama-smoll-345M.yaml"),
    ("TRAINER_CONFIG_2", "", "config/llama-smoll-345M.yaml"),
    ("STUDENT_CONFIG", "", "config/llama-smoll-345M.yaml")
)

env = Environment(
    variables=vars,
    BUILDERS={
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
            "--cutoff_START ${WORK_CUTOFF_START} "
            "--cutoff_END ${WORK_CUTOFF_END} --max_works ${MAX_WORKS} --filter_birth_death"
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
	    "TokenizeSplit" : Builder(
            action = (
                "python scripts/tokenize_split.py "
                "--input ${SOURCES[0]} "
                "--tokenizer ${SOURCES[1]} "
                "--output ${TARGETS}"
            )
        ),
        "TrainTeacher" : Builder(
            interpreter = "accelerate launch",
            action = (
                "python scripts/train_teacher.py "
                "--train_data ${SOURCES[0]} "
                "--eval_data ${SOURCES[1]} "
                "--tokenizer_path ${SOURCES[2]} "
                "--config ${CONFIG} "
                #"--lr ${LR} "
                "--random_seed ${RANDOM_SEED} "
                "--use_wandb ${USE_WANDB} "
                "--wandb_project ${WANDB_PROJECT} "
                "--wandb_name ${WANDB_NAME} "
                "--output_dir ${TARGET}"
            )
        ),
        "DistillTrainStudent" : Builder(
            action = (
                "python scripts/distill_train_student.py "
                "--train_data ${SOURCES[0]} "
                "--eval_data ${SOURCES[1]} "
                "--tokenizer_path ${SOURCES[2]} "
                "--teacher_dir_1 ${SOURCES[3]} "
                "--teacher_dir_2 ${SOURCES[4]} "
                "--config ${CONFIG} "
                #"--lr ${LR} "
                "--random_seed ${RANDOM_SEED} "
                "--use_wandb ${USE_WANDB} "
                "--wandb_project ${WANDB_PROJECT} "
                "--wandb_name ${WANDB_NAME} "
                "--output_dir ${TARGET}"
            )
        ),
        "Evaluate" : Builder(
            action = (
                "python scripts/evaluate.py "
                "--test_data ${SOURCES[0]} "
                "--tokenizer_path ${SOURCES[1]} "
                "--teacher_dir_1 ${SOURCES[2]} "
                "--teacher_dir_2 ${SOURCES[3]} "
                "--student_dir ${SOURCES[4]} "
                "--report ${TARGET}"
            )
        ),
        "BLIMP_to_CSV" : Builder(
            action = (
                "python scripts/blimp_to_csv.py "
                "--blimp_directories ${SOURCES} "
                "--output_directory ${TARGET}"
            )
        ),
        "BLIMP" : Builder(
            action = (
                "python scripts/start_blimp.py "
                "--model_dir ${SOURCES} "
                "--output ${TARGET}"
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
        "HistoricalMinimalPairs": Builder(
			action = (
				"python scripts/start_historical_mp.py "
				"--model_dir ${SOURCES} "
				"--tasks ${TASKS} "
                "--output ${TARGET}"
			)
		)
        
    }
)

if not env["USE_DATES_FILE"]:
   input = env.File(env["SPARQL_QUERY"])

   query_res = env.QueryWD(source = input, target = "${WORK_DIR}/author_query.jsonl")

   gb_authors = env.GBAuthorFuzzy(source = query_res, target = "${WORK_DIR}/gb_authors.jsonl")


   if env["USE_INFERENCE"]:
      gb_authors_dates = env.AttributeDates(source = gb_authors, target = "${WORK_DIR}/gb_authors_dates.jsonl")
      filtered_authors = env.FilterAndSampleWorks(source = gb_authors_dates, target =  "${WORK_DIR}/gb_authors_dates_filtered.jsonl")
      gb_authors = filtered_authors
else:
    gb_authors = env.File(env["DATES_FILE"])
    gb_authors = env.FilterAndSampleWorks(source = gb_authors, target =  "${WORK_DIR}/gb_authors_dates_filtered.jsonl")


authors_and_extracted_works = env.ExtractAuthorWorksFromPG(
    source = gb_authors, # filtered_authors
    target = "${WORK_DIR}/authors_and_extracted_works.jsonl"
)

extracted_structures = env.ExtractDocStructures(
	source = authors_and_extracted_works,
	target = "${WORK_DIR}/extracted_structures.jsonl"
)

#explicitly specify N tokens reserved for train from each work based on target overall train token number
#eg we want 10000000 train tokens, each work contributes N, the rest can be used for test and dev

train_dev_test = env.TrainingSplit(
	source = extracted_structures,
	target = ["${WORK_DIR}/data.train", "${WORK_DIR}/data.dev", "${WORK_DIR}/data.test"]
)

tokenizer = env.TrainTokenizer(
    source = train_dev_test[0],
    target = "${WORK_DIR}/tokenizer.json"
)

tokenized_train_dev_test = []
for data_split in train_dev_test:
    tokenized_train_dev_test.append(env.TokenizeSplit(
        source = [data_split, tokenizer],
        target = str(data_split) + ".pt"
))

train_data, dev_data, test_data = tokenized_train_dev_test

teacher_1 = env.TrainTeacher(
    source = [train_data, dev_data, tokenizer],
    target = Dir(f"{env['WORK_DIR']}/teacher_1"),
    CONFIG = env["TRAINER_CONFIG_1"],
    WANDB_NAME = f"Teacher_1_{env['TRAINER_CONFIG_1'].split('/')[-1].split('.')[0]}"
)

teacher_2 = env.TrainTeacher(
    source = [train_data, dev_data, tokenizer],
    target = Dir(f"{env['WORK_DIR']}/teacher_2"),
    CONFIG = env["TRAINER_CONFIG_2"],
    WANDB_NAME = f"Teacher_2_{env['TRAINER_CONFIG_2'].split('/')[-1].split('.')[0]}"
)

student = env.DistillTrainStudent(
    source = [train_data, dev_data, tokenizer, teacher_1, teacher_2],
    target = Dir(f"{env['WORK_DIR']}/student"),
    CONFIG = env["STUDENT_CONFIG"],
    WANDB_NAME = f"Student_{env['STUDENT_CONFIG'].split('/')[-1].split('.')[0]}"
)

env.Evaluate(
    source = [dev_data, tokenizer, teacher_1, teacher_2, student],
    target = "${WORK_DIR}/evaluation_report.txt"
)

blimp_student = env.BLIMP(
    source = student,
    target = "${WORK_DIR}/student_eval/blimp/blimp_results.json"
)

blimp_teacher_1 = env.BLIMP(
    source = teacher_1,
    target = "${WORK_DIR}/teacher_1_eval/blimp/blimp_results.json"
)

blimp_teacher_2 = env.BLIMP(
    source = teacher_2,
    target = "${WORK_DIR}/teacher_2_eval/blimp/blimp_results.json"
)

env.BLIMP_to_CSV(
    source = [blimp_student, blimp_teacher_1, blimp_teacher_2],
    target = Dir("${WORK_DIR}/combined_BLIMP")
)


env.EWOK(
    source = student,
    target = "${WORK_DIR}/student_eval/ewok/ewok_results.json"
)

env.EWOK(
    source = teacher_1,
    target = "${WORK_DIR}/teacher_1_eval/ewok/ewok_results.json"
)

env.EWOK(
    source = teacher_2,
    target = "${WORK_DIR}/teacher_2_eval/ewok/ewok_results.json"
)

env.HistoricalMinimalPairs(
	source = student,
	target = "${WORK_DIR}/student_eval/historical_mp/historical_mp_results.json",
	TASKS = ["historical_minimal_pairs"]
)

env.HistoricalMinimalPairs(
    source = teacher_1,
    target = "${WORK_DIR}/teacher_1_eval/historical_mp/historical_mp_results.json",
    TASKS = ["historical_minimal_pairs"]
)

env.HistoricalMinimalPairs(
    source = teacher_2,
    target = "${WORK_DIR}/teacher_2_eval/historical_mp/historical_mp_results.json",
    TASKS = ["historical_minimal_pairs"]
)

"""
env.SuperGLUE(
    source = student,
    target = "${WORK_DIR}/student_eval/super_glue/eval_results.json"
)

env.SuperGLUE(
    source = teacher_1,
    target = "${WORK_DIR}/teacher_1_eval/super_glue/eval_results.json"
)

env.SuperGLUE(
    source = teacher_2,
    target = "${WORK_DIR}/teacher_2_eval/super_glue/eval_results.json"
)
"""
