import os
import os.path
import inspect
import sys

import json
from steamroller import Environment


vars = Variables("custom.py")
vars.AddVariables(
    ("DATA", "", "DATA_PREPARATION"), # can also be "LOAD_CUSTOM_DATA"
    ("CUSTOM_DATA_DIRECTORY", "", None),
    ("RUN_PRETRAINING", "", True),
    ("RUN_FINETUNING", "", False),
    ("RUN_EVALUATION", "", True),
)

env = Environment(
    variables=vars,
    BUILDERS={},
)


training_data = {}
if env["DATA"] == "LOAD_CUSTOM_DATA":
    assert env["CUSTOM_DATA_DIRECTORY"] is not None, "CUSTOM_DATA_DIRECTORY must be set when loading custom data"
    assert os.path.isdir(env["CUSTOM_DATA_DIRECTORY"]), f"{env['CUSTOM_DATA_DIRECTORY']} must be a directory"
    subdirectories = os.listdir(env["CUSTOM_DATA_DIRECTORY"])
    for subdirectory in subdirectories:
        if not os.path.isdir(os.path.join(env["CUSTOM_DATA_DIRECTORY"], subdirectory)):
            continue
        assert os.path.isfile(os.path.join(env["CUSTOM_DATA_DIRECTORY"], subdirectory, "data.train")), f"data.train must be in {subdirectory}"
        assert os.path.isfile(os.path.join(env["CUSTOM_DATA_DIRECTORY"], subdirectory, "data.test")), f"data.test must be in {subdirectory}"
        assert os.path.isfile(os.path.join(env["CUSTOM_DATA_DIRECTORY"], subdirectory, "data.dev")), f"data.dev must be in {subdirectory}"
        training_data[subdirectory] = {
            "train": env.File(os.path.join(env["CUSTOM_DATA_DIRECTORY"], subdirectory, "data.train")),
            "test": env.File(os.path.join(env["CUSTOM_DATA_DIRECTORY"], subdirectory, "data.test")),
            "dev": env.File(os.path.join(env["CUSTOM_DATA_DIRECTORY"], subdirectory, "data.dev")),
        }
elif env["DATA"] == "DATA_PREPARATION":
    SConscript("SConscript_data_preparation")
    Import("training_data")
else:
    raise ValueError("DATA must be set to either LOAD_CUSTOM_DATA or DATA_PREPARATION")


Export("training_data", training_data)


if env["RUN_PRETRAINING"]:
    SConscript("SConscript_pretraining")
    Import("pretrained_results")
else:
    pretrained_results = {}
Export("pretrained_results", pretrained_results)

if env["RUN_FINETUNING"]:
    SConscript("SConscript_finetuning")
    Import("finetuned_results")
else:
    finetuned_results = {}
Export("finetuned_results", finetuned_results)

if env["RUN_EVALUATION"]:
    SConscript("SConscript_evaluation")




