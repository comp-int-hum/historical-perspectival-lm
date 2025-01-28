import argparse
import subprocess
import os
import shutil

# this is a very simple script to start the blimp execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, help="Directory containing the model files")
    parser.add_argument("--filtered_blimp_dir", type=str, help="Directory containing the filtered blimp files")
    parser.add_argument("--tasks", type=str, nargs="+", default=["blimp_filtered"], help="Tasks to evaluate")
    parser.add_argument("--output", type=str, help="Output file")
    args = parser.parse_args()

    # replace the content of filtered_blimp_dir to "evaluation-pipeline-2024/eveluation_data/filtered_blimp"

    args.filtered_blimp_dir = os.path.dirname(args.filtered_blimp_dir)
    os.makedirs("evaluation-pipeline-2024/eveluation_data/blimp_filtered", exist_ok=True)
    
    shutil.copytree(args.filtered_blimp_dir, "evaluation-pipeline-2024/evaluation_data/blimp_filtered", dirs_exist_ok=True)

    absolute_path_model = os.path.abspath(args.model_dir)
    absolute_path_output = os.path.abspath(args.output)
    command = ("python -m lm_eval "
                "--model hf --model_args "
                f"pretrained={absolute_path_model},backend='causal' "
                f"--tasks {', '.join(args.tasks)} "
                "--device cuda:0 "
                "--batch_size 512 "
                "--log_samples "
                f"--output_path {absolute_path_output}"
    )
    execution_directory = "evaluation-pipeline-2024/"
    os.chdir(execution_directory)
    subprocess.run(command, shell=True)
