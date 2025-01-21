import argparse
import subprocess
import os

# this is a very simple script to start the ewok execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, help="Directory containing the model files")
    parser.add_argument("--output", type=str, help="Output file")
    args = parser.parse_args()

    absolute_path_model = os.path.abspath(args.model_dir)
    absolute_path_output = os.path.abspath(args.output)
    command = ("python -m lm_eval "
                "--model hf --model_args "
                f"pretrained={absolute_path_model},backend='causal' "
                "--tasks ewok_filtered "
                "--device cuda:0 "
                "--batch_size 1 "
                "--log_samples "
                f"--output_path {absolute_path_output}"
    )
    execution_directory = "evaluation-pipeline-2024/"
    os.chdir(execution_directory)
    subprocess.run(command, shell=True)
