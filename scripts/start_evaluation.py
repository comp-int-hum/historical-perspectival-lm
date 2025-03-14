import argparse
import subprocess
import os

# this is a very simple script to start an evaluation
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Directory containing the model files")
    parser.add_argument("--local_model", action = "store_true", help = "Whether the provided model arg is local or not")
    parser.add_argument("--tasks", type=str, nargs="+", default=["historical_minimal_pairs"], help="Tasks to evaluate")
    parser.add_argument("--output", type=str, help="Output file")
    args = parser.parse_args()
    
    # if args.model is None and args.model_name is None:
    #     raise ValueError("Either model_dir or model_name must be provided")
    
    model_path = os.path.abspath(args.model) if args.local_model else args.model
    print(f"model path {model_path}")
    absolute_path_output = os.path.abspath(args.output)
    
    print(args.tasks)
    command = ("python -m lm_eval "
                "--model hf --model_args "
                f"pretrained={model_path},backend='causal',dtype='float' "
                f"--tasks {', '.join(args.tasks)} "
                "--device cuda:0 "
                "--batch_size 512 "
                "--log_samples "
                f"--output_path {absolute_path_output}"
    )
    execution_directory = "evaluation-pipeline-2024/"
    os.chdir(execution_directory)
    subprocess.run(command, shell=True)