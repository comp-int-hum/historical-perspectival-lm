import argparse
import subprocess
import os

# this is a very simple script to start the glue execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, help="Directory containing the model files")
    parser.add_argument("--output", type=str, help="Output file")
    args = parser.parse_args()

    absolute_path_model = os.path.abspath(args.model_dir)
    absolute_path_output = os.path.abspath(args.output)

    execution_directory = "evaluation-pipeline-2024/"
    os.chdir(execution_directory)
    for task in ["boolq", "copa", "wsc", "wic"]:#["boolq", "cola", "mnli", "mrpc", "multirc", "qnli", "qqp", "rte", "sst2", "wsc"]:
        current_output_dir = os.path.join(absolute_path_output, task)
        os.makedirs(current_output_dir, exist_ok=True)
        command = f"python train_lora.py {absolute_path_model} {task} --output_dir {current_output_dir} --batch_size 64 --num_epochs 32 --learning_rate 3e-4 --max_length 128 --do_predict"
        subprocess.run(command, shell=True)
