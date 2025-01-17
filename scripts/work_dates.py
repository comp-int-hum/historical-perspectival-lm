import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import accelerate
import re
import os.path
import logging
import sys
import argparse
import json

logger = logging.getLogger("annotate")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        help="Authors and gutenberg works jsonl"
    )
    #parser.add_argument("--items", dest="items", help="TSV file of items to ask questions about (must have header row).")
    parser.add_argument("--output", dest="output", help="jsonl output of sampled works files")
    parser.add_argument("--max_tokens", dest="max_tokens", type=int, default=100, help="Maximum tokens to generate in response")
    parser.add_argument(
        "--model",
        dest="model",
        default=os.path.expanduser("~/corpora/models/Llama-3.2-11B-Vision-Instruct"),
        help="Model to use: can be a local path, or Huggingface reference, but should probably resolve to a Llama+vision architecture."
    )
    parser.add_argument("--prompt", help="Prompt template")
    parser.add_argument(
        "--log_level",
        dest="log_level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level: "
    )

    parser.add_argument(
        "--quant_4",
        default=False,
        action="store_true",
        help="If specified, a single value indicating bit quantization (4, 8)"
    )
    
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    with open(args.prompt, "rt") as p_i:
        prompt = p_i.read()

    known_answers = {}

    quantization_config = None
    if args.quant_4:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)   
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.bfloat16, quantization_config=quantization_config)
    processor = AutoTokenizer.from_pretrained(args.model)
    
    with open(args.input, "rt") as jl_i, open(args.output, "wt") as jl_o:
        for line in jl_i:
            jl = json.loads(line)
            work_dates = {}
            for work in jl["gb_works"].keys():
                p_f = prompt.format(work, jl["authorLabel"]["value"])
                prepped = [
                        {
                            "role" : "user",
                            "content" :  [
                                {
                                    "type" : "text",
                                    "text" : p_f
                                }
                            ]
                        }
                    ]
                inp = processor.apply_chat_template(prepped,tokenize=True, add_special_tokens=False, add_generation_prompt=False, return_tensors="pt").to(model.device)
                output = model.generate(inp, max_new_tokens=args.max_tokens)
                ans = re.sub(
                        r"^.*\<\|end_header_id\|\>(.*?)(\<\|eot_id\|\>)?$",
                        r"\1",
                        processor.decode(output[0]).strip(),
                        flags=re.S
                    ).strip()
                work_dates[jl["gb_works"][work]] = ans

            
            jl["work_dates"] = work_dates
            jl_o.write(json.dumps(jl)+"\n")
            
            
