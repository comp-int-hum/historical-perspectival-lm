import argparse
import json
import nltk
from tqdm import tqdm

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", help="CA json files")
    parser.add_argument("--output", help="Output jsonl")
    parser.add_argument("--cutoff_start", type=int, help="Start date")
    parser.add_argument("--cutoff_end", type=int, help="End date")
    args, rest = parser.parse_known_args()

    paras = {}

    for jfile in args.input:
        print(f"Processing file: {jfile}")
        with open(jfile, "rt") as j_in:
            qs = json.loads(j_in.read())
            for q in tqdm(qs):
                if not paras.get(q["para_id"]):
                    work_date = int(q["publication_date"][0:4])
                    if work_date >= args.cutoff_start and work_date < args.cutoff_end:
                        entry = {"id": q["para_id"], "structure":[[[]],[nltk.sent_tokenize(q["context"])]], "work_date": work_date}
                        paras[q["para_id"]] = entry

    with open(args.output, "wt") as j_out:
        for entry in paras.values():
            j_out.write(json.dumps(entry)+"\n")
    print(f"Gathered {len(paras)} samples from {args.cutoff_start} through {args.cutoff_end}")

    
