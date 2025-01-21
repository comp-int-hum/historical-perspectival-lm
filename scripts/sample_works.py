import re
import argparse
import json
import random

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Jsonl of auth control texts")
    parser.add_argument("--output", help="Output jsonl")
    parser.add_argument("--max_works", type=int, default=1000, help="Maximum numbers of work/author")
    parser.add_argument("--cutoff", type=int, default=1849, help="Date to cutoff works beyond")
    parser.add_argument("--filter_birth_death", default=False, action="store_true", help="Remove works from outside author birth/death window")
    parser.add_argument("--random_state", type=int, default=29)
    args, rest = parser.parse_known_args()

    random.seed(args.random_state)

    n_works = 0
    n_authors = 0
    with open(args.input, "rt") as s_in, open(args.output, "wt") as s_o:
        for line in s_in:
            jline = json.loads(line)
            new_works = {}
            for work, gid in jline["gb_works"].items():
                if args.cutoff:
                    try:
                        if args.filter_birth_death:
                            birth = int(jline["birthDate"]["value"][0:4])
                            death = int(jline["deathDate"]["value"][0:4])
                            if int(jline["work_dates"][gid]) <= args.cutoff and birth < int(jline["work_dates"][gid]) and death >= int(jline["work_dates"][gid]):
                                new_works[work] = gid
                        else:
                            if int(jline["work_dates"][gid]) <= args.cutoff:
                                new_works[work] = gid
                            
                    except ValueError:
                        print("Value Error: {}".format(jline["work_dates"][gid]))
                        continue
                else:
                    new_works[work]=gid
                        
            if len(new_works) > 0:
                n_authors += 1
                n_sample = args.max_works if args.max_works <= len(new_works) else len(new_works)
                sampled = random.sample(sorted(new_works), n_sample)
                n_works += len(sampled)
                jline["gb_works"] = {name: new_works[name] for name in sampled}
                if args.cutoff:
                    jline["work_dates"] = {gid: jline["work_dates"][gid] for gid in jline["gb_works"].values()}
                s_o.write(json.dumps(jline)+"\n")
                
    print("Gathered {} works from {} authors".format(n_works, n_authors))

                        
