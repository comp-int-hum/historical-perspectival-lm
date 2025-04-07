import argparse
from wikidata.client import Client
from SPARQLWrapper import SPARQLWrapper, JSON
import json

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--sparql", help="Sparql plaintext input")
    parser.add_argument("--output", help="Output jsonl")
    args, rest = parser.parse_known_args()

    client = Client()

    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    print("Gathering metadata {} from {}".format(args.output, args.sparql))
    with open(args.sparql, "rt") as s_in, open(args.output, "wt") as s_o:
        q = s_in.read()
        sparql.setQuery(q)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        
        wd_c = 0
        for r in results["results"]["bindings"]:
            wd_c+=1
            s_o.write(json.dumps(r)+"\n")
            
    print("Gathered {} authors".format(wd_c))    
    
