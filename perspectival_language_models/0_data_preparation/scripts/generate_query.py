import re
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--end_time", type=int, help="maximum time of death")
    parser.add_argument("--output", help="output prompt file")
    args, rest = parser.parse_known_args()


    sting = """ SELECT DISTINCT  ?birthDate ?deathDate ?author ?authorLabel   
WHERE
{
     {?author wdt:P106 wd:Q482980 .} UNION {?author wdt:P106 wd:Q36180 .}
     ?author wdt:P6886 wd:Q1860 .
     ?author wdt:P569 ?birthDate FILTER (?birthDate < "1835-01-01T00:00:00Z"^^xsd:dateTime) .
     OPTIONAL { ?author wdt:P570 ?deathDate. }
     SERVICE wikibase:label {                                                                                                                                                          
     	     bd:serviceParam wikibase:language "en".
  }
 } ORDER BY ?birthDate """

    sting = re.sub(r"1835", str(args.end_time), sting)
    with open(args.output, "wt") as s_o:
        s_o.write(sting)
