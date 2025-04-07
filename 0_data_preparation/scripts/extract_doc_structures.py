import argparse
import json
import nltk
from bs4 import BeautifulSoup
import logging
import re
from tqdm import tqdm

nltk.download("punkt_tab")
logger = logging.getLogger("extract_text_structures")

def get_chapters(soup):
    # if there are straightforward chapter divs, use those, otherwise collect p-elements between a-elements that have name attributes
    chs = soup.find_all("div", attrs={"class" : "chapter"})
    if chs:
        return chs
    else:
        chapters = []
        cur_p_list = []
        for node in soup.find_all(["a", "p"]):
            if node.name == "a":
                if "name" in node.attrs and len(cur_p_list) > 0:
                    chapters.append(cur_p_list)
                    cur_p_list = []
            elif node.name == "p" and not node.find("p"):
                cur_p_list.append(node)                
        if len(cur_p_list) > 0:
            chapters.append(cur_p_list)
        return chapters

def get_paragraphs(chapter):
    return chapter if isinstance(chapter, list) else [p for p in chapter.find_all("p")]

def get_sentences(paragraph):
    return nltk.sent_tokenize(re.sub(r"\s+", " ", " ".join(paragraph.strings)))

def get_structure(soup):
    return [[get_sentences(par) for par in get_paragraphs(chap)] for chap in get_chapters(soup)]
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help = "input file")
    parser.add_argument("--output", help = "output file")
    args, _ = parser.parse_known_args()
 
    logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")
    
    logger.info("Extracting structure")
 
    with open(args.input, "rt") as input_file, open(args.output, "wt") as output_file:
        for line in tqdm(input_file, desc="Iterating through authors and extracted works"):
            author_info = json.loads(line)
            for work in author_info["extracted_works"]:
                soup = BeautifulSoup(work["raw"], "html.parser")
                doc_json = {k : v for k, v in author_info.items() if k != "extracted_works"}
                doc_json["structure"] = get_structure(soup)
                output_file.write(json.dumps(doc_json) + "\n")

    