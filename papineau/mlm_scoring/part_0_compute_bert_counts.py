# This script computes the frequencies of role nouns in the corpora BERT was trained on.
# This is used as a prior to re-weight the probabilities from the Nangia et al. method.
# adapted from: https://www.philschmid.de/pre-training-bert-habana#1-prepare-the-dataset


import collections
import csv
from datasets import load_dataset
import re
import tqdm

from constants import STIMULI_SETS


ROLE_NOUNS = []
for stimuli_set in STIMULI_SETS:
    ROLE_NOUNS.extend(list(stimuli_set))

QUERY = "|".join([r"\b" + noun + r"\b" for noun in ROLE_NOUNS])


def save_counter(output_path, counter):
    with open(output_path, "w") as f:
        csv_writer = csv.DictWriter(f, fieldnames=["term", "count"])
        csv_writer.writeheader()
        for key in sorted(list(counter.keys())):
            csv_writer.writerow({
                "term": key,
                "count": counter[key]
            })


def compute_counts_for_dataset(output_path, dataset, query=QUERY):
    counter = collections.Counter()
    for item in tqdm.tqdm(dataset):
        occurrences = re.findall(query, item['text'], flags=re.IGNORECASE)
        counter.update([item.lower() for item in occurrences])
    save_counter(output_path, counter)


if __name__ == "__main__":
    wiki = load_dataset("wikipedia", "20200501.en", split="train")   # Takes 1-2 hrs to download (tried 2 times)
    wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])  # only keep the 'text' column
    compute_counts_for_dataset("wiki_counts.csv", wiki)              # takes ~2 hrs to run -- ran once
    del wiki
    
    bookcorpus = load_dataset("bookcorpus", split="train")           # Takes ~3 mins to download; 20 mins to load (tried 4 times)
    compute_counts_for_dataset("bookcorpus_counts.csv", bookcorpus)  # Takes 1-1.5 hrs to run -- ran twice
    del bookcorpus

