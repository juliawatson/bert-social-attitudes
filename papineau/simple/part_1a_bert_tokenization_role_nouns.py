# The goal of this is to understand how BERT tokenizes these words.

import csv
from transformers import BertTokenizer

from constants import STIMULI_SETS as stimuli


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
an_roles = {
    'heir', 'heiress',
    'anchor', 'anchorman', 'anchorwoman',
    'actor', 'actress'}


def get_masked_version(tokenized_stimuli_set):
    first_tokenized_item = tokenized_stimuli_set[0]

    # Return None if they're not all the same length
    if not all(len(item) == len(first_tokenized_item) for item in tokenized_stimuli_set):
        return None, None

    # Return None if the first n-1 subwords aren't equal for all items in the stimuli set
    for i in range(len(first_tokenized_item) - 1):
        if not all(item[i] == first_tokenized_item[i] for item in tokenized_stimuli_set):
            return None, None

    masked_text = " ".join(first_tokenized_item[:-1] + ["[MASK]"])
    mask_variants = [item[-1] for item in tokenized_stimuli_set]
    return masked_text, mask_variants


with open("role_noun_tokenization.csv", "w") as f:
    csv_writer = csv.DictWriter(
        f, fieldnames=["stimuli_set","tokenization","masked_version", "a/an",
                       "mask_variants"])
    csv_writer.writeheader()

    for stimuli_set in stimuli:
        tokenization = [
            tokenizer.tokenize(role_noun) for role_noun in stimuli_set]
        masked_version, mask_variants = get_masked_version(tokenization)
        csv_writer.writerow({
            "stimuli_set": stimuli_set,
            "tokenization": tokenization,
            "masked_version": masked_version,
            "a/an": "an" if stimuli_set[0] in an_roles else "a",
            "mask_variants": mask_variants
        })
