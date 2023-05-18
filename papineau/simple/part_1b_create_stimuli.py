# Generate stimlui to feed into BERT
# Requires role_noun_tokenization.csv created in part_1a_bert_tokenization_role_nouns.py
# Outputs stimuli.csv.

import csv
import pandas as pd
import collections

from constants import MALE_NAMES, NAMES, STATES


def load_masked_roles(stimuli_path="role_noun_tokenization.csv"):
    """Loads a dictionary mapping (determiner, masked_role_noun) -> mask variants
    
    For example, it maps
    ("a", "fire [MASK]") -> ['##fighter', '##man', '##woman']
    """
    stimuli_df = pd.read_csv(stimuli_path)

    result = collections.defaultdict(list)
    for _, row in stimuli_df.iterrows():
        if isinstance(row["masked_version"], str):
            result[(row["a/an"], row["masked_version"])].extend(eval(row["mask_variants"]))
    return result


MASKED_ROLES = load_masked_roles()


with open("stimuli.csv", "w") as f:
    csv_writer = csv.DictWriter(
        f, fieldnames=["stimulus", "name", "gender", "a/an", "masked_role", "state", "variants"])
    csv_writer.writeheader()
    for name in NAMES:
        for state in STATES:
            for (determiner, role), variants in MASKED_ROLES.items():
                csv_writer.writerow({
                    "stimulus": f"{name} is {determiner} {role} from {state}",
                    "name": name,
                    "gender": "man" if name in MALE_NAMES else "woman",
                    "a/an": determiner,
                    "masked_role": role,
                    "state": state,
                    "variants": variants
                })
