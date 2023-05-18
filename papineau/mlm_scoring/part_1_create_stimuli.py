# Create stimuli to feed into BERT, for all combinations of names, 
# states, and role nouns.

import csv

from constants import STATES, NAMES, MALE_NAMES
from constants import STIMULI_SETS as stimuli


an_roles = {
    'heir', 'heiress',
    'anchor', 'anchorman', 'anchorwoman',
    'actor', 'actress'}


with open("stimuli.csv", "w") as f:
    csv_writer = csv.DictWriter(
        f, fieldnames=["stimulus", "name", "gender", "a/an", "role", "role_gender", "state", "variants"])
    csv_writer.writeheader()
    for name in NAMES:
        for state in STATES:
            for variants in stimuli:
                if len(variants) == 2:
                    morph_type = "adoption"
                    variant_genders = ("gender_neutral", "feminine")
                else:
                    assert len(variants) == 3
                    morph_type = "compound"
                    variant_genders = ("gender_neutral", "masculine", "feminine")

                for role, role_gender in zip(variants, variant_genders):
                    determiner = "an" if role in an_roles else "a"
                    csv_writer.writerow({
                        "stimulus": f"{name} is {determiner} {role} from {state}",
                        "name": name,
                        "gender": "man" if name in MALE_NAMES else "woman",
                        "a/an": determiner,
                        "role": role,
                        "role_gender": role_gender,
                        "state": state,
                        "variants": variants
                    })
