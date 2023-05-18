# Prepare stimuli to feed into BERT.
# This requires camilliere_stimuli.csv, and outputs BERT_stimuli.csv.

import pandas as pd

data = pd.read_csv("camilliere_stimuli.csv")
data["masked_sentence"] = [row["sentence"].replace(row["form"], "[MASK]") for _, row in data.iterrows()]
data.to_csv("BERT_stimuli.csv")