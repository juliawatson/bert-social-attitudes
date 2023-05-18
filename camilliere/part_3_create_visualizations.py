# Create bar plot of surprisal of they, by experimental condition for BERT predictions.
# (The experimental conditions come from the Camilliere et al. (2021) stimuli).
#
# Relies on: bert_predictions.csv
# Outputs:
# * visualizations/antecedent_type_surprisal.png
# * bert_predictions_with_p_they.csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


condition_to_label = {
    "gfamily": 'Close, gendered',     # gender marked, socially close (e.g., my sister)
    "gnoun": 'Distant, gendered',     # gender marked, socially distant (the actress)
    "spkknows": 'Close, non-gend',    # nongender marked, socially close (my cousin)
    "ngnoun": 'Distant non-gend',     # nongender marked, socially distant (the dentist)
    "gname": 'Gendered Name',         # gender biased name (Susan)
    "ngname": 'Non-gend Name',        # gender unbiased name (Taylor)
    "plural": 'Plural NP',            # plural NP (the dentists)
    "quantifier": 'Quantified NP',    # quantified NP (every dentist)
    "inanimate": 'Inanim control',    # inanimate NP (the cup)
}


data = pd.read_csv("bert_predictions.csv")
data["p_they"] = [eval(row["alternative_probabilities"])[row["form"]] for _, row in data.iterrows()]
data["surprisal"] = -np.log(data["p_they"])
data["antecedent type"] = [condition_to_label[row["cond"]] for _, row in data.iterrows()]
data.to_csv("bert_predictions_with_p_they.csv")


antecedent_order = [
    'Inanim control',
    'Gendered Name',
    'Non-gend Name',
    'Close, gendered',
    'Distant, gendered',
    'Close, non-gend',
    'Distant non-gend',
    'Quantified NP',
    'Plural NP']


plt.figure(figsize=(3.8, 4.5))
sns.barplot(
    data=data, x="antecedent type", y="surprisal", order=antecedent_order,
    color="#43b560")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("visualizations/antecedent_type_surprisal.png", dpi=750)


data["adjusted surprisal"] = 8 - data["surprisal"]
plt.figure(figsize=(3.8, 4.5))
sns.barplot(
    data=data, x="antecedent type", y="adjusted surprisal", order=antecedent_order,
    color="#43b560")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("visualizations/adjusted_surprisal.png", dpi=750)
