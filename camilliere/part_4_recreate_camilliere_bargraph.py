# Plot participant ratings by condition.
#
# This outputs:
#    * visualizations/camilliere_results_by_cluster.png
#    * visualizations/camilliere_results_overall.png
# This relies on camilliere_data.txt.


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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

CLUSTER_ID_TO_LABEL = {
    2: "non-innovators",
    1: "innovators",
    3: "super-innovators"
}


experimental_data = pd.read_csv("camilliere_data.txt", sep=" ")
experimental_data = experimental_data[experimental_data["exp"] != "practice"]
experimental_data["antecedent type"] = [condition_to_label[row["cond"]] for _, row in experimental_data.iterrows()]
experimental_data["cluster"] = [CLUSTER_ID_TO_LABEL[row["clust"]] for _, row in experimental_data.iterrows()]

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


# Plot grouped by cluster
fig, ax = plt.subplots(figsize=(6, 5))
sns.barplot(
    data=experimental_data,
    x="antecedent type", y="rating", hue="cluster",
    order=antecedent_order,
    hue_order=["non-innovators", "innovators", "super-innovators"],
    palette=["#0094FF", "#F6D046", "#F45638"],
    saturation=0.85
)
plt.axhline(4, linestyle="--", c="black")
plt.xticks(rotation=90)
plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", ncol=3)
plt.tight_layout()
plt.savefig("visualizations/camilliere_results_by_cluster.png", dpi=700)


# Plot without grouping by cluster
plt.figure()
sns.barplot(
    data=experimental_data,
    x="antecedent type", y="rating",
    order=antecedent_order,
    color="#00cc66"
)
plt.axhline(4, linestyle="--", c="black")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("visualizations/camilliere_results_overall.png", dpi=700)
