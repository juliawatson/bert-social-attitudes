# Compute correlations between BERT predictions and participant responses,
# for participants from the 3 clusters from Camilliere et al. (2021)
# (non-innovators, innovators, and super-innovators).
# 
# Relies on: 
# * camilliere_data.txt (experimental data)
# * bert_predictions_with_p_they.csv (BERT predictions data)


import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import mannwhitneyu

# Load experimental data; filter out practice items
experimental_data = pd.read_csv("camilliere_data.txt", sep=" ")
experimental_data = experimental_data[experimental_data["exp"] != "practice"]

# Load BERT predictions
bert_feature = "surprisal"  # p_they or surprisal
bert_data = pd.read_csv("bert_predictions_with_p_they.csv")[["cond", "itm", bert_feature]]
bert_data = bert_data.set_index(["cond", "itm"])


def get_n_participants(data_frame):
    return len(set(data_frame["ID"]))


CLUSTER_ID_TO_LABEL = {
    2: "non-innovators",
    1: "innovators",
    3: "super-innovators"
}

# Compute the correlation between BERT probabilities and the average rating per cluster
for include_control in [True, False]:
    print(f"include inanimate control = {include_control}")

    for cluster, cluster_label in CLUSTER_ID_TO_LABEL.items():
        cluster_data = experimental_data[experimental_data["clust"] == cluster]

        if include_control is False:
            cluster_data = cluster_data[cluster_data["cond"] != "inanimate"]

        # Compute the average rating for each item+condition pair
        cluster_averages = cluster_data[["cond", "itm", "rating"]].groupby(["cond", "itm"]).mean()
        cluster_averages = cluster_averages.join(bert_data, on=("cond", "itm"))

        # Ensure there is at least one response per stimulus
        if include_control:
            assert len(cluster_averages) == len(bert_data)

        n_participants = get_n_participants(cluster_data)
        n_observations = len(cluster_averages)

        statistic, p_value = pearsonr(
            cluster_averages["rating"],
            cluster_averages[bert_feature])
        print(f"BERT correlation with {cluster_label}: r={statistic:.4f}, p={p_value} ({n_participants} participants) ({n_observations} observations)")
    print("\n")

# This outputs the following:

# include inanimate control = True
# BERT correlation with non-innovators: r=-0.6241, p=1.454646959832894e-37 (43 participants) (335 observations)
# BERT correlation with innovators: r=-0.5726, p=1.4003969100064773e-30 (89 participants) (335 observations)
# BERT correlation with super-innovators: r=-0.3787, p=7.252897864658624e-13 (16 participants) (335 observations)


# include inanimate control = False
# BERT correlation with non-innovators: r=-0.6399, p=2.9944475494728764e-38 (43 participants) (320 observations)
# BERT correlation with innovators: r=-0.6037, p=3.713511828464452e-33 (89 participants) (320 observations)
# BERT correlation with super-innovators: r=-0.4351, p=3.27092918262478e-16 (16 participants) (320 observations)


# Compute the average survey score per cluster, for each survey:
# nBAcc (non-binary acceptance), gId (gender identity and familiarity),
# tPhob (transphobia), gEss (gender essentialism)
cluster_ideology_scores = {}
ideology_surveys = ["nBAcc", "gId", "tPhob", "gEss"]
for cluster, cluster_label in CLUSTER_ID_TO_LABEL.items():
    cluster_data = experimental_data[experimental_data["clust"] == cluster]
    n_total = len(cluster_data[["ID"] + ideology_surveys].groupby("ID").mean())

    # Remove paticipants that didn't complete the post surveys
    cluster_data = cluster_data.dropna()

    cluster_participants = cluster_data[["ID"] + ideology_surveys].groupby("ID").mean()
    n_responded = len(cluster_participants)
    cluster_ideology_scores[cluster_label] = cluster_participants

    print(f"{cluster_label} (n_participants={n_responded} of {n_total}):")
    for ideology_survey in ideology_surveys:
        curr_score = cluster_participants[ideology_survey].mean()
        print(f"\tavg_{ideology_survey}={curr_score:.4f}")
print("\n\n")

# non-innovators (n_participants=41 of 43):
# 	avg_nBAcc=1.2927
# 	avg_gId=0.6098
# 	avg_tPhob=6.2195
# 	avg_gEss=14.0000
# innovators (n_participants=89 of 89):
# 	avg_nBAcc=1.2697
# 	avg_gId=0.4944
# 	avg_tPhob=5.6966
# 	avg_gEss=14.7753
# super-innovators (n_participants=16 of 16):
# 	avg_nBAcc=2.1250
# 	avg_gId=1.2500
# 	avg_tPhob=2.5000
# 	avg_gEss=14.4375


# Conduct Mann-Whitney U-tests on ideology scores per cluster
# These are the two features that significantly predicted rates of use of they in
# Camilliere et al. (2021) (see Table 2 in their paper.)
for feature in ["nBAcc", "gId"]:
    for test_group in [["non-innovators", "innovators"], ["innovators", "super-innovators"]]:
        result = mannwhitneyu(
            cluster_ideology_scores[test_group[0]][feature],
            cluster_ideology_scores[test_group[1]][feature],
            alternative="less"  # one-tailed test; prediction: nBAcc and gId for non-innovators < innovators < super-innovators
        )
        print(f"mannwhitney-u {feature} for {test_group[0]} vs. {test_group[1]}: statistic={result.statistic}, p_value={result.pvalue:.8f}")
    print("\n")


# mannwhitney-u nBAcc for non-innovators vs. innovators: statistic=1863.0, p_value=0.58051329
# mannwhitney-u nBAcc for innovators vs. super-innovators: statistic=452.0, p_value=0.00826344

# mannwhitney-u gId for non-innovators vs. innovators: statistic=1980.0, p_value=0.81387922
# mannwhitney-u gId for innovators vs. super-innovators: statistic=516.5, p_value=0.02412112