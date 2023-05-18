# Compute correlations between BERT predictions and participant responses,
# for participants grouped by social attitudes (scores on survey on
# non-binary acceptance).
# 
# Relies on: 
# * camilliere_data_with_ideology_bins.csv (experimental data)
# * bert_predictions_with_p_they.csv (BERT predictions data)


import pandas as pd
from scipy.stats import pearsonr


def get_n_participants(data_frame):
    return len(set(data_frame["ID"]))


# Load experimental data
experimental_data = pd.read_csv("camilliere_data_with_ideology_bins.csv")

# Load BERT predictions
bert_feature = "surprisal"  # p_they or surprisal
bert_data = pd.read_csv("bert_predictions_with_p_they.csv")[["cond", "itm", bert_feature]]
bert_data = bert_data.set_index(["cond", "itm"])


for include_control in [True, False]:
    print(f"include inanimate control = {include_control}")
    bin_type = "nBAcc"
    method = "scale_chunks"

    print(f"\tBERT correlation for scale={bin_type}; bin_approach={method}:")
    for ideology_group in [f'{bin_type}-low', f'{bin_type}-mid', f'{bin_type}-high']:
        group_data = experimental_data[experimental_data[f"{bin_type}_{method}_bin"] == ideology_group]

        if include_control is False:
            group_data = group_data[group_data["cond"] != "inanimate"]

        # Compute the average rating for each item+condition pair
        group_averages = group_data[["cond", "itm", "rating"]].groupby(["cond", "itm"]).mean()
        group_averages = group_averages.join(bert_data, on=("cond", "itm"))

        # Ensure there is at least one response per stimulus
        if include_control:
            assert len(group_averages) == len(bert_data)

        n_participants = get_n_participants(group_data)
        min_rating = min(group_data[bin_type])
        max_rating = max(group_data[bin_type])
        rating_str = f"{min_rating}-{max_rating}"
        n_observations = len(group_averages)

        statistic, p_value = pearsonr(
            group_averages["rating"],
            group_averages[bert_feature])
        print(f"\t\t{ideology_group} ({rating_str}): r={statistic:.4f}, p={p_value:.8f} ({n_participants} participants) ({n_observations} observations)")

    print("\n\n")


# Results for surprisal (not raw p_they)

# include inanimate control = True
# 	BERT correlation for scale=nBAcc; bin_approach=scale_chunks:
# 		nBAcc-low (0.0-0.0): r=-0.5865, p=0.00000000 (42 participants) (335 observations)
# 		nBAcc-mid (1.0-2.0): r=-0.5981, p=0.00000000 (80 participants) (335 observations)
# 		nBAcc-high (3.0-5.0): r=-0.4327, p=0.00000000 (24 participants) (335 observations)

# include inanimate control = False
# 	BERT correlation for scale=nBAcc; bin_approach=scale_chunks:
# 		nBAcc-low (0.0-0.0): r=-0.6073, p=0.00000000 (42 participants) (320 observations)
# 		nBAcc-mid (1.0-2.0): r=-0.6309, p=0.00000000 (80 participants) (320 observations)
# 		nBAcc-high (3.0-5.0): r=-0.4608, p=0.00000000 (24 participants) (320 observations)