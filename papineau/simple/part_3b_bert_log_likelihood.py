# Compute the log likelihood of the production data based on BERT predictions
# This relies on BERT predictions stored in:
#   bert_predictions_averaged_compound.csv
#   bert_predictions_averaged_adoption.csv 
# This outputs results to:
#   results/compound_loglik.csv
#   results/adoption_loglik.csv


import pandas as pd
import numpy as np
import csv


def get_percentile_bin(gender_ideology_score, percentile33, percentile66):
    if gender_ideology_score < percentile33:
        return "progressive"
    elif gender_ideology_score < percentile66:
        return "moderate"
    return "conservative"


def bin_gender_ideology(data):
    participant_data = data.groupby("workerid").mean()
    percentile33 = np.percentile(participant_data["gender_total"], 100/3)
    percentile66 = np.percentile(participant_data["gender_total"], 2 * 100/3)
    percentile_bins = [
        get_percentile_bin(gender_ideology_score, percentile33, percentile66)
        for gender_ideology_score in data["gender_total"]]
    return percentile_bins


def get_p_response(df):
    result = []
    for _, row in df.iterrows():
        if row["response_gender"] == "male":
            result.append(row["p_masculine"])
        elif row["response_gender"] == "female":
            result.append(row["p_feminine"])
        else:
            assert row["response_gender"] == "neutral"
            result.append(row["p_gender_neutral"])
    return result


def get_loglik(probabilities):
    return np.mean(np.log(probabilities))


def compute_log_likelihoods(morph_type, output_path):
    bert_data = pd.read_csv(f"bert_predictions_averaged_{morph_type}.csv")

    data = pd.read_csv("papineau_production_data_filtered.csv")
    data["gender_ideology"] = bin_gender_ideology(
        data[["workerid", "gender_total"]])
    data = data[data["morph_type"] == morph_type]
    data = data[data["lexeme"].isin(set(list(bert_data["lexeme"])))]
    data = data.drop(columns=["gender"])
    data = data.join(bert_data.set_index(["name", "lexeme"]), on=("name", "lexeme"))

    data["p_response"] = get_p_response(data)

    with open(output_path, "w") as f:
        csv_writer = csv.DictWriter(
            f, fieldnames=["group", "loglik for men's names",
                           "loglik for women's names", "combined loglik"])
        csv_writer.writeheader()
        for poli_party in ["Democrat", "Non-Partisan", "Republican"]:
            curr_data = data[data["poli_party"] == poli_party]
            curr_men_names = curr_data[curr_data["gender"] == "man"]
            curr_women_names = curr_data[curr_data["gender"] == "woman"]
            csv_writer.writerow({
                "group": poli_party,
                "loglik for men's names": get_loglik(curr_men_names["p_response"]),
                "loglik for women's names": get_loglik(curr_women_names["p_response"]),
                "combined loglik": get_loglik(curr_data["p_response"]),
            })

        for gender_ideology in ["progressive", "moderate", "conservative"]:
            curr_data = data[data["gender_ideology"] == gender_ideology]
            curr_men_names = curr_data[curr_data["gender"] == "man"]
            curr_women_names = curr_data[curr_data["gender"] == "woman"]
            csv_writer.writerow({
                "group": gender_ideology,
                "loglik for men's names": get_loglik(curr_men_names["p_response"]),
                "loglik for women's names": get_loglik(curr_women_names["p_response"]),
                "combined loglik": get_loglik(curr_data["p_response"]),
            })


if __name__ == "__main__":
    compute_log_likelihoods(morph_type="compound", output_path="results/compound_loglik.csv")
    compute_log_likelihoods(morph_type="adoption", output_path="results/adoption_loglik.csv")

