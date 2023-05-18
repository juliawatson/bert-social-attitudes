# Compute the log likelihood of the production data based on BERT predictions
# This relies on BERT predictions stored in:
#   ../../bert_predictions_averaged_exclude_modified_adoption_frequency_reweighted.csv
#   ../../bert_predictions_averaged_exclude_modified_compound_frequency_reweighted.csv
# It also relies on the human production data from Papineau et al. (2022) storted at:
#   ../../../simple/papineau_production_data_filtered.csv
# This outputs results to the results/ dir

import pandas as pd
import numpy as np
import csv


METHODS = ["bert_likelihood", "corpus_prior", "frequency_weighted_posterior"]


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


def get_p_response(df, method="frequency_weighted_posterior"):
    result = []
    for _, row in df.iterrows():
        if row["response_gender"] == "male":
            result.append(row[f"p_masculine_{method}"])
        elif row["response_gender"] == "female":
            result.append(row[f"p_feminine_{method}"])
        else:
            assert row["response_gender"] == "neutral"
            result.append(row[f"p_gender_neutral_{method}"])
    return result


def get_loglik(probabilities):
    return np.mean(np.log(probabilities))


def compute_log_likelihoods(morph_type, output_format_str):
    bert_data = pd.read_csv(f"../../bert_predictions_averaged_exclude_modified_{morph_type}_frequency_reweighted.csv")

    data = pd.read_csv("../../../simple/papineau_production_data_filtered.csv")
    data["gender_ideology"] = bin_gender_ideology(
        data[["workerid", "gender_total"]])
    data = data[data["morph_type"] == morph_type]

    ### EXCLUDE heir and host ###
    data = data[~data["lexeme"].isin({"heir", "host"})]
    ### EXCLUDE heir and host ###

    data = data.drop(columns=["gender"])
    data = data.join(bert_data.set_index(["name", "lexeme"]), on=("name", "lexeme"))

    for method in METHODS:
        data[f"p_response_{method}"] = get_p_response(data, method=method)
        output_path = output_format_str.format(method)

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
                    "loglik for men's names": get_loglik(curr_men_names[f"p_response_{method}"]),
                    "loglik for women's names": get_loglik(curr_women_names[f"p_response_{method}"]),
                    "combined loglik": get_loglik(curr_data[f"p_response_{method}"]),
                })

            for gender_ideology in ["progressive", "moderate", "conservative"]:
                curr_data = data[data["gender_ideology"] == gender_ideology]
                curr_men_names = curr_data[curr_data["gender"] == "man"]
                curr_women_names = curr_data[curr_data["gender"] == "woman"]
                csv_writer.writerow({
                    "group": gender_ideology,
                    "loglik for men's names": get_loglik(curr_men_names[f"p_response_{method}"]),
                    "loglik for women's names": get_loglik(curr_women_names[f"p_response_{method}"]),
                    "combined loglik": get_loglik(curr_data[f"p_response_{method}"]),
                })
    
    # Add file with improvement over baseline for frequency prior
    df_prior = pd.read_csv(output_format_str.format("corpus_prior"), index_col="group")
    df_posterior = pd.read_csv(output_format_str.format("frequency_weighted_posterior"), index_col="group")
    df_improvement = df_posterior - df_prior
    df_improvement_output_path = output_format_str.format("posterior_improvement_over_prior")
    df_improvement.to_csv(df_improvement_output_path)


if __name__ == "__main__":
    compute_log_likelihoods(morph_type="compound", output_format_str="results/compound_loglik_{}.csv")
    compute_log_likelihoods(morph_type="adoption", output_format_str="results/adoption_loglik_{}.csv")

