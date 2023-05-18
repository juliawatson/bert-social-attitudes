# Unpack the stored BERT predictions so they are useful for regressions and visualizations.
# This relies on bert_predictions.csv from the previous step.
# This outputs:
#   bert_predictions_expanded.csv
#   bert_predictions_averaged_compound.csv
#   bert_predictions_averaged_adoption.csv

import csv
import pandas as pd
import collections


def load_stimuli_sets(data_path="role_noun_tokenization.csv"):
    masked_input_to_stimuli_sets = collections.defaultdict(list)
    stimuli_set_to_variants = {}

    with open(data_path, "r") as f_in:
        csv_reader = csv.DictReader(f_in)
        for line in csv_reader:
            if line["masked_version"] != "":
                stimuli_set = eval(line["stimuli_set"])
                mask_variants = eval(line["mask_variants"])
                masked_input_to_stimuli_sets[
                    (line["a/an"], line["masked_version"])].append(stimuli_set)
                stimuli_set_to_variants[stimuli_set] = mask_variants

    return masked_input_to_stimuli_sets, stimuli_set_to_variants


def expand_bert_predictions(
        data_path="bert_predictions.csv",
        output_full="bert_predictions_expanded.csv"):
    """Expands bert predictions in data_path

    Outputs a file where each row is a prediction for an experimental
    stimulus from Papineau et al. (2022).
    """
    masked_input_to_stimuli_sets, stimuli_set_to_variants = load_stimuli_sets()

    with open(data_path, "r") as f_in:
        csv_reader = csv.DictReader(f_in)
        with open(output_full, "w") as f_out:
            csv_writer = csv.DictWriter(
                f_out, fieldnames=[
                    "name", "gender", "lexeme", "state",
                    "raw_p_gender_neutral", "raw_p_masculine", "raw_p_feminine",
                    "p_gender_neutral", "p_masculine", "p_feminine",
                    "compound"])
            csv_writer.writeheader()
            for line in csv_reader:
                # Select relevant stimuli sets
                stimuli_sets = masked_input_to_stimuli_sets[
                    (line["a/an"], line["masked_role"])]
                variant_probabilities = eval(line["variant_probabilities"])

                for stimuli_set in stimuli_sets:
                    if len(stimuli_set) == 2:
                        gender_neutral, feminine = stimuli_set_to_variants[stimuli_set]
                        masculine = None
                    else:
                        assert len(stimuli_set) == 3
                        gender_neutral, masculine, feminine = stimuli_set_to_variants[stimuli_set]

                    # BERT probabilities (unnormalized for this stimuli set)
                    raw_p_feminine = variant_probabilities[feminine]
                    raw_p_masculine = variant_probabilities.get(masculine, None)
                    raw_p_gender_neutral = variant_probabilities[gender_neutral]

                    # probabilities normalized for this stimuli set
                    total_p = sum([variant_probabilities[item] for item in stimuli_set_to_variants[stimuli_set]])
                    p_gender_neutral = raw_p_gender_neutral / total_p
                    p_feminine = raw_p_feminine / total_p
                    if masculine is None:
                        p_masculine = None
                    else:
                        p_masculine = raw_p_masculine / total_p

                    output_row = {
                        "name": line["name"],
                        "gender": line["gender"],
                        "lexeme": stimuli_set[0],
                        "state": line["state"],
                        "raw_p_gender_neutral": raw_p_gender_neutral,
                        "raw_p_feminine": raw_p_feminine,
                        "raw_p_masculine": raw_p_masculine,
                        "p_gender_neutral": p_gender_neutral,
                        "p_feminine": p_feminine,
                        "p_masculine": p_masculine,
                        "compound": len(stimuli_set) == 3
                    }
                    csv_writer.writerow(output_row)


def average_bert_predictions(
        expanded_bert_path="bert_predictions_expanded.csv",
        averaged_bert_path="bert_predictions_averaged_{}.csv"):
    data_df = pd.read_csv(expanded_bert_path)

    # Each of these dfs has 50 states x 4 role nouns x 24 names = 4800 rows
    compound_df = data_df[data_df["compound"] == True][
        ["name", "gender", "lexeme", "state", "p_gender_neutral", "p_feminine", "p_masculine"]]
    adoption_df = data_df[data_df["compound"] == False][
        ["name", "gender", "lexeme", "state", "p_gender_neutral", "p_feminine"]]

    # Each of these dfs has 4 role nouns x 24 names = 96 rows    
    compound_df = compound_df.groupby(["name", "gender", "lexeme"]).mean()
    adoption_df = adoption_df.groupby(["name", "gender", "lexeme"]).mean()

    compound_df.to_csv(averaged_bert_path.format("compound"))
    adoption_df.to_csv(averaged_bert_path.format("adoption"))


if __name__ == "__main__":
    expand_bert_predictions()
    average_bert_predictions()
