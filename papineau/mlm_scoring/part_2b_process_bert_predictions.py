# Process BERT predictions, so they are ready to compare to human experimental data.
# * This includes applying the method from Nangia et al. to approximate p(word|context).
# * It also includes averaging across predictions per state (which allows us to compare
#   to the Papineau et al. (2022) production data, where state names are not included.

import csv
import numpy as np
import pandas as pd
from transformers import BertForMaskedLM, BertTokenizer
import tqdm

from constants import STIMULI_SETS

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


def load_frequency_data():
    bookcorpus_data = pd.read_csv("bookcorpus_counts.csv")
    bookcorpus_data["bookcorpus_count"] = bookcorpus_data["count"]
    bookcorpus_data = bookcorpus_data[["term", "bookcorpus_count"]]

    wiki_data = pd.read_csv("wiki_counts.csv")
    wiki_data["wiki_count"] = wiki_data["count"]
    wiki_data = wiki_data[["term", "wiki_count"]]

    assert len(bookcorpus_data) == len(wiki_data)
    data = bookcorpus_data.merge(wiki_data, on="term")
    data["frequency"] = data["bookcorpus_count"] + data["wiki_count"]

    data = data.set_index("term")
    return data


FREQUENCY_DATA = load_frequency_data()


def get_role_noun_to_tokens():
    result = {}
    for stimuli_set in STIMULI_SETS:
        for stimulus in stimuli_set:
            result[stimulus] = tokenizer.tokenize(stimulus)
    return result


ROLE_NOUN_TO_TOKENS = get_role_noun_to_tokens()


def get_sentence_probabilities(
        output_path,      # "bert_predictions_by_sentence_{}.csv",
        input_data_path,  # "bert_predictions.csv",
        exclude_modified=True):
    data = pd.read_csv(input_data_path)
    sentences = set(data["stimulus"])  # 50 states x 24 names x 54 role noun variants = 64800 sentences
    fieldnames = [
        'stimulus', 'name', 'gender', 'a/an', 'role', 'role_gender', 'state',
        'variants', 'raw_log_probability']
    with open(output_path, "w") as out_f:
        csv_writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        csv_writer.writeheader()

        for sentence in tqdm.tqdm(sentences):
            # Focus on this sentence
            curr_sentence_data = data[data["stimulus"] == sentence]

            # Get relevant features for sentence (same across all rows in curr_sentence_data)
            row = {
                k: v for k, v in dict(
                    curr_sentence_data.loc[curr_sentence_data.index[0]]).items()
                    if k in fieldnames}

            # Exclude the modified token
            if exclude_modified:
                curr_sentence_data = curr_sentence_data[
                    [item not in ROLE_NOUN_TO_TOKENS[row["role"]]
                     for item in curr_sentence_data["masked_token"]]]

            # Compute log probability
            row["raw_log_probability"] = np.sum(
                np.log(curr_sentence_data["raw_probability"]))

            # Write to file
            csv_writer.writerow(row)


def normalize_per_lexeme(data, column_to_normalize, label):
    data_summed = data[["name", "state", "variants", column_to_normalize]].groupby([
        "name", "state", "variants"]).sum()
    data_summed[f"normalization_constant_{label}"] = data_summed[column_to_normalize]
    data_summed = data_summed.drop(columns=[column_to_normalize])

    data = data.join(data_summed, on=("name", "state", "variants"))

    data[f"normalized_probability_{label}"] = data[column_to_normalize] / data[f"normalization_constant_{label}"]
    return data


def normalize_sentence_probabilities(
        output_path,       # "bert_predictions_by_sentence_normalized_{}.csv",
        input_data_path,   # "bert_predictions_by_sentence_{}.csv"
    ):
    data = pd.read_csv(input_data_path)
    data["raw_probability"] = np.exp(data["raw_log_probability"])
    data = normalize_per_lexeme(data, "raw_probability", "bert_likelihood")

    data["corpus_frequency"] = [
        FREQUENCY_DATA.loc[role]["frequency"] for role in data["role"]]
    data = normalize_per_lexeme(data, "corpus_frequency", "corpus_prior")

    data["frequency_weighted_posterior"] = data["normalized_probability_bert_likelihood"] * data["normalized_probability_corpus_prior"]
    data = normalize_per_lexeme(data, "frequency_weighted_posterior", "frequency_weighted_posterior")

    data.to_csv(output_path)


def average_sentence_probabilities(
        output_format_str,  # bert_predictions_averaged_exclude_modified_{}_frequency_reweighted.csv
        input_data_path):   # bert_predictions_by_sentence_normalized_exclude_modified_frequency_reweighted.csv
    # re-format data in input_data_path to generate:
    # bert_predictions_averaged_exclude_modified_adoption_frequency_reweighted.csv
    # bert_predictions_averaged_exclude_modified_compound_frequency_reweighted.csv
    data = pd.read_csv(input_data_path)
    fieldnames = ["name", "gender", "lexeme", "p_gender_neutral",
                  "p_feminine", "p_masculine"]

    METHODS = ["bert_likelihood", "corpus_prior", "frequency_weighted_posterior"]
    ROLE_GENDERS = ["gender_neutral", "feminine", "masculine"]

    rows = []
    for _, curr_df in data.groupby(["name","gender","state","variants"]):
        row = {
            k: v for k, v in dict(
                curr_df.loc[curr_df.index[0]]).items()
                if k in fieldnames + ["state", "variants"]}
        variants = eval(row["variants"])
        row["lexeme"] = variants[0]
        row["morph_type"] = "compound" if len(variants) == 3 else "adoption"
        for role_gender in ROLE_GENDERS:
            for method in METHODS:
                if role_gender in set(curr_df["role_gender"]):
                    row[f"p_{role_gender}_{method}"] = float(curr_df[curr_df["role_gender"] == role_gender][f"normalized_probability_{method}"])
                else:
                    row[f"p_{role_gender}_{method}"] = None

        rows.append(row)

    melted_dict = pd.DataFrame(rows)
    for method in METHODS:
        for role_gender in ROLE_GENDERS:
            melted_dict[f'p_{role_gender}_{method}'] = melted_dict[f'p_{role_gender}_{method}'].fillna(0)

    melted_dict_compound = melted_dict[melted_dict["morph_type"] == "compound"]
    averaged_dict_compound = melted_dict_compound.groupby(["name", "gender", "lexeme"]).mean()
    averaged_dict_compound.to_csv(output_format_str.format("compound"))

    melted_dict_adoption = melted_dict[melted_dict["morph_type"] == "adoption"]
    averaged_dict_adoption = melted_dict_adoption.groupby(["name", "gender", "lexeme"]).mean()
    averaged_dict_adoption.to_csv(output_format_str.format("adoption"))


def run_analyses(
        path_stem="bert_predictions.csv",
        label="frequency_reweighted"):
    if label != "" and label[0] != "_":
        label = "_" + label

    bert_predictions_by_sentence_path = path_stem.replace(
        ".csv", f"_by_sentence_exclude_modified.csv")
    get_sentence_probabilities(
        output_path=bert_predictions_by_sentence_path,
        input_data_path=path_stem)

    bert_predictions_normalized_path = path_stem.replace(
        ".csv", f"_by_sentence_normalized_exclude_modified" + label + ".csv")
    normalize_sentence_probabilities(
        output_path=bert_predictions_normalized_path,
        input_data_path=bert_predictions_by_sentence_path)

    bert_format_str = path_stem.replace(
        ".csv", "_averaged_exclude_modified_{}" + label + ".csv")
    average_sentence_probabilities(
        output_format_str=bert_format_str,
        input_data_path=bert_predictions_normalized_path)


if __name__ == "__main__":
    run_analyses()
