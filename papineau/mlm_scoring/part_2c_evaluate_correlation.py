# Compute the correlation between the simple method and the mlm_scoring method

import pandas as pd
from scipy.stats import pearsonr
import numpy as np


# Load predictions for the simple method
# Note only 8 lexemes were possible for this approach (4 adoption and 4 compound)
file_simple_method = "../simple/bert_predictions_expanded.csv"
data_simple_method = pd.read_csv(file_simple_method)  # 9600 rows = 24 names x 8 role noun lexemes x 50 states
data_simple_method["morph_type"] = [
    "compound" if is_compound else "adoption" for is_compound in data_simple_method["compound"]]
data_simple_method = pd.melt( 
    data_simple_method, id_vars=['name', "lexeme", "state", "morph_type"],
    value_vars=['p_gender_neutral', 'p_masculine', 'p_feminine'],
    var_name='role_gender', value_name='probability_simple')
data_simple_method["role_gender"] = [item.replace("p_", "") for item in data_simple_method["role_gender"]]

is_undefined = [
    (data_simple_method["morph_type"] == "adoption") &
    (data_simple_method["role_gender"] == "masculine")]
data_simple_method = data_simple_method[np.logical_not(is_undefined)[0]]  # 24000 rows = 24 names x 20 role noun variants x 50 states


# Load predictions for the mlm_scoring method
file_mlm_scoring = "bert_predictions_by_sentence_normalized_exclude_modified_frequency_reweighted.csv"
MLM_SCORING_METHODS = ["bert_likelihood", "corpus_prior", "frequency_weighted_posterior"]

data_mlm_scoring = pd.read_csv(file_mlm_scoring)  # 64800 rows = 24 names x 54 role noun variants x 50 states
data_mlm_scoring["lexeme"] = [eval(item)[0] for item in data_mlm_scoring["variants"]]

method_columns = [f"normalized_probability_{method}" for method in MLM_SCORING_METHODS]
data_mlm_scoring = data_mlm_scoring[['name', "lexeme", "state", "role_gender"] + method_columns]
data_mlm_scoring = data_mlm_scoring.set_index(['name', "lexeme", "state", "role_gender"])


# Join the datasets together, including only stimuli that were possible with both the
# mlm_scoring method and the simple method.
data_simple_method = data_simple_method.join(
    data_mlm_scoring, on=('name', "lexeme", "state", "role_gender"))


# For each of the mlm_scoring methods, print the correlation with the simple method
for method in MLM_SCORING_METHODS:
    statistic, p_value = pearsonr(
        data_simple_method["probability_simple"],
        data_simple_method[f"normalized_probability_{method}"])
    print(f"{method}: r={statistic:.4f}, p={p_value:.8f}")


# This prints the following:
# bert_likelihood: r=0.4195, p=0.00000000
# corpus_prior: r=0.3708, p=0.00000000
# frequency_weighted_posterior: r=0.7600, p=0.00000000