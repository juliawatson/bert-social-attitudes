# Create visualizations showing the rates of use of mascluine, feminine,
# and gender-neutral variants across name types for BERT's output.
#
# Requires BERT predictions in:
#   bert_predictions_averaged_exclude_modified_adoption_frequency_reweighted.csv
#   bert_predictions_averaged_exclude_modified_compound_frequency_reweighted.csv
# Visualizations are stored in visualizations/


import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as st


c_feminine = "#C86B54"
c_masculine = "#2E3047"
c_neutral = "#7BA38A"
COLORS_COMPOUND = [c_feminine, c_masculine, c_neutral]
COLORS_ADOPTION = [c_feminine, c_neutral]

FONT_SIZE = 15

COMPOUND_RESPONSE_GENDERS = ["feminine", "masculine", "gender neutral"]
ADOPTION_RESPONSE_GENDERS = ["feminine", "gender neutral/masculine"]

METHODS = ["bert_likelihood", "corpus_prior", "frequency_weighted_posterior"]

OUTPUT_DIR = "visualizations"


def load_bert_lexemes():
    compound_data = pd.read_csv("../simple/bert_predictions_averaged_compound.csv")
    adoption_data = pd.read_csv("../simple/bert_predictions_averaged_adoption.csv")
    return set(list(compound_data["lexeme"]) + list(adoption_data["lexeme"]))


def get_error_bars(df, response_genders):
    error_bar_lower = []
    error_bar_upper = []
    for gender in ["woman", "man"]:
        curr_df = df[(df["gender"] == gender)]
        lower_dict = {"gender": gender}
        upper_dict = {"gender": gender}
        for response_gender in response_genders:
            # Source: https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
            lower, upper = st.t.interval(
                confidence=0.95, 
                df=len(curr_df[response_gender])-1, 
                loc=np.mean(curr_df[response_gender]), 
                scale=st.sem(curr_df[response_gender])) 

            lower_dict[response_gender] = lower
            upper_dict[response_gender] = upper
        error_bar_lower.append(lower_dict)
        error_bar_upper.append(upper_dict)
    error_bar_lower_df = pd.DataFrame(error_bar_lower).set_index("gender")
    error_bar_upper_df = pd.DataFrame(error_bar_upper).set_index("gender")

    return error_bar_lower_df, error_bar_upper_df


def make_plot(data, response_genders, curr_ax, title):
    if "gender neutral/masculine" in response_genders:
        data["gender neutral/masculine"] = data["gender neutral"]

    # Compute confidence intervals
    error_bar_lower_df, error_bar_upper_df = get_error_bars(data, response_genders)
    error_bar_lower_df = error_bar_lower_df.loc[["woman", "man"]][response_genders]
    error_bar_upper_df = error_bar_upper_df.loc[["woman", "man"]][response_genders]

    # Take means (group data)
    data = data.groupby("gender").mean()[response_genders]
    data = data.loc[["woman", "man"]][response_genders]

    # Compute error bars (adjust for means)
    errors = [
        [data[c] - error_bar_lower_df[c], error_bar_upper_df[c] - data[c]] 
        for c in data.columns]

    if "masculine" in response_genders:
        colors = [c_feminine, c_masculine, c_neutral]
    else:
        colors = [c_feminine, c_neutral]

    # Make plot
    curr_plot = data.plot(
        kind='bar',
        color=colors,
        yerr=errors,
        ax=curr_ax,
        fontsize=FONT_SIZE,
        width=.8,
    )

    # Add hatches/stripes for variants that are used both as masculine and 
    # gender-neutral, to indicate it's both
    gender_neutral_masculine_index = (
        response_genders.index("gender neutral/masculine")
        if "gender neutral/masculine" in response_genders
        else None)
    if gender_neutral_masculine_index:
        i = 0
        for container in curr_ax.containers:
            if isinstance(container, matplotlib.container.BarContainer):
                if i == gender_neutral_masculine_index:
                    for patch in container.patches:
                        patch.set_hatch("/////")
                        patch.set_edgecolor(c_masculine)
                        patch.set_linewidth(0)
                i += 1

    # Adjust legend + labels
    curr_plot.legend_.remove()
    curr_ax.set_xlabel("gender of name", fontsize=FONT_SIZE)
    curr_ax.set_ylim((0, 1))
    curr_ax.set_title(title)
    for item in curr_ax.get_xticklabels():
        item.set_rotation(0)
    plt.tight_layout()


def visualize_BERT_predictions(
        method="frequency_weighted_posterior", reduced_stimuli_set=False):

    output_label = method
    if reduced_stimuli_set:
        output_label = output_label + "_reduced_stimuli_set"

    for morph_type in ["compound", "adoption"]:
        response_genders = COMPOUND_RESPONSE_GENDERS if morph_type == "compound" else ADOPTION_RESPONSE_GENDERS

        morph_type_data = pd.read_csv(f"/Users/julia/Documents/Fall2022/ai_ethics/bert_gender/papineau/mlm_scoring/bert_predictions_averaged_exclude_modified_{morph_type}_frequency_reweighted.csv")
        if reduced_stimuli_set:
            bert_terms = load_bert_lexemes()
            morph_type_data = morph_type_data[morph_type_data["lexeme"].isin(bert_terms)]
        
        morph_type_data["feminine"] = morph_type_data[f"p_feminine_{method}"]
        morph_type_data["gender neutral"] = morph_type_data[f"p_gender_neutral_{method}"]

        if morph_type == "compound":
            morph_type_data["masculine"] = morph_type_data[f"p_masculine_{method}"]

        _, ax = plt.subplots(figsize=(10/3, 2.1))
        make_plot(morph_type_data, response_genders=response_genders,
                curr_ax=ax,
                title="")
        plt.savefig(f"{OUTPUT_DIR}/{morph_type}_bar_BERT_{output_label}.png", dpi=750)


if __name__ == "__main__":
    for method in METHODS:
        visualize_BERT_predictions(method=method)
        visualize_BERT_predictions(method=method, reduced_stimuli_set=True)