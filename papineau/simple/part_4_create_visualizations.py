# Create visualizations showing the rates of use of mascluine, feminine,
# and gender-neutral variants across name types for BERT's output,
# as well as for human participants.
#
# Requires BERT predictions in:
#   bert_predictions_averaged_compound.csv
#   bert_predictions_averaged_adoption.csv
# Requres participant experimental data in:
#   papineau_production_data_filtered.csv
# Visualizations are stored in visualizations/

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
from scipy.stats import binomtest

from part_3b_bert_log_likelihood import bin_gender_ideology


c_feminine = "#C86B54"
c_masculine = "#2E3047"
c_neutral = "#7BA38A"
COMPOUND_COLORS = [c_feminine, c_masculine, c_neutral]
ADOPTION_COLORS = [c_feminine, c_neutral]

FONT_SIZE = 15

COMPOUND_RESPONSE_GENDERS = ["feminine", "masculine", "gender neutral"]
ADOPTION_RESPONSE_GENDERS = ["feminine", "gender neutral/masculine"]


def visualize_BERT_predictions():
    compound_data = pd.read_csv("bert_predictions_averaged_compound.csv")
    compound_data["feminine"] = compound_data[f"p_feminine"]
    compound_data["masculine"] = compound_data[f"p_masculine"]
    compound_data["gender neutral"] = compound_data[f"p_gender_neutral"]

    compound_data = pd.melt(
        compound_data, 
        id_vars=['name', 'gender', 'lexeme'], 
        value_vars=["masculine", "feminine", "gender neutral"],
        var_name="response gender", 
        value_name="p(response)")
    
    plt.figure()
    sns.barplot(
        data=compound_data,
        x="gender", y="p(response)", hue="response gender",
        hue_order=["feminine", "masculine", "gender neutral"],
        order=["woman", "man"],
        palette=COMPOUND_COLORS)
    plt.ylim((0, 1))
    plt.savefig("visualizations/compound_bar_BERT.png", dpi=750)

    adoption_data = pd.read_csv("bert_predictions_averaged_adoption.csv")
    adoption_data["feminine"] = adoption_data[f"p_feminine"]
    adoption_data["gender neutral"] = adoption_data[f"p_gender_neutral"]

    adoption_data = pd.melt(
        adoption_data, 
        id_vars=['name', 'gender', 'lexeme'], 
        value_vars=["feminine", "gender neutral"],
        var_name="response gender", 
        value_name="p(response)")

    plt.figure()
    sns.barplot(
        data=adoption_data,
        x="gender", y="p(response)", hue="response gender",
        hue_order=["feminine", "gender neutral"],
        order=["woman", "man"],
        palette=ADOPTION_COLORS)
    plt.ylim((0, 1))
    plt.savefig("visualizations/adoption_bar_BERT.png", dpi=750)


def load_bert_lexemes():
    compound_data = pd.read_csv("bert_predictions_averaged_compound.csv")
    adoption_data = pd.read_csv("bert_predictions_averaged_adoption.csv")

    return set(list(compound_data["lexeme"]) + list(adoption_data["lexeme"]))


def get_error_bars(df, response_genders):
    # For the papineau data, we plot error bars for each response type (e.g., 'feminine')
    # as the 95% confidence interval for the proportion of responses of that type, for a binomial test.
    # (e.g., the proportion of feminine responses, relative to non-feminine responses, including
    # both masculine and gender-neutral responses)
    # https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html

    error_bar_lower = []
    error_bar_upper = []
    for gender in ["woman", "man"]:
        curr_df = df[(df["gender"] == gender)]
        lower_dict = {"gender": gender}
        upper_dict = {"gender": gender}
        for response_gender in response_genders:
            n_total = len(curr_df)
            n_response_gender = sum(curr_df[response_gender])

            result = binomtest(k=n_response_gender, n=n_total, p=0.1)  # p here doesn't matter, we only care about the interval
            confidence_interval = result.proportion_ci()

            lower_dict[response_gender] = confidence_interval.low
            upper_dict[response_gender] = confidence_interval.high
        error_bar_lower.append(lower_dict)
        error_bar_upper.append(upper_dict)
    error_bar_lower_df = pd.DataFrame(error_bar_lower).set_index("gender")
    error_bar_upper_df = pd.DataFrame(error_bar_upper).set_index("gender")

    return error_bar_lower_df, error_bar_upper_df


def make_plot_production_data(data, response_genders, curr_ax, title):
    data = data.copy()
    data["masculine"] = (data["response_gender"] == "male").astype(int)
    data["gender neutral"] = (data["response_gender"] == "neutral").astype(int)
    data["feminine"] = (data["response_gender"] == "female").astype(int)
    data["gender neutral/masculine"] = (data["response_gender"] == "neutral").astype(int)

    data["gender"] = ["man" if item == "male" else "woman" for item in data["gender"]]

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
        width=.8,
        fontsize=FONT_SIZE
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
    curr_ax.set_title(title, fontsize=FONT_SIZE)
    for item in curr_ax.get_xticklabels():
        item.set_rotation(0)


def visualize_production_data_ideology(bert_terms_only=False, title="all"):
    data = pd.read_csv("papineau_production_data_filtered.csv")
    data["gender_ideology"] = bin_gender_ideology(
        data[["workerid", "gender_total"]])

    if bert_terms_only:
        bert_terms = load_bert_lexemes()
        data = data[data["lexeme"].isin(bert_terms)]

    for morph_type in ["compound", "adoption"]:
        response_genders = COMPOUND_RESPONSE_GENDERS if morph_type == "compound" else ADOPTION_RESPONSE_GENDERS

        fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, figsize=(10, 3))
        for i, gender_ideology in enumerate(["progressive", "moderate", "conservative"]):
            curr_ax = ax[i]
            morph_type_data = data[(data["morph_type"] == morph_type) &
                                (data["gender_ideology"] == gender_ideology)]
            make_plot_production_data(
                morph_type_data, response_genders, curr_ax, gender_ideology)

        handles, labels = ax[0].get_legend_handles_labels()
        legend = fig.legend(
            handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.0),
            title="gender of role noun", fontsize=FONT_SIZE)
        legend.get_title().set_fontsize(FONT_SIZE) 
        plt.tight_layout()
        plt.subplots_adjust(top=0.62)
        plt.savefig(
            f"visualizations/{morph_type}_bar_production_gender_ideology_{title}.png", dpi=750)


if __name__ == "__main__":
    visualize_BERT_predictions()

    visualize_production_data_ideology()
    visualize_production_data_ideology(bert_terms_only=True, title="bert_term_subset")
