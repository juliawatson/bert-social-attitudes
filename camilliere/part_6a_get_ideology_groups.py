# Group participants based on their scores on the non-binary acceptance survey (nBAcc).
# 
# This relies on: camilliere_data.txt.
# This outputs: camilliere_data_with_ideology_bins.csv


import pandas as pd


def get_bin(gender_ideology_score, cutoff1, cutoff2, column_name):
    if gender_ideology_score < cutoff1:
        return f"{column_name}-low"
    elif gender_ideology_score < cutoff2:
        return f"{column_name}-mid"
    return f"{column_name}-high"


def bin_by_column(data, column_name):
    # Note that cut-offs are exclusive.
    if column_name in ["nBAcc"]:
        cutoff1 = 1
        cutoff2 = 3
    else:
        raise ValueError("not sure what cutoff to use")
    percentile_bins = [
        get_bin(gender_ideology_score, cutoff1, cutoff2, column_name)
        for gender_ideology_score in data[column_name]]
    return percentile_bins


if __name__ == "__main__":

    # Load experimental data; filter out practice items; drop nan rows (drops people who didn't do the post surveys)
    experimental_data = pd.read_csv("camilliere_data.txt", sep=" ")
    experimental_data = experimental_data[experimental_data["exp"] != "practice"]
    experimental_data = experimental_data.dropna()

    # Add columns 
    method = "scale_chunks"
    scale = "nBAcc"
    experimental_data[f"{scale}_{method}_bin"] = bin_by_column(experimental_data, scale)

    experimental_data.to_csv("camilliere_data_with_ideology_bins.csv")


# A histogram, indicating the number of participants with each score.
# This was used to determine where to draw cut-offs.

# In [8]: collections.Counter(experimental_data.groupby("ID").mean()["nBAcc"])
# Out[8]: Counter({1.0: 37, 0.0: 42, 2.0: 43, 3.0: 20, 4.0: 3, 5.0: 1})
# 0.0: 42
# 1.0: 37
# 2.0: 43
# 3.0: 20
# 4.0: 3
# 5.0: 1