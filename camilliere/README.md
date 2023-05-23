# Analyses with Camilliere et al. (2021) stimuli

This directory includes the code for analyses with stimuli from Camilliere et al. (2021).  Note that running these analyses requires their data, which is not included in this repository. We requested the data from the authors. (See description of the required data files at the end of this README.)

## Step 1: Generate masked stimuli to feed into BERT
  * Run script part_1_create_stimuli.py
  * This relies on camilliere_stimuli.csv
  * This outputs BERT_stimuli.csv

## Step 2: Compute BERT predictions for the stimuli
* Run script part_2_compute_bert_predictions.py
* This relies on BERT_stimuli.csv from the previous step
* This outputs bert_predictions.csv
* It takes ~2 minutes to run on my laptop

## Step 3: Create visualizations by condition
* Run script part_3_create_visualizations.py
* This relies on bert_predictions.csv from the previous step.
* This outputs visualizations/antecedent_type_surprisal.png and bert_predictions_with_p_they.csv.

## Step 4: Compute correlation with results by cluster
* Run script part_4_cluster_correlations.py
  - This prints correlations between BERT predictions and participant responses per cluster.
  - This relies on bert_predictions_with_p_they.csv from the previous step, as well as 
    camilliere_data.txt (participant responses from Camilliere et al., 2021).

* Run script part_4_recreate_camilliere_bargraph.py 
  - This recreates their bar graph, allowing us to double check which cluster is which
  - This outputs:
     * visualizations/camilliere_results_by_cluster.png
     * visualizations/camilliere_results_overall.png
  - This relies on camilliere_data.txt.

## Step 5: Regression evaluating the effect of gender and closeness
* Run script part_5_regression.R
* Relies on bert_predictions_with_p_they.csv from step 3.
* Outputs regression summary to: gender_closeness_regression.txt.
* Regression predicting they_surprisal ~ gender + socially_close
  with only ("socially distant, non-gendered antecedents; socially close,
  non-gendered antecedents; socially distant, gendered antecedents;
  and socially close, gendered antecedents"), following Camilliere et al.

## Step 6: Compute correlation with participants, grouped by gender ideology.
* Run script part_6a_get_ideology_groups.py
  - This groups participants based on their scores on the non-binary acceptance survey (nBAcc).
  - This relies on: camilliere_data.txt.
  - This outputs: camilliere_data_with_ideology_bins.csv

* Run script part_6b_ideology_correlations.py
  - This prints correlations between BERT predictions and participant responses, for participants
    grouped by social attitudes about gender (specificall, the non-binary acceptance survey).
  - This relies on bert_predictions_with_p_they.csv from step 3, as well as 
    camilliere_data_with_ideology_bins.csv from the previous step.


## Camilliere et al. Data:

* camilliere_stimuli.csv - The stimuli. I adapted this format from the javascript code for Camilliere et al's experiment. Columns include:
  - cond - experimental condition
  - itm - item identifier (1-55, 1-40 are eight critical conditions, 41-55 are inanimates)
  - sentence - the stimulus sentence where they refers to the antecedent
  - form: form of they used in the item (they, them, their, themselves)
  - antecedent: the antecedent (e.g., "the stewardess")

* camilliere_data.txt - The experimental results
