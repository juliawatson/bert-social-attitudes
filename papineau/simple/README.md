# bert_gender: papineau stimuli - simple method

This directory uses the direct/simple method of obtaining predictions from BERT, which only works for a subset of the stimuli from Papineau et al. (2022).

## Step 1: Generate masked stimuli

a) Identify sets of words that are appropriate for masking
  * Run script part_1a_bert_tokenization_role_nouns.py
  * This outputs role_noun_tokenization.csv

b) Generate stimuli to feed into BERT
  * Run script part_1b_create_stimuli.py
  * This relies on role_noun_tokenization.csv from the previous step
  * This outputs stimuli.csv


## Step 2: Compute BERT predictions for the stimuli

a) Compute BERT predictions for the stimuli in stimuli.csv
* Run script part_2a_compute_bert_predictions.py
* This outputs bert_predictions.csv
* This relies on stimuli.csv from the previous step
* It takes 10-15 minutes to run on my laptop

b) Unpack the BERT predictions so they are useful for regressions and visualizations
* Run script part_2b_process_bert_predictions.py
* Outputs:
  bert_predictions_expanded.csv
  bert_predictions_averaged_compound.csv
  bert_predictions_averaged_adoption.csv
* This relies on bert_predictions.csv from the previous step


## Step 3: Compute log likelihood of responses using BERT

a) Filter Papineau production data (select only critical trials; 
   remove participants that failed attention checks)
* Run script part_3a_filter_production_data.R
* Relies on having downloaded Papineau production data, which is in their github repo 
  (https://github.com/BranPap/gender_ideology). 
  Right now, this script is set up to run with the Papineau et al. github repo 
  cloned in the same directory as this github repo.
  If you clone it somewhere else, you will need to update the path where data is loaded
  from in this script.
* Outputs papineau_production_data_filtered.csv

b) Compute the log likelihood of the production data based on BERT predictions
* Run script part_3b_bert_log_likelihood.py
* Relies on:
  bert_predictions_averaged_compound.csv
  bert_predictions_averaged_adoption.csv 
  papineau_production_data_filtered.csv
* Outputs results to:
  results/compound_loglik.csv
  results/adoption_loglik.csv


## Step 4: Create visualizations
* Run script part_4_create_visualizations.py
* Relies on:
  bert_predictions_averaged_compound.csv
  bert_predictions_averaged_adoption.csv 
* Outputs visualizations to visualizations/ dir
