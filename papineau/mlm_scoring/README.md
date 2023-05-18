# bert_gender: papineau stimuli - mlm_scoring method

The key difference between this approach and the simple/direct approach (in
../simple) is that, rather than using p(actress|context), we use a
masked language modeling approach. This allows us to use their full set of
stimuli.

This draws on:
* Nangia et al. (2020) https://arxiv.org/pdf/2010.00133.pdf
* Salazar et al. (2020) https://arxiv.org/pdf/1910.14659.pdf


## Step 0: Compute frequencies
  * Run script part_0_compute_bert_counts.py
  * This computes the counts in BERT's training data (bookcorpus + wikipedia)
  * This outputs wiki_counts.csv and bookcorpus_counts.csv

## Step 1: Generate sentences to feed into BERT
  * Run script part_1_create_stimuli.py
  * This outputs stimuli.csv

## Step 2: Compute BERT predictions for the stimuli

a) Compute BERT predictions per word for the sentences in stimuli.csv
  * Run script part_2a_compute_bert_predictions.py
  * This outputs bert_predictions.csv. This is different from the
    bert_predictions.csv in the simple approach, since it computes the
    probability of each masked word, which can be aggregated in the approaches
    described in the Nangia and Salazar papers above.
  * This relies on stimuli.csv from the previous step
  * It takes 7-8 hrs on my laptop.

b) Aggregate the results of step 2a into per-sentence probabilities.
  * Run script part_2b_process_bert_predictions.py
  * Outputs:
      bert_predictions_by_sentence_exclude_modified.csv
      bert_predictions_by_sentence_normalized_exclude_modified_frequency_reweighted.csv
      bert_predictions_averaged_exclude_modified_adoption_frequency_reweighted.csv
      bert_predictions_averaged_exclude_modified_compound_frequency_reweighted.csv
  * This relies on bert_predictions.csv from the previous step.
  * This relies on the frequency counts computed in step 0
  * This takes ~40 minutes on my laptop -- ran it twice

c) Evaluate the correlation between the simple method and the approximations
   with and without the modified included.
  * Run script part_2c_evaluate_correlation.py
  * Relies on: 
      output of 2b above.
      bert_predictions_expanded.csv from step 2b of the simple method.
  * This prints:
      bert_likelihood: r=0.4195, p=0.0
      corpus_prior: r=0.3708, p=0.0
      frequency_weighted_posterior: r=0.7600, p=0.0

## Step 3: Compute log likelihood of responses using BERT -- same as simple method

b) Compute the log likelihood of the production data based on BERT predictions
* Run script part_3b_bert_log_likelihood.py
* Relies on:
     output of part_3a_filter_production_data.py from simple method (human data)
     output of part 2b above (processed BERT predictions)
* Outputs results csvs to results/ dir.


## Step 4: Create visualizations -- same as simple method
* Run script part_4_create_visualizations.py
* This creates png files in the visualizations/ dir.

