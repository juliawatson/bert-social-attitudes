# Analyses with Papineau et al. (2022) stimuli

This directory includes analyses on data from Papineau et al. (2022).

There are two subdirectories, corresponding to two versions of the analysis:
* simple/ - the BERT "direct method" from Watson et al., which is only possible for
  a subset of stimuli.
* mlm_scoring - uses the method from Nangia et al. (2020) to approximate
  BERT predictions. This is possible for all stimuli, and is the main focus
  of the paper.