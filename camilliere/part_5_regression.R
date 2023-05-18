# Run linear mixed effects regression predicting BERT surprisal, as a function
# of gender and closeness. 
#
# A regression summary is saved to gender_closeness_regression.txt.
# This relies on: bert_predictions_with_p_they.csv

require(lme4)
library(lmerTest)

data <- read.csv("bert_predictions_with_p_they.csv")

gendered_conditions <- c("Socially distant gendered", "Socially close gendered")
socially_close_conditions <- c("Socially close non-gendered", "Socially close gendered")
conditions <- c("Socially distant gendered", "Socially close gendered",
                "Socially close non-gendered", "Socially distant non-gendered")

# Filter to remove antecedent types that aren't in conditions
data <- data[data$antecedent_type %in% conditions,]

# Add columns gendered and closeness for use in regression
data["gendered"] <- as.integer(ifelse(data$antecedent_type %in% gendered_conditions, 1, -1))
data["close"] <- as.integer(ifelse(data$antecedent_type %in% socially_close_conditions, 1, -1))

# Left out gendered:close interaction because it resulted in isSingular error
model <- lmer(
  surprisal_they ~ gendered + close + (1|itm),
  data=data)

sink("gender_closeness_regression.txt")
summary(model)
anova(model)
sink()
