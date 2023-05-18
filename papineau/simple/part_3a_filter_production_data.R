## Filter Papineau production data, using their filtering code.
## This flitering includes, e.g., selecting only critical trials and remove participants
## that failed attention checks.
## Adapted from gender_ideology/results/final/final_production_model.Rmd
## in https://github.com/BranPap/gender_ideology

library(ggplot2)
library(tidyverse)
library(lme4)
library(stringr)
library(languageR)
library(reshape2)
library(grid)

## Data Read-in


# This points to Papineau et al.'s production data in, which can be
# downloaded from their github repo: https://github.com/BranPap/gender_ideology
prod_data <- read.csv("../../../gender_ideology/results/final/production_data.csv")


## Exclusions


prod_exclusion <- prod_data %>% filter(name=='attention') %>%
  group_by(workerid) %>%
  summarise(accuracy = mean(correct)) %>%
  mutate(exclude = ifelse(accuracy < 0.80,'Yes','No')) %>%
  filter(exclude == "Yes")



prod_data <- prod_data[!(prod_data$workerid %in% prod_exclusion$workerid),]


## Additional Information

gender_transcendence_cols <- c('subject_information.gender_q1','subject_information.gender_q2','subject_information.gender_q3','subject_information.gender_q4','subject_information.gender_q5')

gender_linked_cols <- c('subject_information.gender_q6','subject_information.gender_q7','subject_information.gender_q8','subject_information.gender_q9','subject_information.gender_q10','subject_information.gender_q11','subject_information.gender_q12','subject_information.gender_q13')


prod_data <- prod_data %>%
  mutate(gender_trans = 100 - (rowMeans(prod_data[gender_transcendence_cols]))) %>%
  mutate(gender_link = rowMeans(prod_data[gender_linked_cols]))

gender_all = c('gender_trans','gender_link')

prod_data <- prod_data %>%
  mutate(gender_total = rowMeans(prod_data[gender_all]))


prod_data <- prod_data %>%
  filter(type == "critical") %>%
  mutate(response_gender = ifelse(response == "actress" | response == "anchorwoman" | response == "stewardess" | response == "businesswoman" | response == 'camerawoman' | response == 'congresswoman' | response == 'craftswoman' | response == 'crewwoman' | response == 'firewoman' | response == 'forewoman'  | response == 'heiress' | response == 'heroine' | response == 'hostess' | response == 'huntress' | response == 'laywoman' | response == 'policewoman' | response == 'saleswoman' | response == 'stuntwoman' | response == 'villainess' | response == 'weatherwoman',"female",ifelse(response == "anchor" | response == "flight attendant" | response == "businessperson" | response == 'camera operator' | response == 'congressperson' | response == 'craftsperson' | response == 'crewmember' | response == 'firefighter' | response == 'foreperson' | response == 'layperson' | response == 'police officer' | response == 'salesperson' | response == 'stunt double' | response == 'meteorologist',"neutral",ifelse(response == "anchorman" | response == "steward" | response == "businessman" | response == 'cameraman' | response == 'congressman' | response == 'craftsman' | response == 'crewman' | response == 'fireman' | response == 'foreman' | response == 'layman' | response == 'policeman' | response == 'salesman' | response == 'stuntman' | response == 'weatherman',"male",'neutral')))) %>%
  mutate(congruency = ifelse(gender == response_gender,"true","false")) %>%
  mutate(incongruent = ifelse(gender == "male" & response_gender == "female","incongruent_mtf",ifelse(gender == "female" & response_gender == "male","incongruent_ftm","real"))) %>%
  mutate(neutrality = ifelse(response_gender == "neutral","true","false"))%>%
  mutate(morph_type = ifelse(lexeme!= 'actor' & lexeme!= 'host' & lexeme !='hunter' & lexeme!= 'villain' & lexeme!= 'heir' & lexeme!= 'hero','compound','adoption')) %>%
  mutate(poli_party = ifelse(subject_information.party_alignment == 1 | subject_information.party_alignment == 2,'Republican',ifelse(subject_information.party_alignment == 4 | subject_information.party_alignment == 5,'Democrat','Non-Partisan'))) %>%
  mutate(response_neutral = ifelse(response_gender == "neutral",1,0)) %>%
  mutate(young_old = ifelse(subject_information.age > 40,"old","young")) %>%
  rename(form = response) %>%
  filter(!is.na(subject_information.age)) %>%
  filter(!is.na(poli_party))


write.csv(prod_data, "papineau_production_data_filtered.csv")
