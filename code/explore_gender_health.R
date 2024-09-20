# Setup ####
list.of.packages <- c("data.table", "rstudioapi", "ggplot2", "Hmisc", "tidyverse", "stringr", "scales")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only=T)

wd <- dirname(getActiveDocumentContext()$path) 
setwd(wd)
setwd("../")

crs = fread("large_data/crs_for_gender_climate_disability_predictions.csv")

textual_cols_for_classification = c(
  "project_title",
  "short_description",
  "long_description"
)

crs = crs %>%
  unite(text, all_of(textual_cols_for_classification), sep=" ", na.rm=T, remove=F)

crs$text = trimws(crs$text)

# Set blanks to false and 0
blanks = c("", "-")
blank_indices = which(crs$project_title %in% blanks & crs$short_description %in% blanks & crs$long_description %in% blanks)
crs$`Gender equality - significant objective confidence`[blank_indices] = 0
crs$`Gender equality - significant objective predicted`[blank_indices] = F
crs$`Gender equality - principal objective confidence`[blank_indices] = 0
crs$`Gender equality - principal objective predicted`[blank_indices] = F

crs$`Principal gender equality` = F
crs$`Principal gender equality`[which(crs$gender==2)] = T
crs$`Principal gender equality`[which(
  crs$`Gender equality - principal objective predicted` &
    crs$`Gender keyword match`
)] = T

crs$health = crs$sector_name %in% c(
  "I.2.b. Basic Health",
  "I.2.a. Health, General"
)

crs$commitment_year = as.numeric(substr(crs$commitment_date, 1, 4))

health_disb = sum(subset(crs, health & commitment_year %in% c(2018:2022))$usd_disbursement_deflated, na.rm=T)
health_comm = sum(subset(crs, health & commitment_year %in% c(2018:2022))$usd_commitment_deflated, na.rm=T)
dollar(health_disb * 1e6)
dollar(health_comm * 1e6)
label_percent()(health_disb / health_comm)

gender_disb = sum(subset(crs, `Principal gender equality` & commitment_year %in% c(2018:2022))$usd_disbursement_deflated, na.rm=T)
gender_comm = sum(subset(crs, `Principal gender equality` & commitment_year %in% c(2018:2022))$usd_commitment_deflated, na.rm=T)
dollar(gender_disb * 1e6)
dollar(gender_comm * 1e6)
label_percent()(gender_disb / gender_comm)

gender_health_disb = sum(subset(crs, `Principal gender equality` & health & commitment_year %in% c(2018:2022))$usd_disbursement_deflated, na.rm=T)
gender_health_comm = sum(subset(crs, `Principal gender equality` & health & commitment_year %in% c(2018:2022))$usd_commitment_deflated, na.rm=T)
dollar(gender_health_disb * 1e6)
dollar(gender_health_comm * 1e6)
label_percent()(gender_health_disb / gender_health_comm)
