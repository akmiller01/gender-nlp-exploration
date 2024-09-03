# Setup ####
list.of.packages <- c("data.table", "rstudioapi")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only=T)

wd <- dirname(getActiveDocumentContext()$path) 
setwd(wd)
setwd("../")

dat = fread("large_data/crs_screened_climate_predictions.csv")
blanks = c("", "-", " ")
blank_indices = which(dat$project_title %in% blanks & dat$short_description %in% blanks & dat$long_description %in% blanks)
nonblank_indicies=  setdiff(1:nrow(dat), blank_indices)
dat = dat[nonblank_indicies,]

keep = c(
  "crs_id",
  "year", "donor_name", "recipient_name", "purpose_name", "sector_name",
  "project_title", "short_description", "long_description", "climate_adaptation", "climate_mitigation", "Climate adaptation - principal objective confidence",
  "Climate mitigation - principal objective confidence", "Climate keyword match"
)

dat = dat[,keep,with=F]
dat = unique(dat)
dat = subset(dat, crs_id!="")

actual_adaptation_mislabeled_mitigation = subset(dat,
    `Climate keyword match` &
    climate_adaptation==0 &
    climate_mitigation==2 &
    `Climate mitigation - principal objective confidence` < 0.5 &
    `Climate adaptation - principal objective confidence` > 0.95
)

fwrite(actual_adaptation_mislabeled_mitigation, "data/actual_adaptation_mislabeled_mitigation.csv")

actual_mitigation_mislabeled_adaptation = subset(dat,
  `Climate keyword match` &
  climate_mitigation==0 &
  climate_adaptation==2 &
  `Climate adaptation - principal objective confidence` < 0.5 &
  `Climate mitigation - principal objective confidence` > 0.95
)

fwrite(actual_mitigation_mislabeled_adaptation, "data/actual_mitigation_mislabeled_adaptation.csv")


non_climate_mislabel = subset(
  dat,
  !`Climate keyword match` &
    climate_adaptation==2 &
    climate_mitigation==2 &
    `Climate adaptation - principal objective confidence` < 0.01 &
    `Climate mitigation - principal objective confidence` < 0.01
)
fwrite(non_climate_mislabel, "data/non_climate_mislabeled.csv")
