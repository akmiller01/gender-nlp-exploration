list.of.packages <- c("data.table", "ggplot2", "Hmisc", "tidyverse", "stringr", "eulerr")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only=T)

wd = "~/git/gender-nlp-exploration/"
setwd(wd)

crs = fread("data/crs_for_gender_climate_disability_automated.csv")

crs_agg = crs[,.(usd_disbursement_deflated=sum(usd_disbursement_deflated, na.rm=T)),by=.(
  year, `Principal gender equality`, `Principal all climate`, `Principal disability`
)]
