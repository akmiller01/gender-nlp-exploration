list.of.packages <- c("data.table")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only=T)

wd = "~/git/gender-nlp-exploration/"
setwd(wd)

crs_nonau = fread("large_data/crs_nonau_for_gender_climate_disability_automated.csv")

crs_nonau_agg = crs_nonau[,.(usd_disbursement_deflated=sum(usd_disbursement_deflated, na.rm=T)),by=.(
  year, `Principal gender equality`, `Principal all climate`, `Principal disability`, region_name
)]

fwrite(crs_nonau_agg, "data/crs_nonau_region.csv")

crs = fread("large_data/crs_nonau_for_gender_climate_disability_automated.csv")

crs_agg = crs[,.(usd_disbursement_deflated=sum(usd_disbursement_deflated, na.rm=T)),by=.(
  year, `Principal gender equality`, `Principal all climate`, `Principal disability`, flow_name
)]

fwrite(crs_agg, "data/crs_au_flow.csv")
