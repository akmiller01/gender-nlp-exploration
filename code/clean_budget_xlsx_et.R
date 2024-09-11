list.of.packages <- c("data.table", "openxlsx", "zoo" , "tidyverse", "stringr")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only=T)

setwd("~/git/gender-nlp-exploration/")

years = 2011:2015

dat_list = list()
dat_index = 1

for(year in years){
  message(year)
  filename = paste0("large_data/eth_",year,".xlsx")
  dat_recurrent = read.xlsx(
    filename,
    sheet="RECURRENTin AMha",
    startRow = 5,
    cols = c(9:15)
  )
  names(dat_recurrent) = c(
    "public_body_code",
    "program_code",
    "activity_code",
    "description",
    "treasury",
    "retained",
    "total"
  )
  dat_recurrent$year = year
  dat_recurrent$type = "recurrent"
  dat_list[[dat_index]] = dat_recurrent
  dat_index = dat_index + 1
  
  if(!year %in%  c(2011,2015)){
    dat_capital = read.xlsx(
      filename,
      sheet="CAPITAL in AMHAric",
      startRow = 5,
      cols = c(12:23)
    )
    names(dat_capital) = c(
      "public_body_code",
      "program_code",
      "activity_code",
      "sub_program_code",
      "project_code",
      "description",
      "treasury",
      "assistance",
      "loan",
      "total"
    )
  } else {
    dat_capital = read.xlsx(
      filename,
      sheet="CAPITAL in AMHAric",
      startRow = 5,
      cols = c(13:24)
    )
    names(dat_capital) = c(
      "public_body_code",
      "program_code",
      "activity_code",
      "sub_program_code",
      "project_code",
      "description",
      "treasury",
      "retained",
      "assistance",
      "loan",
      "total"
    )
  }
  
  dat_capital$year = year
  dat_capital$type = "capital"
  dat_list[[dat_index]] = dat_capital
  dat_index = dat_index + 1
}

budget = rbindlist(dat_list, fill=T)

budget$project_name = NA
budget$project_name[which(!is.na(budget$project_code))] = budget$description[which(!is.na(budget$project_code))]
budget$project_name = na.locf(budget$project_name, na.rm=F)

budget$sub_program_name = NA
budget$sub_program_name[which(!is.na(budget$sub_program_code))] = budget$description[which(!is.na(budget$sub_program_code))]
budget$sub_program_name = na.locf(budget$sub_program_name, na.rm=F)

budget$activity_name = NA
budget$activity_name[which(!is.na(budget$activity_code))] = budget$description[which(!is.na(budget$activity_code))]
budget$activity_name = na.locf(budget$activity_name, na.rm=F)

budget$program_name = NA
budget$program_name[which(!is.na(budget$program_code))] = budget$description[which(!is.na(budget$program_code))]
budget$program_name = na.locf(budget$program_name, na.rm=F)

budget$public_body_name = NA
budget$public_body_name[which(!is.na(budget$public_body_code))] = budget$description[which(!is.na(budget$public_body_code))]
budget$public_body_name = na.locf(budget$public_body_name, na.rm=F)

budget = subset(budget, 
                (type == "recurrent" & !is.na(activity_code)) |
                (type == "capital" & !is.na(project_code))
                )

textual_cols_for_classification = c(
  "public_body_name",
  "program_name",
  "activity_name",
  "sub_program_name",
  "project_name"
)

budget = budget %>%
  unite(text, all_of(textual_cols_for_classification), sep=" ", na.rm=T, remove=F)

drop = c("public_body_code", "program_code", "activity_code", "description", "sub_program_code", "project_code")

budget_out = budget[,c(
  "year",
  "type",
  "public_body_name",
  "program_name",
  "activity_name",
  "sub_program_name",
  "project_name",
  "text",
  "treasury",
  "retained",
  "assistance",
  "loan",
  "total"
)]

fwrite(budget_out, "data/eth_budget.csv")
