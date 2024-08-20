list.of.packages <- c("data.table", "openxlsx", "zoo")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only=T)

setwd("~/git/gender-nlp-exploration/")

wb = loadWorkbook(file="large_data/Uganda.xlsx")
sheets = names(wb)

data_list = list()
data_index = 1

for(sheet in sheets){
  dat = read.xlsx("large_data/Uganda.xlsx", sheet=sheet)
  dat[,c("Gender", "Climate", "Disability")] = NULL
  data_list[[data_index]] = dat
  data_index = data_index + 1
}

budget = rbindlist(data_list, fill=T)
budget$text = paste(
  budget$Vote,
  budget$Programme,
  budget$SubProgramme,
  budget$KeyOutputDescription,
  budget$Description
)
fwrite(budget, "data/uganda_budget.csv")
