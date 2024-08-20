list.of.packages <- c("data.table", "openxlsx", "zoo")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only=T)

setwd("~/git/gender-nlp-exploration/")

wb = loadWorkbook(file="large_data/Kenya.xlsx")
sheets = names(wb)

data_list = list()
data_index = 1

for(sheet in sheets){
  year = substr(sheet, 1, 4)
  dat = read.xlsx("large_data/Kenya.xlsx", sheet=sheet)
  names(dat) = c("program", "value")
  dat = subset(dat, !is.na(program))
  dat = subset(dat, program != "Programme")
  dat$vote = NA
  dat$vote[which(startsWith(dat$program, "Total Expenditure for Vote"))] = dat$program[which(startsWith(dat$program, "Total Expenditure"))]
  dat$vote = gsub("Total Expenditure for Vote ", "", dat$vote)
  dat$program = gsub("_x005F", "", dat$program, fixed=T)
  dat$program = gsub("_x000D_", " ", dat$program, fixed=T)
  dat$vote = gsub("_x005F", "", dat$vote, fixed=T)
  dat$vote = gsub("_x000D_", " ", dat$vote, fixed=T)
  dat$vote = na.locf(dat$vote, fromLast=T, na.rm=F)
  dat = subset(dat, !is.na(vote))
  dat = subset(dat, !startsWith(dat$program, "Total Expenditure for Vote"))
  dat$value = as.numeric(dat$value)
  dat$year = year
  data_list[[data_index]] = dat
  data_index = data_index + 1
}

kenya_budget = rbindlist(data_list)
kenya_budget$text = paste(kenya_budget$vote, kenya_budget$program)
fwrite(kenya_budget, "data/kenya_budget.csv")
