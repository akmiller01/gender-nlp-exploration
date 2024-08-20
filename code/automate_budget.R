list.of.packages <- c("data.table", "ggplot2", "Hmisc", "tidyverse", "stringr")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only=T)

wd = "~/git/gender-nlp-exploration/"
setwd(wd)

# dat = fread("data/kenya_budget_predictions.csv")
dat = fread("data/eth_budget_predictions.csv")
dat$`__index_level_0__` = NULL
original_names = names(dat)[1:13]


# Set blanks to false and 0
blanks = c("", "-")
blank_indices = which(dat$text %in% blanks)
dat$`Gender equality - significant objective confidence`[blank_indices] = 0
dat$`Gender equality - significant objective predicted`[blank_indices] = F
dat$`Gender equality - principal objective confidence`[blank_indices] = 0
dat$`Gender equality - principal objective predicted`[blank_indices] = F

dat$`Climate adaptation - significant objective confidence`[blank_indices] = 0
dat$`Climate adaptation - significant objective predicted`[blank_indices] = F
dat$`Climate adaptation - principal objective confidence`[blank_indices] = 0
dat$`Climate adaptation - principal objective predicted`[blank_indices] = F

dat$`Climate mitigation - significant objective confidence`[blank_indices] = 0
dat$`Climate mitigation - significant objective predicted`[blank_indices] = F
dat$`Climate mitigation - principal objective confidence`[blank_indices] = 0
dat$`Climate mitigation - principal objective predicted`[blank_indices] = F

dat$`Disability - significant objective confidence`[blank_indices] = 0
dat$`Disability - significant objective predicted`[blank_indices] = F
dat$`Disability - principal objective confidence`[blank_indices] = 0
dat$`Disability - principal objective predicted`[blank_indices] = F

dat$`Principal gender equality` = F
dat$`Principal gender equality`[which(
  dat$`Gender equality - principal objective predicted` &
    dat$`Gender keyword match`
)] = T

dat$`Significant gender equality` = F
dat$`Significant gender equality`[which(
  dat$`Gender equality - significant objective predicted` &
    dat$`Gender keyword match`
)] = T

dat$`Principal climate adaptation` = F
dat$`Principal climate adaptation`[which(
  dat$`Climate adaptation - principal objective predicted` &
    dat$`Climate keyword match`
)] = T

dat$`Significant climate adaptation` = F
dat$`Significant climate adaptation`[which(
  dat$`Climate adaptation - significant objective predicted` &
    dat$`Climate keyword match`
)] = T

dat$`Principal climate mitigation` = F
dat$`Principal climate mitigation`[which(
  dat$`Climate mitigation - principal objective predicted` &
    dat$`Climate keyword match`
)] = T

dat$`Significant climate mitigation` = F
dat$`Significant climate mitigation`[which(
  dat$`Climate mitigation - significant objective predicted` &
    dat$`Climate keyword match`
)] = T

dat$`Principal disability` = F
dat$`Principal disability`[which(
  dat$`Disability - principal objective predicted` &
    dat$`Disability keyword match`
)] = T

dat$`Significant disability` = F
dat$`Significant disability`[which(
  dat$`Disability - significant objective predicted` &
    dat$`Disability keyword match`
)] = T

dat$`Principal all climate` = dat$`Principal climate adaptation` | dat$`Principal climate mitigation`

describe(dat$`Principal gender equality`)
describe(dat$`Principal all climate`)
describe(dat$`Principal disability`)

check_g = subset(dat, `Gender equality - principal objective confidence` > 0.9 & !`Gender keyword match`)
check_a = subset(dat, `Climate adaptation - principal objective confidence` > 0.9 & !`Climate keyword match`)
check_m = subset(dat, `Climate mitigation - principal objective confidence` > 0.9 & !`Climate keyword match`)
check_d = subset(dat, `Disability - principal objective confidence` > 0.9 & !`Disability keyword match`)

# View(check_d[,c("project_title", "short_description", "long_description", "Disability - principal objective confidence")])

check_rev = subset(dat, 
                   `Climate adaptation - principal objective confidence` < 0.1 &
                     `Climate mitigation - principal objective confidence` < 0.1 &
                     `Climate keyword match`
                  )

keep = c(original_names,
        "Principal gender equality",
        "Principal all climate",
        "Principal disability"
)

out_dat = dat[,keep, with=F]

# out_dat = out_dat[,c(
#   "year", "program", "vote", 
#   "Principal gender equality",
#   "Principal all climate",
#   "Principal disability",
#   "value"
# )]

# fwrite(out_dat,
#        "data/kenya_budget_automated.csv")

fwrite(out_dat,
       "data/eth_budget_automated.csv")
