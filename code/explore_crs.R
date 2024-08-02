list.of.packages <- c("data.table", "ggplot2", "Hmisc", "tidyverse", "stringr")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only=T)

wd = "~/git/gender-nlp-exploration/"
setwd(wd)

quotemeta <- function(string) {
  str_replace_all(string, "(\\W)", "\\\\\\1")
}

remove_punct = function(string){
  str_replace_all(string, "[[:punct:]]", " ")
}

collapse_whitespace = function(string){
  str_replace_all(string, "\\s+", " ")
}

# supplemental_disability_keywords = c(
#   'albino',
#   'albinism',
#   'autism',
#   'autistic',
#   'chronic',
#   'deformity',
#   'deformities',
#   'déficience',
#   'difficult',
#   'difficulty',
#   'difficulties',
#   'disable',
#   'eye',
#   'eyes',
#   'handicapés',
#   'handicapées',
#   'helpage',
#   'impaired',
#   'impairment',
#   'impairments',
#   'inclusive',
#   'parkinson'
# )

crs = fread(paste0("large_data/crs_nonau_for_gender_climate_disability_predictions.csv"))
original_names = names(crs)[1:95]

# textual_cols_for_classification = c(
#   "project_title",
#   "short_description",
#   "long_description"
# )
# 
# crs = crs %>%
#   unite(text, all_of(textual_cols_for_classification), sep=" ", na.rm=T, remove=F)
# 
# crs$text = trimws(collapse_whitespace(remove_punct(tolower(crs$text))))
# supplemental_disability_regex = paste0(
#   "\\b",
#   paste(supplemental_disability_keywords, collapse="\\b|\\b"),
#   "\\b"
# )
# crs$`Supplemental disability keyword match` = grepl(supplemental_disability_regex, crs$text, perl=T, ignore.case = T)
# crs$`Disability keyword match` = crs$`Disability keyword match` | crs$`Supplemental disability keyword match`
# 
# crs[,c("text","Supplemental disability keyword match")] = NULL

# Set blanks to false and 0
blanks = c("", "-")
blank_indices = which(crs$project_title %in% blanks & crs$short_description %in% blanks & crs$long_description %in% blanks)
crs$`Gender equality - significant objective confidence`[blank_indices] = 0
crs$`Gender equality - significant objective predicted`[blank_indices] = F
crs$`Gender equality - principal objective confidence`[blank_indices] = 0
crs$`Gender equality - principal objective predicted`[blank_indices] = F

crs$`Climate adaptation - significant objective confidence`[blank_indices] = 0
crs$`Climate adaptation - significant objective predicted`[blank_indices] = F
crs$`Climate adaptation - principal objective confidence`[blank_indices] = 0
crs$`Climate adaptation - principal objective predicted`[blank_indices] = F

crs$`Climate mitigation - significant objective confidence`[blank_indices] = 0
crs$`Climate mitigation - significant objective predicted`[blank_indices] = F
crs$`Climate mitigation - principal objective confidence`[blank_indices] = 0
crs$`Climate mitigation - principal objective predicted`[blank_indices] = F

crs$`Disability - significant objective confidence`[blank_indices] = 0
crs$`Disability - significant objective predicted`[blank_indices] = F
crs$`Disability - principal objective confidence`[blank_indices] = 0
crs$`Disability - principal objective predicted`[blank_indices] = F

crs$`Principal gender equality` = F
crs$`Principal gender equality`[which(crs$gender==2)] = T
crs$`Principal gender equality`[which(
  crs$`Gender equality - principal objective predicted` &
    crs$`Gender keyword match`
)] = T

crs$`Significant gender equality` = F
crs$`Significant gender equality`[which(crs$gender==1)] = T
crs$`Significant gender equality`[which(
  crs$`Gender equality - significant objective predicted` &
    crs$`Gender keyword match`
)] = T

crs$`Principal climate adaptation` = F
crs$`Principal climate adaptation`[which(crs$climate_adaptation==2)] = T
crs$`Principal climate adaptation`[which(
  crs$`Climate adaptation - principal objective predicted` &
    crs$`Climate keyword match`
)] = T

crs$`Significant climate adaptation` = F
crs$`Significant climate adaptation`[which(crs$climate_adaptation==1)] = T
crs$`Significant climate adaptation`[which(
  crs$`Climate adaptation - significant objective predicted` &
    crs$`Climate keyword match`
)] = T

crs$`Principal climate mitigation` = F
crs$`Principal climate mitigation`[which(crs$climate_mitigation==2)] = T
crs$`Principal climate mitigation`[which(
  crs$`Climate mitigation - principal objective predicted` &
    crs$`Climate keyword match`
)] = T

crs$`Significant climate mitigation` = F
crs$`Significant climate mitigation`[which(crs$climate_mitigation==1)] = T
crs$`Significant climate mitigation`[which(
  crs$`Climate mitigation - significant objective predicted` &
    crs$`Climate keyword match`
)] = T

crs$`Principal disability` = F
crs$`Principal disability`[which(crs$disability==2)] = T
crs$`Principal disability`[which(
  crs$`Disability - principal objective predicted` &
    crs$`Disability keyword match`
)] = T

crs$`Significant disability` = F
crs$`Significant disability`[which(crs$disability==1)] = T
crs$`Significant disability`[which(
  crs$`Disability - significant objective predicted` &
    crs$`Disability keyword match`
)] = T

crs$`Principal all climate` = crs$`Principal climate adaptation` | crs$`Principal climate mitigation`

describe(crs$`Principal gender equality`)
describe(crs$`Principal all climate`)
describe(crs$`Principal disability`)

check_g = subset(crs, `Gender equality - principal objective confidence` > 0.9 & !`Gender keyword match`)
check_a = subset(crs, `Climate adaptation - principal objective confidence` > 0.9 & !`Climate keyword match`)
check_m = subset(crs, `Climate mitigation - principal objective confidence` > 0.9 & !`Climate keyword match`)
check_d = subset(crs, `Disability - principal objective confidence` > 0.9 & !`Disability keyword match`)

# View(check_d[,c("project_title", "short_description", "long_description", "Disability - principal objective confidence")])

check_rev = subset(crs, 
                   `Climate adaptation - principal objective confidence` < 0.1 &
                     `Climate mitigation - principal objective confidence` < 0.1 &
                     `Climate keyword match`
                  )

keep = c(original_names,
        "Principal gender equality",
        "Principal all climate",
        "Principal disability"
)

out_crs = subset(
  crs, 
  `Principal gender equality` |
    `Principal all climate` |
    `Principal disability`,select=keep
)

fwrite(out_crs,
       "large_data/crs_nonau_for_gender_climate_disability_automated.csv")
