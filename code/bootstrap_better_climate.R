list.of.packages <- c("data.table", "stringr")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only=T)

# Criteria for good training data:
#    Not too short, but try to fit within context window
#    Not clearly mislabeled
#    Strong significant signal, weak principal signal true positive
#    Strong principal signal, weak significant signal true positive
#    False negative with low confidence
#    False positive with high confidence
#    True negative with low confidence


# Load ####
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

dat = read.csv(
  "large_data/iati_climate_predictions.csv",
  check.names=F,
  colClasses = list(
    character=c(
      "text",
      "Climate keyword match"
    ),
    numeric=c(
      "climate_adaptation_sig",
      "climate_mitigation_sig",
      "Climate adaptation - significant objective",
      "Climate adaptation - principal objective",
      "Climate mitigation - significant objective",
      "Climate mitigation - principal objective"
    )
  )
)

dat$adapt_sig_pred = dat$`Climate adaptation - significant objective` > 0.5
dat$adapt_pri_pred = dat$`Climate adaptation - principal objective` > 0.5
dat$mitig_sig_pred = dat$`Climate mitigation - significant objective` > 0.5
dat$mitig_pri_pred = dat$`Climate mitigation - principal objective` > 0.5

names(dat) = c(
  "text", "mitig_sig", "adapt_sig", "match", "adapt_sig_conf", "adapt_pri_conf",
  "mitig_sig_conf", "mitig_pri_conf", "adapt_sig_pred", "adapt_pri_pred",
  "mitig_sig_pred", "mitig_pri_pred"
)

hist(dat$adapt_sig_conf)
hist(dat$adapt_pri_conf)
hist(dat$mitig_sig_conf)
hist(dat$mitig_pri_conf)
# End load ####

#Add new vocabulary
dat$clean_text = trimws(collapse_whitespace(remove_punct(tolower(dat$text))))

supplemental_vocab = c(
  "clim",
  "klimatanpassning",
  "sequías",
  "sequía",
  "sequias",
  "sequia",
  "inundaciones",
  "inun daciones",
  "inundación",
  "inundacion",
  "gef",
  "adaption",
  "climateshot",
  "landdegradatie",
  "catastrophic",
  "risk insurance",
  "ndcs",
  "global warming",
  "cng",
  "air",
  "coal",
  "cleaner",
  "ozone",
  "montreal",
  "paris",
  "kyoto",
  "energetica"
)

supplemental_vocab = quotemeta(trimws(collapse_whitespace(remove_punct(tolower(supplemental_vocab)))))

supplemental_regex = paste0(
  "\\b",
  paste(supplemental_vocab, collapse="\\b|\\b"),
  "\\b"
)

dat$supplemental_match = grepl(supplemental_regex, dat$clean_text, perl=T, ignore.case = T)

dat$match[which(dat$supplemental_match)] = "True"

dat[,c("clean_text", "supplemental_match")] = NULL

# Fix known mislabels
dat$adapt_sig[which(dat$adapt_sig > 2)] = 2
dat$mitig_sig[which(dat$mitig_sig > 2)] = 2

# There are too many cases of donors not properly screening
# So throw out vocab matches that are marked as 0
dat = subset(dat,
             !(
               match == "True" & adapt_sig == 0 & mitig_sig == 0
             )
)

# Examples without vocabulary matches are so bad we just throw them out
dat = subset(dat,
             !(
               match == "False" & (adapt_sig > 0 | mitig_sig > 0)
             )
)

# The model also can't know the information past the first 512 tokens
# Throw away too long and too short cases
count_spaces <- function(s) { sapply(gregexpr(" ", s), function(p) { sum(p>=0) } ) }
text_len = nchar(dat$text)
text_token_estimate = count_spaces(dat$text)
dat = dat[which(text_len > 10 & text_token_estimate < 500),]


# Strong significant signal, weak principal signal and vice versa
adaptation_strong_principal_weak_significant = which(
  dat$adapt_pri_conf >= (dat$adapt_sig_conf + 0.01) &
    dat$adapt_sig == 2
)

mitigation_strong_principal_weak_significant = which(
  dat$mitig_pri_conf >= (dat$mitig_sig_conf + 0.01) &
    dat$mitig_sig == 2
)

adaptation_strong_significant_weak_principal = which(
  dat$adapt_sig_conf >= (dat$adapt_pri_conf + 0.68)  & 
    dat$adapt_sig == 1
)

mitigation_strong_significant_weak_principal = which(
  dat$mitig_sig_conf >= (dat$mitig_pri_conf + 0.62)  & 
    dat$mitig_sig == 1
)

# Joint confidences
adaptation_significant_stronger_than_mitigation_significant = which(
  dat$adapt_sig_conf >= (dat$mitig_sig_conf + 0.5) &
    dat$adapt_sig == 1 &
    (dat$adapt_sig > dat$mitig_sig)
)

mitigation_significant_stronger_than_adaptation_significant = which(
  dat$mitig_sig_conf >= (dat$adapt_sig_conf + 0.1) &
    dat$mitig_sig == 1 &
    (dat$mitig_sig > dat$adapt_sig)
)

adaptation_principal_stronger_than_mitigation_principal = which(
  dat$adapt_pri_conf >= (dat$mitig_pri_conf + 0.6) &
    dat$adapt_sig == 2 &
    (dat$adapt_sig > dat$mitig_sig)
)

mitigation_principal_stronger_than_adaptation_principal = which(
  dat$mitig_pri_conf >= (dat$adapt_pri_conf + 0.3) &
    dat$mitig_sig == 2 &
    (dat$mitig_sig > dat$adapt_sig)
)


# False negative with low confidence
positive_adaptation_principal = dat$adapt_sig == 2
positive_mitigation_principal = dat$mitig_sig == 2
positive_adaptation_significant = dat$adapt_sig == 1
positive_mitigation_significant = dat$mitig_sig == 1
negative_adaptation = dat$adapt_sig == 0
negative_mitigation = dat$mitig_sig == 0
adaptation_principal_predicted = dat$adapt_pri_pred
mitigation_principal_predicted = dat$mitig_pri_pred
adaptation_significant_predicted = dat$adapt_sig_pred
mitigation_significant_predicted = dat$mitig_sig_pred

fn_sig_a = which(
  positive_adaptation_significant &
    !adaptation_significant_predicted & 
    dat$adapt_sig_conf <= 0.3
)

fn_sig_m = which(
  positive_mitigation_significant &
    !mitigation_significant_predicted & 
    dat$mitig_sig_conf <= 0.25
)

fn_pri_a = which(
  positive_adaptation_principal &
    !adaptation_principal_predicted & 
    dat$adapt_pri_conf <= 0.4
)

fn_pri_m = which(
  positive_mitigation_principal &
    !mitigation_principal_predicted & 
    dat$mitig_pri_conf <= 0.4
)

# False positive with high confidence
fp_sig_a = which(
  !positive_adaptation_significant &
    adaptation_significant_predicted & 
    dat$adapt_sig_conf >= 0.976
)

fp_sig_m = which(
  !positive_mitigation_significant &
    mitigation_significant_predicted & 
    dat$mitig_sig_conf >= 0.985
)

fp_pri_a = which(
  !positive_adaptation_principal &
    adaptation_principal_predicted & 
    dat$adapt_pri_conf >= 0.98
)

fp_pri_m = which(
  !positive_mitigation_principal &
    mitigation_principal_predicted & 
    dat$mitig_pri_conf >= 0.98
)

# Investigate false positives for new vocab
all_fp = union(fp_sig_a, fp_sig_m)
all_fp = union(all_fp, fp_pri_a)
all_fp = union(all_fp, fp_pri_m)

fp_dat = dat[all_fp,]
fp_dat = subset(fp_dat, adapt_sig == 0 & mitig_sig == 0)

# True negative with low confidence

tn_a = which(
  negative_adaptation &
    dat$adapt_sig_conf <= 0.0128 &
    dat$adapt_pri_conf <= 0.0128
)

tn_m = which(
  negative_mitigation &
    dat$mitig_sig_conf <= 0.01132 &
    dat$mitig_pri_conf <= 0.01132
)

# Merge
training_indices = c()
to_union = c(
  "adaptation_strong_principal_weak_significant",
  "mitigation_strong_principal_weak_significant",
  "adaptation_strong_significant_weak_principal",
  "adaptation_strong_significant_weak_principal",
  "adaptation_significant_stronger_than_mitigation_significant",
  "mitigation_significant_stronger_than_adaptation_significant",
  "adaptation_principal_stronger_than_mitigation_principal",
  "mitigation_principal_stronger_than_adaptation_principal",
  "fn_sig_a",
  "fn_sig_m",
  "fn_pri_a",
  "fn_pri_m",
  "fp_sig_a",
  "fp_sig_m",
  "fp_pri_a",
  "fp_pri_m",
  "tn_a",
  "tn_m"
)

for(var_name in to_union){
  training_indices = union(training_indices, get(var_name))
}

training_data = dat[training_indices,]
t_neg_a = subset(training_data, adapt_sig == 0)
mean(!t_neg_a$adapt_sig_pred & !t_neg_a$adapt_pri_pred)
t_neg_m = subset(training_data, mitig_sig == 0)
mean(!t_neg_m$mitig_sig_pred & !t_neg_m$mitig_pri_pred)

t_sig_a = subset(training_data, adapt_sig == 1)
mean(t_sig_a$adapt_sig_pred)
t_sig_m = subset(training_data, mitig_sig == 1)
mean(t_sig_m$mitig_sig_pred)

t_pri_a = subset(training_data, adapt_sig == 2)
mean(t_pri_a$adapt_pri_pred)
t_pri_m = subset(training_data, mitig_sig == 2)
mean(t_pri_m$mitig_pri_pred)

table(training_data$adapt_sig)
table(training_data$mitig_sig)

training_data$adaptation_label = "No climate adaptation objective"
training_data$adaptation_label[which(training_data$adapt_sig == 1)] = "Significant climate adaptation objective"
training_data$adaptation_label[which(training_data$adapt_sig == 2)] = "Principal climate adaptation objective"

training_data$mitigation_label = "No climate mitigation objective"
training_data$mitigation_label[which(training_data$mitig_sig == 1)] = "Significant climate mitigation objective"
training_data$mitigation_label[which(training_data$mitig_sig == 2)] = "Principal climate mitigation objective"


keep = c("text", "adaptation_label", "mitigation_label")
training_data = training_data[,keep]
fwrite(training_data, "large_data/curated_climate_training_data.csv")
