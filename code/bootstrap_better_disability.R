list.of.packages <- c("data.table")
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

dat = read.csv(
  "large_data/iati_disability_predictions.csv",
  check.names=F,
  colClasses = list(
    character=c(
      "text",
      "Disability keyword match",
      "Disability - significant objective predicted",
      "Disability - principal objective predicted"
      ),
    numeric=c(
      "disability_sig",
      "Disability - significant objective confidence",
      "Disability - principal objective confidence"
    )
  )
)

names(dat) = c(
  "text", "sig", "match", "sig_pred", "sig_conf", "pri_pred", "pri_conf"
)

hist(dat$sig_conf)
hist(dat$pri_conf)
# End load ####

# There are also too many cases of donors not properly screening
# So throw out vocab matches that are marked as 0
dat = subset(dat,
  match == "True" |
    sig == 0
)


# Examples without vocabulary matches are so bad we just throw them out
dat = subset(dat,
  !(match == "True" & sig == 0)
)

# The model also can't know the information past the first 512 tokens
# Throw away too long and too short cases
count_spaces <- function(s) { sapply(gregexpr(" ", s), function(p) { sum(p>=0) } ) }
text_len = nchar(dat$text)
text_token_estimate = count_spaces(dat$text)
dat = dat[which(text_len > 10 & text_token_estimate < 500),]


# Strong significant signal, weak principal signal and vice versa
principal_percentiles = quantile(dat$pri_conf, probs=c(0.01, 0.1, 0.5, 0.9, 0.99))
significant_percentiles = quantile(dat$sig_conf, probs=c(0.01, 0.1, 0.5, 0.9, 0.99))

strong_principal_weak_significant = which(
  dat$pri_conf >= principal_percentiles[5] & # 99% percentile
    dat$sig_conf <= significant_percentiles[3] & # 50% median
    dat$sig == 2
)

strong_significant_weak_principal = which(
  dat$sig_conf >= significant_percentiles[4] & # 90% percentile
    dat$pri_conf <= principal_percentiles[3] & # 50% median
    dat$sig == 1
)

# False negative with low confidence
positive_principal = dat$sig == 2
positive_significant = dat$sig == 1
negative_gender = dat$sig == 0
principal_predicted = dat$pri_pred == "True"
significant_predicted = dat$sig_pred == "True"

fn_sig = which(
  positive_significant &
    !significant_predicted & 
    dat$sig_conf <= significant_percentiles[3] # 50% percentile
)

fn_pri = which(
  positive_principal &
    !principal_predicted & 
    dat$pri_conf <= principal_percentiles[3] # 50% percentile
)

# False positive with high confidence
fp_sig = which(
  !positive_significant &
    significant_predicted & 
    dat$sig_conf >= significant_percentiles[5] # 99% percentile
)

fp_pri = which(
  !positive_principal &
    principal_predicted & 
    dat$pri_conf >= principal_percentiles[5] # 99% percentile
)

# True negative with low confidence

tn = which(
  negative_gender &
    dat$sig_conf <= significant_percentiles[2] & # 10% percentile
    dat$pri_conf <= principal_percentiles[2] # 10% percentile
)

# Merge
training_indices =
  union(
    strong_principal_weak_significant,
    strong_significant_weak_principal
  )

training_indices =
  union(
    training_indices,
    fn_sig
  )
training_indices =
  union(
    training_indices,
    fn_pri
  )
training_indices =
  union(
    training_indices,
    fp_sig
  )
training_indices =
  union(
    training_indices,
    fp_pri
  )
training_indices =
  union(
    training_indices,
    tn
  )

training_data = dat[training_indices,]
t_neg = subset(training_data, sig == 0)
mean(t_neg$sig_pred == "False" & t_neg$pri_pred == "False")
t_sig = subset(training_data, sig == 1)
mean(t_sig$sig_pred == "True")
t_pri = subset(training_data, sig == 2)
mean(t_pri$pri_pred == "True")
table(training_data$sig)

training_data$label = "No disability objective"
training_data$label[which(training_data$sig == 1)] = "Significant disability objective"
training_data$label[which(training_data$sig == 2)] = "Principal disability objective"

keep = c("text", "label")
training_data = training_data[,keep]
fwrite(training_data, "large_data/disability_training_data.csv")

