list.of.packages <- c("data.table")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only=T)


# Load ####
wd = "~/git/gender-nlp-exploration/"
setwd(wd)

dat = read.csv(
  "large_data/iati_predictions2.csv",
  check.names=F,
  colClasses = list(
    character=c(
      "text",
      "label",
      "Gender equality - significant objective predicted",
      "Gender equality - principal objective predicted"
    ),
    numeric=c(
      "Gender equality - significant objective confidence",
      "Gender equality - principal objective confidence"
    )
  )
)

names(dat) = c(
  "text", "label", "sig_pred", "sig_conf", "pri_pred", "pri_conf"
)


true_negative = subset(
  dat, 
  label == "No gender equality objective" &
    sig_pred == "False" &
    pri_pred == "False"
)

true_pos_sig = subset(
  dat, 
  label == "Significant gender equality objective" &
    sig_pred == "True"
)

true_pos_pri = subset(
  dat, 
  label == "Principal gender equality objective" &
    pri_pred == "True"
)

false_neg_sig = subset(
  dat, 
  label == "Significant gender equality objective" &
    sig_pred == "False"
)

false_neg_pri = subset(
  dat, 
  label == "Principal gender equality objective" &
    pri_pred == "False"
)

false_pos_sig = subset(
  dat, 
  label != "Significant gender equality objective" &
    sig_pred == "True"
)

false_pos_pri = subset(
  dat, 
  label != "Principal gender equality objective" &
    pri_pred == "True"
)
