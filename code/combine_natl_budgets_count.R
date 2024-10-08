list.of.packages <- c("data.table", "ggplot2", "Hmisc", "tidyverse", "stringr", "scales")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only=T)

wd = "~/git/gender-nlp-exploration/"
setwd(wd)

eth = fread("data/eth_budget_automated.csv")
eth = eth[,.(value=.N), by=.(
  `Principal gender equality`,
  `Principal all climate`,
  `Principal disability`
)]
eth$total = sum(eth$value)
eth$country = "Ethiopia"
ken = fread("data/kenya_budget_automated.csv")
ken$number = sapply(strsplit(ken$program, split=" "), `[[`, 1)
ken$aggregate = endsWith(ken$number, "000")
ken = subset(ken, !aggregate)
ken = ken[,.(value=.N), by=.(
  `Principal gender equality`,
  `Principal all climate`,
  `Principal disability`
)]
ken$total = sum(ken$value)
ken$country = "Kenya"
uga = fread("large_data/uganda_budget_automated.csv")
uga = uga[,.(value=.N), by=.(
  `Principal gender equality`,
  `Principal all climate`,
  `Principal disability`
)]
uga$total = sum(uga$value)
uga$country = "Uganda"

dat = rbind(eth, ken, uga)
dat$percentage = dat$value / dat$total
dat$label = ""
dat$label[which(
  dat$`Principal gender equality`
)] = "Gender"
dat$label[which(
  dat$`Principal all climate`
)] = "Climate change"
dat$label[which(
  dat$`Principal disability`
)] = "Disability"

dat = subset(dat, label != "")

reds = c(
  "#e84439", "#f8c1b2", "#f0826d", "#bc2629", "#8f1b13", "#fce3dc", "#fbd7cb", "#f6b0a0", "#ec6250", "#dc372d", "#cd2b2a", "#a21e25", "#6b120a"
)
oranges = c(
  "#eb642b", "#f6bb9d", "#f18e5e", "#d85b31", "#973915", "#fde5d4", "#fcdbbf", "#facbad", "#f3a47c", "#ee7644", "#cb5730", "#ac4622", "#7a2e05"
)
yellows = c(
  "#f49b21", "#fccc8e", "#f9b865", "#e48a00", "#a85d00", "#feedd4", "#fee7c1", "#fedcab", "#fac47e", "#f7a838", "#df8000", "#ba6b15", "#7d4712"
)
pinks = c(
  "#c2135b", "#e4819b", "#d64278", "#ad1257", "#7e1850", "#f9cdd0", "#f6b8c1", "#f3a5b6", "#e05c86", "#d12568", "#9f1459", "#8d0e56", "#65093d"
)
purples = c(
  "#893f90", "#c189bb", "#a45ea1", "#7b3b89", "#551f65", "#ebcfe5", "#deb5d6", "#cb98c4", "#af73ae", "#994d98", "#732c85", "#632572", "#42184c"
)
blues = c(
  "#0089cc", "#88bae5", "#5da3d9", "#0071b1", "#0c457b", "#d3e0f4", "#bcd4f0", "#a3c7eb", "#77adde", "#4397d3", "#105fa3", "#00538e", "#0a3a64"
)
greens = c(
  "#109e68", "#92cba9", "#5ab88a", "#1e8259", "#16513a", "#c5e1cb", "#b1d8bb", "#a2d1b0", "#74bf93", "#3b8c61", "#00694a", "#005b3e", "#07482e"
)
greys = c(
  "#6a6569", "#a9a6aa", "#847e84", "#555053", "#443e42", "#d9d4da", "#cac5cb", "#b3b0b7", "#b9b5bb", "#5a545a", "#736e73", "#4e484c", "#302b2e"
)

di_style = theme_bw() +
  theme(
    panel.border = element_blank()
    ,panel.grid.major.x = element_blank()
    ,panel.grid.minor.x = element_blank()
    ,panel.grid.major.y = element_line(colour = greys[2])
    ,panel.grid.minor.y = element_blank()
    ,panel.background = element_blank()
    ,plot.background = element_blank()
    ,axis.line.x = element_line(colour = "black")
    ,axis.line.y = element_blank()
    ,axis.ticks = element_blank()
    ,legend.position = "bottom"
  )

rotate_x_text_45 = theme(
  axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)
)

# CC
dat_max = max(subset(dat, label=="Climate change")$value)
ggplot(subset(dat, label=="Climate change"), aes(x=label, y=value, group=country, fill=country)) + 
  geom_bar(stat="identity", position="dodge") +
  geom_text(aes(label=value), position = position_dodge(width = .9), vjust=-1) +
  scale_fill_manual(values=reds) +
  scale_y_continuous(expand = c(0, 0), limits = c(0, dat_max * 1.1)) +
  di_style +
  labs(
    y="Count of relevant budget lines",
    x="",
    fill=""
  )

# G
dat_max = max(subset(dat, label=="Gender")$value)
ggplot(subset(dat, label=="Gender"), aes(x=label, y=value, group=country, fill=country)) + 
  geom_bar(stat="identity", position="dodge") +
  geom_text(aes(label=value), position = position_dodge(width = .9), vjust=-1) +
  scale_fill_manual(values=reds) +
  scale_y_continuous(expand = c(0, 0), limits = c(0, dat_max * 1.1)) +
  di_style +
  labs(
    y="Count of relevant budget lines",
    x="",
    fill=""
  )

# D
dat = rbind(
  dat,
  data.table(country="Kenya", label="Disability", value=0),
  fill=T
)
dat_max = max(subset(dat, label=="Disability")$value)
ggplot(subset(dat, label=="Disability"), aes(x=label, y=value, group=country, fill=country)) + 
  geom_bar(stat="identity", position="dodge") +
  geom_text(aes(label=value), position = position_dodge(width = .9), vjust=-1) +
  scale_fill_manual(values=reds) +
  scale_y_continuous(expand = c(0, 0), limits = c(0, dat_max * 1.1)) +
  di_style +
  labs(
    y="Count of relevant budget lines",
    x="",
    fill=""
  )

fwrite(dat, "data/natl_budgets_count.csv")
