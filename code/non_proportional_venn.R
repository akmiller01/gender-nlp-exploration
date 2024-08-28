list.of.packages <- c("eulerr", "data.table", "VennDiagram", "gridExtra", "scales")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only=T)

wd = "~/git/gender-nlp-exploration/"
setwd(wd)

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

crs = fread("large_data/crs_for_gender_climate_disability_automated.csv")
crs = subset(crs, `Principal gender equality` | `Principal all climate` | `Principal disability`)
crs_agg = crs[,.(usd_disbursement_deflated=sum(usd_disbursement_deflated, na.rm=T)),by=.(
  `Principal gender equality`, `Principal all climate`, `Principal disability`
)]
setnames(crs_agg,
         c("Principal gender equality", "Principal all climate", "Principal disability"),
         c("Gender", "Climate", "Disability")
)
fwrite(crs_agg, "data/venn_data.csv")

# Make sets
c_value = round(
  sum(crs_agg[Climate==T]$usd_disbursement_deflated)
)
g_value = round(
  sum(crs_agg[Gender==T]$usd_disbursement_deflated)
)
d_value = round(
  sum(crs_agg[Disability==T]$usd_disbursement_deflated)
)
cg_value = round(
  sum(crs_agg[Climate==T & Gender==T]$usd_disbursement_deflated)
)
cd_value = round(
  sum(crs_agg[Climate==T & Disability==T]$usd_disbursement_deflated)
)
gd_value = round(
  sum(crs_agg[Gender==T & Disability==T]$usd_disbursement_deflated)
)
cgd_value = round(
  sum(crs_agg[Gender==T & Disability==T & Climate==T]$usd_disbursement_deflated)
)

# Build climate set
climate_elements = c(1:c_value)
gender_element_start = c_value + 1

# Build gender set
gender_elements = c(1:cg_value)
gender_remainder = (g_value - cg_value) - 1
gender_elements = c(
  gender_elements,
  gender_element_start:(gender_element_start + gender_remainder)
)
disability_element_start = (gender_element_start + gender_remainder) + 1

# Build disability set
disability_elements = c(1:cgd_value)
gd_remainder = (gd_value - cgd_value) - 1
disability_elements = c(
  disability_elements,
  gender_element_start:(gender_element_start + gd_remainder)
)
cd_remainder = (cd_value - cgd_value) - 1
disability_elements = c(
  disability_elements,
  (cg_value + 1):((cg_value + 1) + cd_remainder)
)
d_remainder = (((d_value - cd_value) - gd_value) + cgd_value) - 1
disability_elements = c(
  disability_elements,
  disability_element_start:(disability_element_start + d_remainder)
)

length(disability_elements)
length(climate_elements)
length(gender_elements)
length(
  intersect(intersect(gender_elements, disability_elements), climate_elements)
  )
length(
  intersect(climate_elements, disability_elements)
)

# venn.diagram(
#   x = list(climate_elements, gender_elements, disability_elements),
#   category.names=c("Climate", "Gender", "Disability"),
#   filename="data/unstyled_venn.tiff",
#   output=T,
#   disable.logging=T
# )
venn.diagram(
    x = list(climate_elements, gender_elements, disability_elements),
    category.names=c("Climate", "Gender", "Disability"),
    filename="data/styled_venn.png",
    output=TRUE,
    disable.logging=T,
    
    # Output features
    imagetype="png" ,
    height = 480 , 
    width = 480 , 
    resolution = 300,
    compression = "lzw",
    
    # Circles
    lwd = 2,
    lty = 'blank',
    fill = c(greens[1], blues[1], yellows[1]),
    
    # Numbers
    cex = .6,
    fontface = "bold",
    fontfamily = "sans",

    # Set names
    cat.cex = 0.6,
    cat.fontface = "bold",
    cat.default.pos = "outer",
    cat.pos = c(-27, 27, 135),
    cat.dist = c(0.055, 0.055, 0.085),
    cat.fontfamily = "sans",
    rotation = 1
)

fit = euler(list(
  "Gender"=gender_elements,
  "Climate"=climate_elements,
  "Disability"=disability_elements
), shape="ellipse")
plot(
  fit, 
  quantities = list(fontsize=8),
  legend=T,
  fills=c(blues[1], greens[1], yellows[1])
)
