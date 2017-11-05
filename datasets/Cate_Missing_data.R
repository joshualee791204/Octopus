df = read.csv("train.csv")
library(ggplot2)
library(gridExtra)
library(tabplot)
library(lsr)
library(corrplot)
library(dplyr)

########Transfer categorical columns to factor 
factorLevel <- list()
a <- file("data_description.txt", open="r")
f <-readLines(a)
for (line in f){
  if(!grepl("^[[:blank:]]", line) & grepl(": ", line)) {
    col_name <<- trimws(gsub(":.*", "", line))
  } else {
    level <- trimws(gsub("\t.*", "", line))
    if (level != "") {

      factorLevel[[col_name]] <- c(factorLevel[[col_name]], level)
    }
  }
}
close(a)
factorLevel
for (i in names(df)[-1]) {
  if (i %in% names(factorLevel)) {
    df[[i]] <- factor(df[[i]], 
    levels = factorLevel[[i]])
  } else {
    df[[i]] <- as.numeric(df[[i]])
  }
}



###########Assign value to all missing data of categorical columns

df = df %>% mutate_if(is.factor, as.character)

for (i in names(df)[-1]) {
  if (i %in% names(factorLevel)){
    df[[i]][is.na(df[[i]])] = "Octopus"
  }}

write.csv(df, file = "Data.csv")

