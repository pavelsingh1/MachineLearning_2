#Machine Learning, Assignment 2#
#Owner: JK, PS#

#Loading required packages#
library(vtable)
library(readr)


#Goal: Predicting financial literacy question answers

#####Dataset: National Financial Capability Survey####
data <- read_csv("2021-SxS-Data-and-Data-Info/NFCS 2021 State Data 220627.csv")
View(data)

####Getting Summary Statistics####
table(data$M6)
length(which(data$M6==1))/nrow(data)

table(data$M7)
length(which(data$M7==3))/nrow(data)

table(data$M8)
length(which(data$M8==98))/nrow(data)
length(which(data$M8==2))/nrow(data)

table(data$M31)
length(which(data$M31==2))/nrow(data)
length(which(data$M31==3))/nrow(data)


table(data$M9)
length(which(data$M9==1))/nrow(data)

table(data$M10)
length(which(data$M10==2))/nrow(data)

#Gender
table(data$A50A)


####Data cleaning####



####Analysis####

#Logistic regression

#Classification tree