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

#Gender (1 male, 2 female)
table(data$A50A)

length(which(data$M6==1 & data$A50A==1))/length(which(data$M6==1))
length(which(data$M6==1 & data$A50A==2))/length(which(data$M6==1))

length(which(data$M7==3 & data$A50A==1))/length(which(data$M7==3))
length(which(data$M7==3 & data$A50A==2))/length(which(data$M7==3))

length(which(data$M8==2 & data$A50A==1))/length(which(data$M8==2))
length(which(data$M8==2 & data$A50A==2))/length(which(data$M8==2))

length(which(data$M31==2 & data$A50A==1))/length(which(data$M31==2))
length(which(data$M31==2 & data$A50A==2))/length(which(data$M31==2))

length(which(data$M9==1 & data$A50A==1))/length(which(data$M9==1))
length(which(data$M9==1 & data$A50A==2))/length(which(data$M9==1))

length(which(data$M10==2 & data$A50A==1))/length(which(data$M10==2))
length(which(data$M10==2 & data$A50A==2))/length(which(data$M10==2))

#Financial Education
table(data$M20)
length(which(data$M20==3))/nrow(data)
length(which(data$M20==2))/nrow(data)

table(data$M4)
hist(data$M4[data$M4<97])
summary(data$M4[data$M4<97])

hist(data$M4[data$M4<97 & data$M8==2])
summary(data$M4[data$M4<97 & data$M8==2])
hist(data$M4[data$M4<97 & !data$M8==2])
summary(data$M4[data$M4<97 & !data$M8==2])



####Data cleaning####



####Analysis####

#Logistic regression

#Classification tree