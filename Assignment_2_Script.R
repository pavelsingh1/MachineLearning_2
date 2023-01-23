#Machine Learning, Assignment 2#
#Owner: JK, PS#

#Loading required packages#
library(vtable)
library(readr)
library(caret)


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

#Generating three data sets for each of the three questions:

df1<-subset(data, select = c(M6, A50A, A3Ar_w, A5_2015, A6, A8_2021, A9, A41, J2, J8, B14, M4, M20, J43))
View(df1)
df1_names<-c("y_interest", "gender", "age", "educ", "marital", "income", "employment", "educ_raised", 
             "risk", "retirement", "investment", "knowledge", "educ_fin", "goals")
colnames(df1)<-df1_names

#Replacing "Prefer not to say" with missing values
df1[df1==99]<-NA

#Transform outcome into binary variable
df1$y_interest[df1$y_interest !=1 | df1$y_interest != NA]<-0

#Check for correct imputation of values
sum(is.na(df1$y_interest))-table(data$M6)

#Needing to one-hot encode the other variables. First convert to factors.
str(df1)
names <- c("gender", "age", "educ", "marital", "income", "employment", "educ_raised", 
           "risk", "retirement", "investment", "knowledge", "educ_fin", "goals")
df1[,names] <- lapply(df1[,names] , factor)

#Then one-hot encode.
dmy1 <- dummyVars(" ~ .", data = df1)
df1 <- data.frame(predict(dmy1, newdata = df1))

#Check final dataframe
View(df1)


df2<-subset(data, select = c(M8, A50A, A3Ar_w, A5_2015, A6, A8_2021, A9, A41, J2, J8, B14, M4, M20, J43))
View(df2)

df3<-subset(data, select = c(M10, A50A, A3Ar_w, A5_2015, A6, A8_2021, A9, A41, J2, J8, B14, M4, M20, J43))
View(df3)
####Analysis####

#Logistic regression

#Classification tree