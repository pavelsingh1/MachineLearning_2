#Machine Learning, Assignment 2#
#Owner: JK, PS#

#Loading required packages#
library(vtable)
library(readr)
library(caret)
library(xgboost)
library(ParBayesianOptimization)
library(doParallel)
library(rpart)

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

df1<-subset(data, select = c(M6, A50A, A3Ar_w, A5_2015, A6, A8_2021, A9, A41, J2, J8, B14, M4, 
                             M20, J43, A4A_new_w, A11, A7))
View(df1)
df1_names<-c("y_interest", "gender", "age", "educ", "marital", "income", "employment", "educ_raised", 
             "risk", "retirement", "investment", "knowledge", "educ_fin", "goals", "ethnicity", "children",
             "living")
colnames(df1)<-df1_names

#Replacing "Prefer not to say" with missing values
df1[df1==99]<-NA

#Transform outcome into binary variable
df1$y_interest[df1$y_interest !=1 | df1$y_interest != NA]<-0

#Check for correct imputation of values
sum(is.na(df1$y_interest))
table(data$M6)

#Needing to one-hot encode the other variables. First convert to factors.
str(df1)
names <- c("gender", "age", "educ", "marital", "income", "employment", "educ_raised", 
           "risk", "retirement", "investment", "knowledge", "educ_fin", "goals", "ethnicity", "children",
           "living")
df1[,names] <- lapply(df1[,names] , factor)

#Then one-hot encode.
dmy1 <- dummyVars(" ~ .", data = df1)
df1 <- data.frame(predict(dmy1, newdata = df1))

#Check final dataframe
View(df1)
summary(df2)

#Repeat for the question on bonds:
df2<-subset(data, select = c(M8, A50A, A3Ar_w, A5_2015, A6, A8_2021, A9, A41, J2, J8, B14, M4, M20, J43,
                             A4A_new_w, A11, A7))
df2_names<-c("y_bond", "gender", "age", "educ", "marital", "income", "employment", "educ_raised", 
             "risk", "retirement", "investment", "knowledge", "educ_fin", "goals", "ethnicity", "children",
             "living")
colnames(df2)<-df2_names

#Replacing "Prefer not to say" with missing values
df2[df2==99]<-NA

#Transform outcome into binary variable
df2$y_bond[df2$y_bond !=1 | df2$y_bond != NA]<-0

#Check for correct imputation of values
sum(is.na(df2$y_bond))
table(data$M8)

#Needing to one-hot encode the other variables. First convert to factors.
str(df2)

df2[,names] <- lapply(df2[,names] , factor)

#Then one-hot encode.
dmy2 <- dummyVars(" ~ .", data = df2)
df2 <- data.frame(predict(dmy2, newdata = df2))

#Check final dataframe
View(df2)
summary(df2)

#Repeat for the question on diversification:
df3<-subset(data, select = c(M10, A50A, A3Ar_w, A5_2015, A6, A8_2021, A9, A41, J2, J8, B14, M4, M20, J43,
                             A4A_new_w, A11, A7))
df3_names<-c("y_diversification", "gender", "age", "educ", "marital", "income", "employment", "educ_raised", 
             "risk", "retirement", "investment", "knowledge", "educ_fin", "goals", "ethnicity", "children",
             "living")
colnames(df3)<-df3_names

#Replacing "Prefer not to say" with missing values
df3[df3==99]<-NA

#Transform outcome into binary variable
df3$y_diversification[df3$y_diversification !=1 | df3$y_diversification != NA]<-0

#Check for correct imputation of values
sum(is.na(df3$y_diversification))
table(data$M10)

#Needing to one-hot encode the other variables. First convert to factors.
str(df3)

df3[,names] <- lapply(df3[,names] , factor)

#Then one-hot encode.
dmy3 <- dummyVars(" ~ .", data = df3)
df3 <- data.frame(predict(dmy3, newdata = df3))

#Check final dataframe
View(df3)
summary(df3)

#####Analysis#####

#Splitting the data randomly in half
set.seed(12345)

train<-sample(c(TRUE, FALSE), nrow(data), replace=TRUE, prob = c(0.5, 0.5))
test<-!train

#Check ratio
sum(train)
sum(test)
#Looks okay

#Generate the train datasets and the test datasets
df1_train<-df1[train,]
df1_test<-df1[test,]

df2_train<-df2[train,]
df2_test<-df2[test,]

df3_train<-df3[train,]
df3_test<-df3[test,]

####Logistic regression####

##Question on interest rates
model1_log <- glm(y_interest ~.,family=binomial(link='logit'),data=df1_train)
summary(model1_log)
fitted1<-predict(model1_log, newdata=subset(df1_test, select= -c(y_interest)))
View(fitted1)

#Calculating mean error
err1_log <- mean(as.numeric(fitted1 > 0.5) != df1_test$y_interest, na.rm=TRUE)

##Question on bonds
model2_log <- glm(y_bond ~.,family=binomial(link='logit'),data=df2_train)
summary(model2_log)
fitted2<-predict(model2_log, newdata=subset(df2_test, select=-c(y_bond)))
View(fitted2)

#Calculating mean error
err2_log <- mean(as.numeric(fitted2 > 0.5) != df2_test$y_bond, na.rm=TRUE)

##Question on diversification
model3_log <- glm(y_diversification ~.,family=binomial(link='logit'),data=df3_train)
summary(model3_log)
fitted3<-predict(model3_log, newdata=subset(df3_test, select=-c(y_diversification)))
View(fitted3)

#Calculating mean error
err3_log <- mean(as.numeric(fitted3 > 0.5) != df3_test$y_diversification, na.rm=TRUE)

####Classification tree####

#Not working properly! Predictions are off. No idea how to fix it.

model1_tree<-rpart(y_interest~., data = df1_train, method="class")
summary(model1_tree)

fitted1_tree<-predict(model1_tree, data=df1_test, type="class")
err1_tree <- mean(as.numeric(fitted1_tree > 0.5) != df1_test$y_interest, na.rm=TRUE)