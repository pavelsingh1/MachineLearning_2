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
library(ggplot2)
library(car)
library(broom)
library(dplyr)

#library(rpart.plot)

#Goal: Predicting financial literacy question answers

#####Dataset: National Financial Capability Survey####
data <- read_csv("2021-SxS-Data-and-Data-Info/NFCS 2021 State Data 220627.csv")
View(data)

### (i had to add the following code in order to set the header to the actual column name)
### remove headers and replace by first row
names(data) <- as.matrix(data[1, ])
data <- data[-1, ]
data[] <- lapply(data, function(x) type.convert(as.character(x)))
data


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
summary(df2)  #### this should be summary df1 i assume

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

dim(df1)
df1<-df1[!is.na(df1$y_interest),]
dim(df1)
sum(is.na(df1$y_interest))

train1<-sample(c(TRUE, FALSE), nrow(df1), replace=TRUE, prob = c(0.5, 0.5))
test1<-!train1

#Check ratio
sum(train1)
sum(test1)
#Looks okay

#Generate the train datasets and the test datasets
df1_train<-df1[train1,]
df1_test<-df1[test1,]

df2_train<-df2[train,]      # here the variable gives us an error
df2_test<-df2[test,]        # update -> i realised its because of train and test not having the 1 behind which needs to be adjusted. 

df3_train<-df3[train,]      # same for this part
df3_test<-df3[test,]

####Logistic regression####

##Question on interest rates
model1_log <- glm(y_interest ~.,family=binomial(link='logit'),data=df1_train)
summary(model1_log)
fitted1<-predict(model1_log, newdata=subset(df1_test, select= -c(y_interest)))
View(fitted1)

#Calculating mean error
err1_log <- mean(as.numeric(fitted1 > 0.5) != df1_test$y_interest, na.rm=TRUE)

#Model diagnostics: checking for extreme values 
#(http://www.sthda.com/english/articles/36-classification-methods-essentials/148-logistic-regression-assumptions-and-diagnostics-in-r/#logistic-regression-diagnostics)
  plot(model1_log, which = 4, id.n = 3)

  model.data1 <- augment(model1_log) %>% 
  mutate(index = 1:n()) 

  model.data1 %>% top_n(3, .cooksd)

  ggplot(model.data1, aes(index, .std.resid)) + 
  geom_point(aes(color = "red"), alpha = .5) +
  theme_bw()
#Visual inspection: No standard residual in absolute value above three.

##Question on bonds
model2_log <- glm(y_bond ~.,family=binomial(link='logit'),data=df2_train)
summary(model2_log)
fitted2<-predict(model2_log, newdata=subset(df2_test, select=-c(y_bond)))
View(fitted2)

#Calculating mean error
err2_log <- mean(as.numeric(fitted2 > 0.5) != df2_test$y_bond, na.rm=TRUE)

#Model diagnostics: checking for extreme values
plot(model2_log, which = 4, id.n = 3)

model.data2 <- augment(model2_log) %>% 
  mutate(index = 1:n()) 

model.data2 %>% top_n(3, .cooksd)

ggplot(model.data2, aes(index, .std.resid)) + 
  geom_point(aes(color = "red"), alpha = .5) +
  theme_bw()
#Visual inspection: No standard residual in absolute value above three.

##Question on diversification
model3_log <- glm(y_diversification ~.,family=binomial(link='logit'),data=df3_train)
summary(model3_log)
fitted3<-predict(model3_log, newdata=subset(df3_test, select=-c(y_diversification)))
View(fitted3)

#Calculating mean error
err3_log <- mean(as.numeric(fitted3 > 0.5) != df3_test$y_diversification, na.rm=TRUE)

#Model diagnostics: checking for extreme values
plot(model3_log, which = 4, id.n = 3)

model.data3 <- augment(model3_log) %>% 
  mutate(index = 1:n()) 

model.data3 %>% top_n(3, .cooksd)

ggplot(model.data3, aes(index, .std.resid)) + 
  geom_point(aes(color = "red"), alpha = .5) +
  theme_bw()
#Visual inspection: Some standard residuals seem to be in absolute value slightly above three.
#Ignore, as only marginal. 


####Classification tree####

#Not working properly! Predictions are off. No idea how to fix it.

model1_tree<-rpart(y_interest~., data = df1_train, method="class")
summary(model1_tree)
#rpart.plot(model1_tree) #would show us the tree with the nodes
                 
 #model1_tree<-rpart(y_interest~., data = df1_train, method="class", control = rpart.control(minsplit = 20, minbucket = 7, maxdepth = 10, usesurrogate = 2, xval =10 ))
 # googled a bit and found this control command... dont know if its useful tho because the problem is that the predictions are not being created.                

fitted1_tree<-predict(model1_tree, data=df1_test, type="class")
err1_tree <- mean(as.numeric(fitted1_tree > 0.5) != df1_test$y_interest, na.rm=TRUE) 
# length(df1_test$y_interest) #length of both fitted1_tree and df_test$y_interest arent the same, meaning the err1_tree command doesnt work. 
                 
#table_mat <- table(df1_test$y_interest, fitted1_tree)
#table_mat    #creates a table to differentiate between correct and wrong decisions.            
                 
#Fitting does not work properly

 #https://cran.r-project.org/web/packages/tree/tree.pdf       #(page 13 of this website shows how to create class bar charts for classification tree)            
 
 #accuracy_Test <- sum(diag(table_mat)) / sum(table_mat)                
 #print(paste('Accuracy for test', accuracy_Test))  #to check accuracy from the table above
                 

train1_x = data.matrix(df1_train[, -1])
train1_y = df1_train[,1]

#define predictor and response variables in testing set
test1_x = data.matrix(df1_test[, -1])
test1_y = df1_test[, 1]

#define final training and testing sets
xgb_train1 = xgb.DMatrix(data = train1_x, label = train1_y)
xgb_test1 = xgb.DMatrix(data = test1_x, label = test1_y)

#defining a watchlist
watchlist = list(train=xgb_train1, test=xgb_test1)

#Setting the tuning parameters:
max.depth <- 2 #Maximum depth of each tree. Set to zero for no constraints
nrounds <- 1000 #Number of boosting iterations
eta <-0.1 #Standard is 0.3; learning rate
gamma<- 0 #Standard is 0; Minimum loss reduction required
subsample <- 1 #Standard is 1; prevent overfitting by randomly sampling the from training
min_child_weight <- 1 #Standard is 1; Minimum sum of instance weight (hessian) needed in a child
lambda <- 1 #Standard is 1; L2 regularization term on weights
alpha <- 0 #Standard is 0; L1 regularization term on weights.


#fit XGBoost model and display training and testing data at each iteration
model1_boost = xgb.train(data = xgb_train1, 
                  max.depth = max.depth, 
                  watchlist = watchlist, 
                  nrounds = nrounds,
                  eta = eta,
                  gamma = gamma,
                  subsample = subsample, 
                  min_child_weight = min_child_weight,
                  objective = "binary:logistic")

#Evaluation on Hold Out Sample
pred_y_interest = predict(model1_boost, xgb_test1)
err1_boost<-mean(as.numeric(pred_y_interest > 0.5) != df1_test$y_interest, na.rm=TRUE)

#Running diagnostics

xgb.plot.multi.trees(model=model1_boost)

# get information on how important each feature is
importance_matrix_1 <- xgb.importance(model = model1_boost)
# and plot it
xgb.plot.importance(importance_matrix_1)

########################Code seems to work until here#########################
#Current problem: Paramter Optimisation does not find the data and fails to initialize. Weird, as it is 
#more or less copy and paste of last assignments' code.
#Overview over parameters: https://xgboost.readthedocs.io/en/stable/parameter.html

#Parameter Optimisation
obj_func <- function(eta, max_depth, min_child_weight, subsample, lambda, alpha) {
  
  param <- list(
    
    xgb_train1 = xgb.DMatrix(data = train1_x, label = train1_y),
    
    # Hyperparameters 
    eta = eta,
    max_depth = max_depth,
    min_child_weight = min_child_weight,
    subsample = subsample,
    lambda = lambda,
    alpha = alpha,
    
    # Tree model; default booster
    booster = "gbtree",
    
    # Regression problem; default objective function
    objective = "binary:logistic",
    
    # Use RMSE
    eval_metric = "error")
  
  xgbcv <- xgb.cv(params = param,
                  data = xgb_train1,
                  nrounds = 100,
                  nfold=10,
                  prediction = TRUE,
                  early_stopping_rounds = 5,
                  verbose = 1,
                  maximize = F)
  
  lst <- list(
    
    # First argument must be named as "Score"
    # Function finds maxima so inverting the output
    Score = min(xgbcv$evaluation_log$test-error),
    
    # Get number of trees for the best performing model
    nrounds = xgbcv$best_iteration
  )
  
  return(lst)
}

bounds <- list(eta = c(0.0001, 0.3),
               max_depth = c(1L, 10L),
               min_child_weight = c(1, 50),
               subsample = c(0.1, 1),
               lambda = c(1, 10),
               alpha = c(0, 10))

#Setting Seed for reproducibility
set.seed(1234)

#Initializing the process to run in parallel
cl <- makeCluster(8)
registerDoParallel(cl)
clusterExport(cl,c("train1_x", "train1_y"))
clusterEvalQ(cl,expr= {
  library(xgboost)
})


#Bayesian Optimzation. Plot gives back the progress of the optimization. If lower plot (utility)
#is approaching zero, one can be optimistic that optimal parameter values were identified
#(see the instructions manual for bayesOpt)
bayes_out <- bayesOpt(FUN = obj_func, 
                      bounds = bounds, 
                      initPoints = length(bounds) + 2, 
                      iters.n = 30,
                      verbose=2,
                      plotProgress = TRUE,
                      parallel = TRUE)

# Show relevant columns from the summary object 
bayes_out$scoreSummary[1:5, c(3:8, 13)]
# Get best parameters
data.frame(getBestPars(bayes_out))

opt_params1 <- append(list(booster = "gbtree", 
                           objective = "binary:logistic", 
                           eval_metric = "error"), 
                      getBestPars(bayes_out))

# Run cross validation 
xgbcv1 <- xgb.cv(params = opt_params1,
                 data = xgb_train1,
                 nrounds = 100,
                 nfold=10,
                 prediction = TRUE,
                 early_stopping_rounds = 5,
                 verbose = 1,
                 maximize = F)
# Get optimal number of rounds
nrounds1 = xgbcv1$best_iteration

# Fit a xgb model
model1_boost_opt <- xgboost(data = xgb_train1, 
                  params = opt_params, 
                  maximize = F, 
                  early_stopping_rounds = 5, 
                  nrounds = nrounds2, 
                  verbose = 1
)

pred_y_interest_opt = predict(model1_boost_opt, xgb_test1)

