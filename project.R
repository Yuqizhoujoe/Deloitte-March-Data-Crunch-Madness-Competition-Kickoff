require(caret)
require(caretEnsemble)
require(doParallel)
require(dplyr)
require(usdm)
require(tidyverse)
require(randomForest)
require(ROCR)
require(MASS)
require(glmnet)
require(dplyr)
require(neuralnet)
require(tidyselect)
require(Hmisc)
require(corrplot)
require(GGally)
require(elasticnet)
require(brnn)
require(obliqueRF)
require(C50)
require(gbm)
require(earth)
require(pdp)
require(vip)
require(xgboost)
require(broom)
require(e1071)
require(earth)
require(Formula)
require(plotmo)
require(plotrix)

data <- read_csv("swap_data_50%.csv", col_names = TRUE)[,-1]
dim(data)
# Since we decide to only keep numeric data, we will remove character information
# Take a look at which variables are characters 
data <- data[,!sapply(data, is.character)]
# Then we have a data set with 1111 rows and 93 columns after removing character variables 
dim(data)

# remove redundant columns
data <- data[,!names(data)%in%c("team1_id","team2_id","team1_score",
                                "team2_score","num_ot","season")] 
# After removing redundant variables, we have 86 explanatory variables
dim(data)

# missing values
apply(is.na(data),2,sum)
# we only have 1111 records but those variables have over 500 missing values.
# There is no doubt that we will remove them all
temp <- apply(is.na(data),2,sum)
temp <- as.data.frame(as.table(temp))
temp <- temp$Var1[temp$Freq!=0]
temp
data <- data[,!colnames(data)%in%temp]
dim(data)
rm(temp)
# strongseed and weakseed might be repetitive to team1_seed and team2_seed
# thus we are not going include them in our new dataset
new_data = data[,!colnames(data)%in%c('strongseed', 'weakseed')]
dim(new_data)

# check for infinite data
sapply(new_data, function(y) sum(length(which(is.infinite(y)))))

# Modeling
registerDoParallel(3)
getDoParWorkers()

set.seed(123)
dt <- createDataPartition(new_data$result.id, p=0.7,list = FALSE) 
train <- new_data[dt,]
test <- new_data[-dt,]

prop.table(table(train$result.id))
prop.table(table(test$result.id))

x_train <- model.matrix(result.id~., data = train)[,-1]
# convert dummy dependent variable into character for modeling later
y_train <- as.factor(ifelse(train$result.id==1,"yes","no"))
x_test <- model.matrix(result.id~.,data = test)[,-1]
y_test <- as.factor(ifelse(test$result.id==1, "yes","no"))
registerDoParallel(4)
getDoParWorkers()

# k-fold cross validation 
my_train <- trainControl(method = "repeatedcv", # repeated cross validation
                         number = 10, # 10 folds
                         repeats = 5, # repeat 5 times
                         savePredictions = "final",
                         classProbs = TRUE,
                         allowParallel = TRUE,
                         index = createResample(y_train,2),# list with elements for each resampling iteration
                         search = "grid") # find the optimal model hyperparameter 
metric <- "Accuracy"

# logistic regression 
fit.glm <- train(x=x_train, y=y_train, method="glm",metric=metric, trControl=my_train)
# regularization regression 
fit.glmnet <- train(x=x_train,y=y_train,method="glmnet",metric=metric,trControl=my_train) 
# tune hyperparameters - alpha and lambda
tuneglmnet <- expand.grid(alpha = seq(0,1,by=0.1),lambda = seq(0, 1, by=0.1))
fit.glmnet1 <- train(x=x_train,y=y_train,method="glmnet",metric=metric,
                     trControl=my_train,
                     tuneGrid=tuneglmnet) 
glmnetresult <- resamples(list(glm=fit.glm,
                               glmnet=fit.glmnet,
                               glmnet1=fit.glmnet1))
# regularized regression address the problem of overfitting and multicollinearity

# Ensemble - bagging, boosting & stacking
# random forest tune 
fit.rf <- train(x=x_train,y=y_train,
                method="rf",
                metric=metric,
                importance=TRUE,
                trControl=my_train)
# tune hyperparameter - the number of random variables used in each tree
tunerf <- expand.grid(mtry=1:50)
fit.rf1 <- train(x=x_train,y=y_train,
                 method="rf",
                 metric=metric,
                 trControl=my_train,
                 tuneGrid=tunerf,
                 importance=TRUE) 

# xgbtree
fit.xgbTree <- train(x=x_train,y=y_train,
                     method="xgbTree",
                     metric=metric,
                     tuneLength=5,
                     trControl=my_train)

# MARS  
fit.mars <- train(x=x_train,
                  y=y_train,
                  method="earth",
                  metric="Accuracy",
                  trControl=my_train,
                  tuneLength=5)
tunemars <- expand.grid(degree=1:3, nprune=seq(2,100,length.out = 10) %>% floor())
fit.mars1 <- train(x=x_train,
                   y=y_train,
                   method="earth",
                   metric="Accuracy",
                   trControl=my_train,
                   tuneGrid=tunemars)
models <- resamples(list(mars=fit.mars1,
                         glmnet=fit.glmnet1,rf=fit.rf1,xgbm=fit.xgbTree))
summary(models)
# the models with strongest predictive power are ensemble models, random forest - bagging & extreme gradient boosting - boosting
# random forest: 1. adds additional randomness to the model 2. reduces variance
# extreme gradient boosting: 1. optimizes total loss 2. allows us to build multiple models each of which learns to fix the prediction errors of a prior model in the chain   

# stacking
algorithmlist1 <- c("xgbTree","glmnet","rf")
model1 <- caretList(x=x_train,y=y_train,methodList = algorithmlist1,metric=metric,trControl=my_train,
                    tuneLength=5)

algorithmlist2 <- c("rf","glmnet")
model2 <- caretList(x=x_train,y=y_train,methodList = algorithmlist2,metric=metric,trControl=my_train,
                    tuneLength=5)

algorithmlist3 <- c("xgbTree","glmnet")
model3 <- caretList(x=x_train,y=y_train,methodList = algorithmlist3,metric=metric,trControl=my_train,
                    tuneLength=5)

my_train1 <- trainControl(method = "repeatedcv", # cross validation
                          number = 10,# 10 folds
                          repeats = 5, # repeat 5 times
                          savePredictions = "final",
                          classProbs = TRUE,
                          allowParallel = TRUE,
                          search = "grid")
ensemble1 <- caretEnsemble(model1,metric=metric,tuneLength=5,trControl=my_train1)
summary(ensemble1)
ensemble2 <- caretEnsemble(model2,metric=metric,tuneLength=5,trControl=my_train1)
summary(ensemble2)
ensemble3 <- caretEnsemble(model3, metric=metric, tuneLength=5,trControl=my_train1)
summary(ensemble3)

# model results comparison
modelresult <- data.frame(
  rf=max(fit.rf1$results$Accuracy),
  glmnet=max(fit.glmnet1$results$Accuracy),
  xgboost=max(fit.xgbTree$results$Accuracy),
  mars=max(fit.mars1$results$Accuracy),
  rf_glmnet_xgb=ensemble1$ens_model$results$Accuracy,
  rf_glmnet=ensemble2$ens_model$results$Accuracy,
  xgb_glmnet=ensemble3$ens_model$results$Accuracy
)
modelresult 

# test prediction for accuracy
predxgb <- predict(fit.xgbTree, x_test,type="raw") 
predglmnet <- predict(fit.glmnet1, x_test,type="raw")
predrf <- predict(fit.rf1,x_test,type="raw")
predmars <- predict(fit.mars1, x_test, type="raw")
predensemble1 <- predict(ensemble1,x_test,type="raw")
predensemble2 <- predict(ensemble2,x_test,type="raw")
predensemble3 <- predict(ensemble3,x_test,type="raw")
predaccuracy <- data.frame(glmnet=mean(predglmnet==y_test),
                           rf=mean(predrf==y_test),
                           mars=mean(predmars==y_test),
                           rf_xgb_glmnet=1-mean(predensemble1==y_test),
                           rf_glmnet=1-mean(predensemble2==y_test),
                           xgb_glmnet=1-mean(predensemble3==y_test),
                           xgb=mean(predxgb==y_test))
# Accuracy table 
predaccuracy
# the best performer is stacked model: extreme gradient boosting and regularized regression 

# Accuracy table
accuracy_table <- as.data.frame(as.table(t(predaccuracy)))
accuracy_table$Var2 <- NULL
colnames(accuracy_table) <- c("model","accuracy")
accuracy_table$model <- as.factor(accuracy_table$model)
accuracy_table %>% ggplot(aes(x=model,y=accuracy, fill=model))+geom_bar(stat = "identity")