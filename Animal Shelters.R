# INSTALL AND READ IN LIBRARIES -------------------------------------------
install.packages("caTools")
install.packages("foriegn")
install.packages("nnet")
install.packages("stargazer")
install.packages("MLmetrics")
install.packages("xgboost")
install.packages("data.table")
install.packages("mlr")


library(caTools)
library(foreign)
library(nnet)
library(stargazer)
library(MLmetrics)
library(ModelMetrics)
library(xgboost)
library(data.table)
library(mlr)
library(dplyr)
library(caret) 
library(e1071)





# READ IN AND ORGANIZE TRAINING DATA ------------------------------------------

setwd("~/Desktop/Learning Data Science/Animal Shelters") #Setting working directory

animalfulltrain <- read.csv("AnimalTrain.csv") #Read in training data
head(animalfulltrain) #View data
nrow(animalfulltrain)


set.seed(101) #Set seed for reproducibility
sample <- sample.split(animalfulltrain$AnimalID, SplitRatio = .20) #Get random sample, 20% of data
animaltest <- subset(animalfulltrain, sample == TRUE) #Create test dataset
animaltrain <- subset(animalfulltrain, sample == FALSE) #Create training dataset

nrow(animaltest)
nrow(animaltrain)

catstest <- subset(animaltest, AnimalType == "Cat") #Create Cat Subsets
catstrain <- subset(animaltrain, AnimalType == "Cat")
catstest$BreedClean <- factor(catstest$BreedClean)
catstrain$BreedClean <- factor(catstrain$BreedClean)
catstest$Color <- factor(catstest$Color)
catstrain$Color <- factor(catstrain$Color)

write.csv(catstest, "cats.test.csv") #Creating training and testing datasets for cats
write.csv(catstrain, "cats.train.csv")


dogstest <- subset(animaltest, AnimalType == "Dog") #Create training and testing datasets for dogs
dogstrain <- subset(animaltrain, AnimalType == "Dog")
dogstest$BreedClean <- factor(dogstest$BreedClean)
dogstrain$BreedClean <- factor(dogstrain$BreedClean)
dogstest$Color <- factor(dogstest$Color)
dogstrain$Color <- factor(dogstrain$Color)





# DATA MINING -------------------------------------------------------------

#Graph Outcomes
catoutcomes1 <- prop.table(table(catstrain$Outcome))*100 #Table of cat outcomes
dogoutcomes1 <- prop.table(table(dogstrain$Outcome))*100 #Table of dog outcomes


par(mfrow=c(2,1), mar=c(3.5,4,3.5,2)+0.1, oma=c(0,0,0,0)+0.1 ) #Plot outcomes by animal

catsplot1 <- barplot(catoutcomes1, main = "Cats", ylab = "Percentage (%)", ylim = c(0, 60), space = c(.25, .25, 1, 1, .25), col  = c("limegreen", "limegreen", "orange", "firebrick2", "firebrick2"), names.arg = c("Reunited", "Adopted", "Transferred", "Died", "Euthanized"))
abline(h=0, lwd = 2)
text(x = catsplot1, y = catoutcomes1, label = paste0(round(catoutcomes1, 0), "%"), pos = 3)

dogsplot1 <- barplot(dogoutcomes1, main = "Dogs", ylab = "Percentage (%)", ylim = c(0, 60), space = c(.25, .25, 1, 1, .25), col  = c("limegreen", "limegreen", "orange", "firebrick2", "firebrick2"), names.arg = c("Reunited", "Adopted", "Transferred", "Died", "Euthanized"))
abline(h=0, lwd = 2)
text(x = dogsplot1, y = dogoutcomes1, label = paste0(round(dogoutcomes1, 0), "%"), pos = 3)

table(catstrain$Color)


#Graph Outcomes by Name Status
catoutcomes2 <- prop.table(table(catstrain$NameGiven, catstrain$Outcome), margin = 2)*100 #Table of cat outcomes by name
catoutcomes2 <- rbind(catoutcomes2[2,], catoutcomes2[1,])

dogoutcomes2 <- prop.table(table(dogstrain$NameGiven, dogstrain$Outcome), margin = 2)*100 #Table of dog outcomes by name
dogoutcomes2 <- rbind(dogoutcomes2[2,], dogoutcomes2[1,])


par(mfrow=c(2,1), mar=c(3.5,4,3.5,2)+0.1, oma=c(0,0,0,0)+0.1 ) #Plot outcomes by animal

catsplot2 <- barplot(catoutcomes2, main = "Cats", ylab = "Percentage (%)", ylim = c(0, 105), space = c(.5, .5, 1.5, 1.5, .5), col  = c("limegreen", "red"), names.arg = c("Reunited", "Adopted", "Transferred", "Died", "Euthanized"))
abline(h=0, lwd = 2)
text(x = catsplot2, y = (((catoutcomes2[1,])/2) - 10), label = paste0(round(catoutcomes2[1,], 0), "%\nNamed"), pos = 3, cex = .8)

dogsplot2 <- barplot(dogoutcomes2, main = "Dogs", ylab = "Percentage (%)", ylim = c(0, 105), space = c(.5, .5, 1.5, 1.5, .5), col  = c("limegreen", "red"), names.arg = c("Reunited", "Adopted", "Transferred", "Died", "Euthanized"))
abline(h=0, lwd = 2)
text(x = dogsplot2, y = (((dogoutcomes2[1,])/2) - 10), label = paste0(round(dogoutcomes2[1,], 0), "%\nNamed"), pos = 3, cex = .8)




# LOG LOSS FUNCTION -------------------------------------------------------

#Write function to evaluate the accuracy of predictive model 

LogLoss <- function(actual, predicted, eps=1e-15) {
  
  predicted[predicted < eps] <- eps;
  predicted[predicted > 1 - eps] <- 1 - eps;
  -1/nrow(actual)*(sum(actual*log(predicted)))
}








# MACHINE LEARNING USING XGBOOST FOR CATS --------------------------------------------

cats.train <- subset(catstrain, select = c("OutcomeType", "NameGiven", "SpayNueter", "Female", "BreedCollapse2",  "Juvenile", "Senior", "AgeYears", "ColorCollapse", "Missing.Info"))
cats.test <- subset(catstest, select = c("OutcomeType", "NameGiven", "SpayNueter", "Female", "BreedCollapse2", "Juvenile", "Senior", "AgeYears", "ColorCollapse", "Missing.Info"))


cats.train <- setDT(cats.train) #Convert data frames to data tables
cats.test <- setDT(cats.test)

table(is.na(cats.train))
sapply(cats.train, function(x) sum(is.na(x))/length(x))*100

table(is.na(cats.test))
sapply(cats.test, function(x) sum(is.na(x))/length(x))*100


char_col <- colnames(cats.train)[ sapply (cats.test,is.character)]
for(i in char_col) set(cats.train,j=i,value = str_trim(cats.test[[i]],side = "left"))
for(i in char_col) set(cats.test,j=i,value = str_trim(cats.test[[i]],side = "left"))

cats.train[is.na(cats.train)] <- "Missing" 
cats.test[is.na(cats.test)] <- "Missing"

labels <- cats.train$OutcomeType
ts_label <- cats.test$OutcomeType
numberOfClasses <- length(unique(cats.train$OutcomeType))

new_tr <- model.matrix(~.+0,data = cats.train[,-c("OutcomeType"),with=F]) 
new_ts <- model.matrix(~.+0,data = cats.test[,-c("OutcomeType"),with=F])

labels <- as.numeric(labels)-1
ts_label <- as.numeric(ts_label)-1

dtrain <- xgb.DMatrix(data = new_tr,label = labels) 
dtest <- xgb.DMatrix(data = new_ts,label=ts_label)

params <- list(booster = "gbtree", objective = "multi:softprob", eval_metric = "mlogloss", num_class = numberOfClasses)

nround    <- 50 # number of XGBoost rounds
cv.nfold  <- 5

# Fit cv.nfold * cv.nround XGB models and save OOF predictions
cv_model <- xgb.cv(params = params,
                   data = dtrain, 
                   nrounds = nround,
                   nfold = cv.nfold,
                   verbose = FALSE,
                   prediction = TRUE)


bst_model <- xgb.train(params = params,
                       data = dtrain,
                       nrounds = nround)

test_pred <- predict(bst_model, newdata = dtest)

test_prediction <- matrix(test_pred, nrow = numberOfClasses,
                          ncol=length(test_pred)/numberOfClasses) %>%
  t() %>%
  data.frame() %>%
  mutate(label = ts_label + 1,
         max_prob = max.col(., "last"))


confusionMatrix(factor(test_prediction$label),
                factor(test_prediction$max_prob),
                mode = "everything")

union <- union(test_prediction$label, test_prediction$max_prob)
tableunion <- table(factor(test_prediction$label, union), factor(test_prediction$max_prob, union))

confusionMatrix(tableunion)

names <-  colnames(cats.train[,-1])
importance_matrix = xgb.importance(feature_names = names, model = bst_model)
importance_matrix

catpred <- subset(test_prediction, select = c("X1", "X2", "X3", "X4", "X5"))
names(catpred) <- c(1, 2, 3, 4, 5)

mlogLoss(test_prediction$label, catpred)

     








# MACHINE LEARNING USING XGBOOST FOR DOGS --------------------------------------------

dogs.train <- subset(dogstrain, select = c("OutcomeType", "NameGiven", "Purebred", "Ban", "AgeYears", "SpayNueter", "Missing.Info", "Female", "PopularityColl", "BreedClean"))
dogs.test <- subset(dogstest, select = c("OutcomeType", "NameGiven", "Purebred", "Ban", "AgeYears", "SpayNueter", "Missing.Info", "Female", "PopularityColl", "BreedClean"))


dogs.train <- subset(dogstrain, select = c("OutcomeType", "NameGiven", "Purebred", "Ban", "AgeYears", "SpayNueter", "Missing.Info", "PopularityColl", "BreedClean"))
dogs.test <- subset(dogstest, select = c("OutcomeType", "NameGiven", "Purebred", "Ban", "AgeYears", "SpayNueter", "Missing.Info", "PopularityColl", "BreedClean"))





dogs.train <- setDT(dogs.train) #Convert data frames to data tables
dogs.test <- setDT(dogs.test)

table(is.na(dogs.train))
sapply(dogs.train, function(x) sum(is.na(x))/length(x))*100

table(is.na(dogs.test))
sapply(dogs.test, function(x) sum(is.na(x))/length(x))*100


char_col <- colnames(dogs.train)[ sapply (dogs.test,is.character)]
for(i in char_col) set(dogs.train,j=i,value = str_trim(dogs.test[[i]],side = "left"))
for(i in char_col) set(dogs.test,j=i,value = str_trim(dogs.test[[i]],side = "left"))

dogs.train[is.na(dogs.train)] <- "Missing" 
dogs.test[is.na(dogs.test)] <- "Missing"

labels <- dogs.train$OutcomeType
ts_label <- dogs.test$OutcomeType
numberOfClasses <- length(unique(dogs.train$OutcomeType))

new_tr <- model.matrix(~.+0,data = dogs.train[,-c("OutcomeType"),with=F]) 
new_ts <- model.matrix(~.+0,data = dogs.test[,-c("OutcomeType"),with=F])

labels <- as.numeric(labels)-1
ts_label <- as.numeric(ts_label)-1

dtrain <- xgb.DMatrix(data = new_tr,label = labels) 
dtest <- xgb.DMatrix(data = new_ts,label=ts_label)

params <- list(booster = "gbtree", objective = "multi:softprob", eval_metric = "mlogloss", num_class = numberOfClasses, eta = .1)

nround    <- 100 # number of XGBoost rounds
cv.nfold  <- 5

# Fit cv.nfold * cv.nround XGB models and save OOF predictions
cv_model <- xgb.cv(params = params,
                   data = dtrain, 
                   nrounds = nround,
                   nfold = cv.nfold,
                   verbose = FALSE,
                   prediction = TRUE)


bst_model <- xgb.train(params = params,
                       data = dtrain,
                       nrounds = nround)

test_pred <- predict(bst_model, newdata = dtest)

test_prediction <- matrix(test_pred, nrow = numberOfClasses,
                          ncol=length(test_pred)/numberOfClasses) %>%
  t() %>%
  data.frame() %>%
  mutate(label = ts_label + 1,
         max_prob = max.col(., "last"))


confusionMatrix(factor(test_prediction$label),
                factor(test_prediction$max_prob),
                mode = "everything")

union <- union(test_prediction$label, test_prediction$max_prob)
tableunion <- table(factor(test_prediction$label, union), factor(test_prediction$max_prob, union))

confusionMatrix(tableunion)

names <-  colnames(dogs.train[,-1])
importance_matrix = xgb.importance(feature_names = names, model = bst_model)
importance_matrix

dogpred <- subset(test_prediction, select = c("X1", "X2", "X3", "X4", "X5"))
names(catpred) <- c(1, 2, 3, 4, 5)

mlogLoss(test_prediction$label, dogpred) #STILL NEEDS WORK

