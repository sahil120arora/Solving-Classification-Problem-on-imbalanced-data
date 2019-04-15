#install.packages

install.packages("randomForest")
install.packages("dplyr")
install.packages("caret")
install.packages("InformationValue")
install.packages("car")
install.packages("pscl")


#libraries used
library(randomForest)
library(dplyr)
library(caret)
library(InformationValue)
library(car)
library(pscl)




#import data
data <- read.csv("C:/Users/arorsahi/Desktop/Resume/Ironman/Analytics Exercise/IM.csv", header = T, na.strings=c(""))

#Check missing values
sapply(data,function(x) sum(is.na(x)))

#view first few rows
head(data)

#get basic descriptives
summary(data)

#inspect variable datatypes
str(data)


#outlier treatment
data <- subset(data, data$Age > 1 & data$Finish.Time < 999999 & data$Swim.Time < 999999 & data$Run.Time < 999999 & data$Bike.Time < 999999)

#Age buckets
n <- nrow(data)
age.group <- rep(NA, n)
age.group[data$Age <= 30] <- "prime"
age.group[data$Age > 30 & data$Age <=50] <- "seasoned"
age.group[data$Age > 50] <- "veteran"

data$Age2 <- as.factor(age.group)


#remove unecessary variable
data <- select(data, -Contact.Key, -Age)

#Convert int to factor
columnss <- c("Age2","Resp","Min_Year","club_aff","Finish.Rank","Swim.Rank","Bike.Rank","Run.Rank")
data[,columnss] <- lapply(data[,columnss] , as.factor)
str(data)

#scaling
trans <- c('R2016','R2015','R2014','Finish.Time','Swim.Time','Bike.Time','Run.Time','prior_races')
x <- preProcess(data[,trans],method = c("range"), thresh =0.9)
data <- predict(x,data)


#check class bias
table(data$Resp)
#we observe class bias

# Create Training Data 
input_ones <- data[which(data$Resp == 1), ]  # all 1's
input_zeros <- data[which(data$Resp == 0), ]  # all 0's
set.seed(100)  # for repeatability of samples
input_ones_training_rows <- sample(1:nrow(input_ones), 0.7*nrow(input_ones))  # 1's for training
input_zeros_training_rows <- sample(1:nrow(input_zeros), 0.7*nrow(input_ones))  # 0's for training. Pick as many 0's as 1's
training_ones <- input_ones[input_ones_training_rows, ]  
training_zeros <- input_zeros[input_zeros_training_rows, ]
trainingData <- rbind(training_ones, training_zeros)  # row bind the 1's and 0's 

# Create Test Data
test_ones <- input_ones[-input_ones_training_rows, ]
test_zeros <- input_zeros[-input_zeros_training_rows, ]
testData <- rbind(test_ones, test_zeros)  # row bind the 1's and 0's 


#model
logitmod <- glm(Resp ~ . , data=trainingData, family=binomial(link="logit"))
#logitmodel <- glm(Resp ~ R2016 + R2015 + R2014 + Age2 + Finish.Time +Swim.Time + Bike.Time + prior_races + Min_Year + club_aff + Finish.Rank + Swim.Rank + Bike.Rank + Run.Rank, data=trainingData, family=binomial(link="logit"))
summary(logitmod)
pR2(logitmod)
vif(logitmod)

#calculate probabilities
x <- summary(logitmod)$coefficients[,1]
cbind(coef_log_odds = x, coef_prob = exp(x) / (1 + exp(x)))


#Calculate accuracy on Test data
predicted2 <- plogis(predict(logitmod, testData))

optCutOff <- optimalCutoff(testData$Resp, predicted2)[1] 
optCutOff

e <- misClassError(testData$Resp, predicted2,threshold = optCutOff)
print(paste('Accuracy',1-e))

#Model Dignostics
confusionMatrix(testData$Resp, predicted2, threshold = optCutOff)
sensitivity(testData$Resp, predicted2, threshold = optCutOff)
specificity(testData$Resp, predicted2, threshold = optCutOff)
plotROC(testData$Resp, predicted2)
Concordance(testData$Resp, predicted2)



#Variable Importance
varImp(logitmod, scale = FALSE)


#Extra material
#For reference only
#random forest

model1 <- randomForest(Resp ~ . , data = trainingData, mtry= 6,importance = TRUE,type=)
model1
summary(model1)


predValid2 <- predict(model1, testData, type = "class")
# Checking classification accuracy
mean(predValid2 == testData$Resp)                    
table(predValid2,testData$Resp)


importance(model1)        
varImpPlot(model1)
varImpPlot(model1,  
           sort = T,
           n.var=10,
           main="Top 10 - Variable Importance")

var.imp = data.frame(importance(model1,  
                                type=2))
# make row names as columns
var.imp$Variables = row.names(var.imp)  
print(var.imp[order(var.imp$MeanDecreaseGini,decreasing = T),])
