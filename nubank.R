setwd("~/Desktop/puzzle/")

library(ggplot2)
library(reshape)
library(randomForest)
library(mice)
library(VIM)

train = read.csv("puzzle_train_dataset.csv", na.strings=c(""," ", "NA"))
train <- train[complete.cases(train[,2]),]
train$default <- ifelse(train["default"]=="True", 1, 0)
train$last_payment <- as.Date(train$last_payment)
train$end_last_loan <- as.Date(train$end_last_loan)

# Find columns that have many missing values and remove them
mice_plot <- aggr(train, col=c('navyblue','yellow'),
                  numbers=TRUE, sortVars=TRUE,
                  labels=names(train), cex.axis=.7,
                  gap=3, ylab=c("Missing data","Pattern"))
train$ok_since <- NULL
train$credit_limit <- NULL
train$sign <- NULL
train$n_issues <- NULL
train$job_name <- NULL

# remove columns with too many categorical factors
train$zip <- NULL
train$reason <- NULL

# Remove rows with NAs (look into imputing)
train2 <- train[complete.cases(train),]
train2.noID <- train2
train2.noID$ids <- NULL

# keep only numeric columns (look into making dummies)
numeric_cols <- sapply(train2, is.numeric)
train2.numeric <- train2[, numeric_cols]

# look at difference between default and non_default
#train.lng <- melt(train.numeric, id="default")
#ggplot(aes(x=value, group=default, colour=factor(default)), data=train.lng) +
#  geom_histogram() +
#  facet_wrap(~variable, scales="free")

# imute data
#mice_plot <- aggr(train2, col=c('navyblue','yellow'),
#                  numbers=TRUE, sortVars=TRUE,
#                  labels=names(train2), cex.axis=.7,
#                  gap=3, ylab=c("Missing data","Pattern"))
#imputed_Data <- mice(train.numeric, m=5, maxit = 50, method = 'pmm', seed = 500)

# make train and test subsets (look into cross validation)
idx <- runif(nrow(train.numeric)) > 0.8
training <- train.numeric[idx==FALSE,]
testing <- train.numeric[idx==TRUE,]


rf <- randomForest(data=train2.noID, factor(default) ~ .,
                   type="classification", importance=TRUE, na.action = na.omit)
probs <- predict(rf, type="prob")[,2] #don't need to break into train/test?
probs <- ifelse(probs>0.5, 1, 0)
