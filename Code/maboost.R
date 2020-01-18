library(readr)
library(forecast)
library(randomForest)
library(maboost)
library(OneR)
spss <- read_csv("OneDrive - Swansea University/Personal/PhD/spss_new.csv")
spss <- spss[sample(nrow(spss)),]
train <- spss[1:698,]
test <- spss[698:nrow(spss),]


train_x <- train[,c(5,7,10,12,14,16,18)]
test_x <- test[,c(5,7,10,12,14,16,18)]
train <- train[,c(5,7,10,12,14,16,18,19)]
test <- test[,c(5,7,10,12,14,16,18,19)]
train_y <- train[,c(8)]
test_y <- test[,c(8)]
test_y <- as.matrix(test_y)
test_y <- as.factor(test_y)

train_y <- as.matrix(train_y)
train_y <- as.factor(train_y)

colnames(train)[2] <- "Urban"
colnames(train)[3] <- "Univariate"
colnames(train)[4] <- "DCM"
colnames(train)[5] <- "Large"
colnames(train)[7] <- "Realtime"
colnames(train)[8] <- "DAM"

colnames(test)[2] <- "Urban"
colnames(test)[3] <- "Univariate"
colnames(test)[4] <- "DCM"
colnames(test)[5] <- "Large"
colnames(test)[7] <- "Realtime"
colnames(test)[8] <- "DAM"

train$DAM = factor(train$DAM) 
iris.rf <- randomForest(DAM ~ ., data=train, ntree=5000, keep.forest = TRUE,  importance=TRUE,
                        proximity=TRUE)
print(iris.rf)
## Look at variable importance:
round(importance(iris.rf), 2)

pr <- predict(iris.rf, test)

pr <- round(pr, 0)
gdis<-maboost(train$DAM~.,data=train,iter=7,nu=1
              ,breg="l2", type="sparse",bag.frac=0.6,random.feature=FALSE
              ,random.cost=FALSE, C50tree=FALSE, maxdepth=6,verbose=TRUE)

oneR <- OneR(train$DAM~., data=train)




model_suggest <- function(analysis, urban, univariate, dcm, large, ph, realtime)
{
  require(ggplot2)
  require(data.table)
  require(maboost)
  colnames(train)[2] <- "Urban"
  colnames(train)[3] <- "Univariate"
  colnames(train)[4] <- "DCM"
  colnames(train)[5] <- "Large"
  colnames(train)[7] <- "Realtime"
  colnames(train)[8] <- "DAM"
  gdis<-randomForest(DAM ~ ., data=train, ntree=500, keep.forest = TRUE,  importance=TRUE,
                     proximity=TRUE)
  new_data <- data.frame(Analysis=analysis, Urban=urban, Univariate=univariate, DCM=dcm, Large=large, PH=ph, Realtime=realtime)
  pred.model= predict(gdis,new_data,type="prob")
  new_d <- as.data.frame(t(pred.model))
  library(data.table)
  setDT(new_d, keep.rownames = TRUE)[]
  new_d <- new_d[order(new_d$rn, decreasing=T),]
  colnames(new_d)[1] <- "rn"
  colnames(new_d)[2] <- "V1"
  return(ggplot(data = new_d, aes(x=reorder(rn, -V1), y=V1)) +
           geom_bar(stat="identity") +
           geom_text(aes(label=round(V1, digits=4)), vjust=1.6, color="white", size=3.5)+
           theme_bw()+ labs(x="Model", y="Probability (Confidence Level)"))
           
}


model_suggest(1,1,0,2,1,60,0)














new_data <- data.frame(Analysis=1, Urban=0, Univariate=1, DCM=1, Large=1, PH=5, Realtime=0)
pred.gdis= predict(gdis,train,type="prob")
