setwd("C:/Users/Yang/Sumo/1034to1033/StockportArea_A6_Bluetooth_Data/TrainingData_Jan_to_Jun_2014_and_2015/SpeedBetweenSites/")
temp=list.files("C:/Users/Yang/Sumo/1034to1033/StockportArea_A6_Bluetooth_Data/TrainingData_Jan_to_Jun_2014_and_2015/SpeedBetweenSites/",pattern="*.csv")
myfiles=as.data.frame(lapply(temp, (read.csv)))


myfiles<-myfiles[-c(1:8), ]

T1<-t(myfiles)
T1<-as.data.frame(T1)
T1<-cbind(rownames(T1),T1)
rownames(T1)<-NULL

T1<-T1[,-c(27:29)]

Finals<-select(T1,-`rownames(T1)`)
Final<-separate(Finals, `9`, into =c("Weekday","Day","Month","Year"))


#Omit missing values#
Final<-na.omit(Final)

sapply(Final,mode)

#for loop to convert the row values to numeric values#
for(i in c(5:28:ncol(Final)))
{Final[,i]<-as.numeric(as.character(Final[,i]))}
sapply(Final, mode)
Final<-na.omit(Final)


#Normalise data#
maxs<- apply(Final[,5:28],2,max)
mins<- apply(Final[,5:28],2,min)
scaled.data<-as.data.frame(scale(Final[,5:28],center = mins, scale = maxs-mins))


#Split into train and test datasets#

nrow<-nrow(Final)
train<-nrow*0.75
test<-nrow*0.25
train_<-scaled.data[1:train,]

names(train_)[names(train_)=="10"]<-"V1"
names(train_)[names(train_)=="11"]<-"V2"
names(train_)[names(train_)=="12"]<-"V3"
names(train_)[names(train_)=="13"]<-"V4"
names(train_)[names(train_)=="14"]<-"V5"
names(train_)[names(train_)=="15"]<-"V6"
names(train_)[names(train_)=="16"]<-"V7"
names(train_)[names(train_)=="17"]<-"V8"
names(train_)[names(train_)=="18"]<-"V9"
names(train_)[names(train_)=="19"]<-"V10"
names(train_)[names(train_)=="20"]<-"V11"
names(train_)[names(train_)=="21"]<-"V12"
names(train_)[names(train_)=="22"]<-"V13"
names(train_)[names(train_)=="23"]<-"V14"
names(train_)[names(train_)=="24"]<-"V15"
names(train_)[names(train_)=="25"]<-"V16"
names(train_)[names(train_)=="26"]<-"V17"
names(train_)[names(train_)=="27"]<-"V18"
names(train_)[names(train_)=="28"]<-"V19"
names(train_)[names(train_)=="29"]<-"V20"
names(train_)[names(train_)=="30"]<-"V21"
names(train_)[names(train_)=="31"]<-"V22"
names(train_)[names(train_)=="32"]<-"V23"
names(train_)[names(train_)=="33"]<-"V24"

names(Final)[names(Final)=="10"]<-"V1"
names(Final)[names(Final)=="11"]<-"V2"
names(Final)[names(Final)=="12"]<-"V3"
names(Final)[names(Final)=="13"]<-"V4"
names(Final)[names(Final)=="14"]<-"V5"
names(Final)[names(Final)=="15"]<-"V6"
names(Final)[names(Final)=="16"]<-"V7"
names(Final)[names(Final)=="17"]<-"V8"
names(Final)[names(Final)=="18"]<-"V9"
names(Final)[names(Final)=="19"]<-"V10"
names(Final)[names(Final)=="20"]<-"V11"
names(Final)[names(Final)=="21"]<-"V12"
names(Final)[names(Final)=="22"]<-"V13"
names(Final)[names(Final)=="23"]<-"V14"
names(Final)[names(Final)=="24"]<-"V15"
names(Final)[names(Final)=="25"]<-"V16"
names(Final)[names(Final)=="26"]<-"V17"
names(Final)[names(Final)=="27"]<-"V18"
names(Final)[names(Final)=="28"]<-"V19"
names(Final)[names(Final)=="29"]<-"V20"
names(Final)[names(Final)=="30"]<-"V21"
names(Final)[names(Final)=="31"]<-"V22"
names(Final)[names(Final)=="32"]<-"V23"
names(Final)[names(Final)=="33"]<-"V24"

test_<-scaled.data[(train+1):nrow,]
test<-Final[(train+1):nrow,]

names(test_)[names(test_)=="10"]<-"V1"
names(test_)[names(test_)=="11"]<-"V2"
names(test_)[names(test_)=="12"]<-"V3"
names(test_)[names(test_)=="13"]<-"V4"
names(test_)[names(test_)=="14"]<-"V5"
names(test_)[names(test_)=="15"]<-"V6"
names(test_)[names(test_)=="16"]<-"V7"
names(test_)[names(test_)=="17"]<-"V8"
names(test_)[names(test_)=="18"]<-"V9"
names(test_)[names(test_)=="19"]<-"V10"
names(test_)[names(test_)=="20"]<-"V11"
names(test_)[names(test_)=="21"]<-"V12"
names(test_)[names(test_)=="22"]<-"V13"
names(test_)[names(test_)=="23"]<-"V14"
names(test_)[names(test_)=="24"]<-"V15"
names(test_)[names(test_)=="25"]<-"V16"
names(test_)[names(test_)=="26"]<-"V17"
names(test_)[names(test_)=="27"]<-"V18"
names(test_)[names(test_)=="28"]<-"V19"
names(test_)[names(test_)=="29"]<-"V20"
names(test_)[names(test_)=="30"]<-"V21"
names(test_)[names(test_)=="31"]<-"V22"
names(test_)[names(test_)=="32"]<-"V23"
names(test_)[names(test_)=="33"]<-"V24"

sapply(train_, mode)

#using Neuralnet#
n<-names(train_)
sapply(n,mode)
f<-as.formula(paste("V18~",paste(n[!n %in% "V18"],collapse = "+")))
print(f)
nn<-neuralnet(f,data = train_,hidden=c(5,3),linear.output = TRUE)
plot(nn)

pr.nn<-compute(nn,test_[,2:24])


pr.nn_ <- pr.nn$net.result*(max(Final$V18)-min(Final$V18))+min(Final$V18)
test.r <- (test_$V18)*(max(Final$V18)-min(Final$V18))+min(Final$V18)

MSE.nn <- sum((test.r - pr.nn_)^2)/nrow(test_)


plot(test.r,pr.nn_,col='red',main='Real vs predicted Speed',pch=18,cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend='NN',pch=18,col='red', bty='n')


Test_set<-select(test, V18)
result<-as.data.frame(pr.nn_)
result<-round(result, digits = 1)

Compare<-bind_cols(result,Test_set)
Compare<-round(Compare,digits=1)
Compare<-mutate(Compare,Error=V18-V1)
names(Compare)[names(Compare)=="V1"]<-"Predicted Speed"
names(Compare)[names(Compare)=="V18"]<-"Acutal Speed"



setwd("C:/Users/Yang/Sumo/1034to1033/StockportArea_A6_Bluetooth_Data/TrainingData_Jan_to_Jun_2014_and_2015/DataOfIndividualSites/")
Vol1<-read.csv("C:/Users/Yang/Sumo/1034to1033/StockportArea_A6_Bluetooth_Data/TrainingData_Jan_to_Jun_2014_and_2015/DataOfIndividualSites/pvr_2014-01-01_181d_site_1034_volume.csv")
Vol2<-read.csv("C:/Users/Yang/Sumo/1034to1033/StockportArea_A6_Bluetooth_Data/TrainingData_Jan_to_Jun_2014_and_2015/DataOfIndividualSites/pvr_2015-01-01_181d_site_1034_volume.csv")
Vol1<-select(Vol1,c(Sdate,Volume))
Vol1<-filter(Vol1,Volume>0)
Vol1<-separate(Vol1,Sdate, into = c("Day","Month","Year","Time"))
Vol2<-select(Vol2,c(Sdate,Volume))
Vol2<-filter(Vol2,Volume>0)
Vol2<-separate(Vol2,Sdate, into = c("Day","Month","Year","Time"))

FinalD<-rbind(Vol1, Vol2)
FinalD<-t(FinalD)

colnames(FinalD)=FinalD[4, ]
FinalD<-FinalD[-4, ]


# tempD=list.files("C:/Users/Yang/Sumo/1034to1033/StockportArea_A6_Bluetooth_Data/TrainingData_Jan_to_Jun_2014_and_2015/DataOfIndividualSites/",pattern="*.csv")
# myfilesD=as.data.frame(lapply(tempD, (read.csv)))
# 
# 
# myfilesD<-select(myfilesD, Sdate, `Volume`)
# myfilesD<-separate(myfilesD,Sdate, into =c("Day","Month","Year","Time"))
# myfilesD<-unite(myfilesD,"Date",Day,Month,Year,sep = "-")
# FinalD<-filter(myfilesD, Volume>0)
# FinalD<-spread(FinalD,Time,Volume)
# FinalD<-separate(FinalD,Date, into =c("Day","Month","Year"))

#Omit missing values#
# FinalD<-na.omit(FinalD)

sapply(FinalD, mode)
for(i in c(1:ncol(FinalD)))
{FinalD[,i]<-as.numeric(as.character(FinalD[,i]))}
sapply(FinalD, mode)
FinalD<-na.omit(FinalD)
FinalD<-rename(FinalD,c("00"="V1","01"="V2","02"="V3","03"="V4","04"="V5","05"="V6","06"="V7","07"="V8","08"="V9","09"="V10","10"="V11","11"="V12","12"="V13","13"="V14","14"="V15","15"="V16","16"="V17","17"="V18","18"="V19","19"="V20","20"="V21","21"="V22","22"="V23","23"="V24"))


#Normalise data#
maxs<- apply(FinalD[,4:27],2,max)
mins<- apply(FinalD[,4:27],2,min)
scaled.dataD<-as.data.frame(scale(FinalD[,4:27],center = mins, scale = maxs-mins))


#Split into train and test datasets#
nrowD<-nrow(FinalD)
trainD<-nrowD*0.75
testD<-nrowD*0.25
trainD_<-scaled.dataD[1:trainD,]
testD_<-scaled.dataD[(trainD+1):nrowD,]
testD<-FinalD[(trainD+1):nrowD,]

sapply(trainD_, mode)


#using Neuralnet#
nD<-names(trainD_)
sapply(nD,mode)
fD<-as.formula(paste("V18~",paste(nD[!nD %in% "V18"],collapse = "+")))
print(fD)
nnD<-neuralnet(fD,data = trainD_,hidden=c(5,3),linear.output = TRUE)
plot(nnD)

pr.nnD<-compute(nnD,testD_[,2:24])


pr.nnD_ <- pr.nnD$net.result*(max(FinalD$V18)-min(FinalD$V18))+min(FinalD$V18)
test.rD <- (testD_$V18)*(max(FinalD$V18)-min(FinalD$V18))+min(FinalD$V18)

MSE.nnD <- sum((test.rD - pr.nnD_)^2)/nrow(testD_)


plot(test.rD,pr.nnD_,col='red',main='Real vs predicted NN Density',pch=18,cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend='NN',pch=18,col='red', bty='n')


Test_setD<-select(testD, V18)
HmmmmD<-as.data.frame(pr.nnD_)
HmmmmD<-round(HmmmmD,digits = 0)
CompareD<-bind_cols(HmmmmD,Test_setD)
CompareD<-round(CompareD,digits=0)
CompareD<-mutate(CompareD,Error=V18-V1)
CompareD<-rename(CompareD,c("V1"="Predicted Density","V18"="Actual Density"))


#Final_output<-bind_cols(CompareD,Compare)
# load data


Date<-select(test,Weekday,Day,Month,Year)
DateD<-select(testD,Day,Month,Year)
Date<-unite(Date,"Date",Weekday,Day,Month,Year,sep='/')
DateD<-unite(DateD,"Date",Day,Month,Year,sep='/')
Compare<-bind_cols(Compare,Date)
CompareD<-bind_cols(HmmmmD,Test_setD)
CompareD<-bind_cols(CompareD,DateD)




CompareB<-bind_cols(test,final)
CompareB<-bind_cols(CompareB,Compare)
CompareB<-bind_cols(CompareB,HmmmmD)

Compare<-select(Compare,-Error)

write.table(Compare, "/users/nakessien/downloads/dayta/file.txt", sep = " ")

write.table(HmmmmD, "/users/nakessien/downloads/dayta/fileD.txt", sep = " ")


data("HouseVotes84")
#barplots for specific issue
plot(as.factor(HouseVotes84[,2]))
title(main="Votes cast for issue", xlab="vote", ylab="# reps")
#by party
plot(as.factor(HouseVotes84[HouseVotes84$Class=='republican',2]))
title(main="Republican votes cast for issue 1", xlab="vote", ylab="# reps")
plot(as.factor(HouseVotes84[HouseVotes84$Class=='democrat',2]))
title(main="Democrat votes cast for issue 1", xlab="vote", ylab="# reps")
#Functions needed for imputation
#function to return number of NAs by vote and class (democrat or republican)
na_by_col_class <- function (col,cls){return(sum(is.na(HouseVotes84[,col]) & HouseVotes84$Class==cls))}
#function to compute the conditional probability that a member of a party will cast
#a 'yes' vote for a particular issue. The probability is based on all members of the party who
#actually cast a vote on the issue (ignores NAs).
p_y_col_class <- function(col,cls){
  sum_y<-sum(HouseVotes84[,col]=='y' & HouseVotes84$Class==cls,na.rm = TRUE)
  sum_n<-sum(HouseVotes84[,col]=='n' & HouseVotes84$Class==cls,na.rm = TRUE)
  return(sum_y/(sum_y+sum_n))}
#impute missing values.
for (i in 2:ncol(HouseVotes84)) {
  if(sum(is.na(HouseVotes84[,i])>0)) {
    c1 <- which(is.na(HouseVotes84[,i])& HouseVotes84$Class=='democrat',arr.ind = TRUE)
    c2 <- which(is.na(HouseVotes84[,i])& HouseVotes84$Class=='republican',arr.ind =
                  TRUE)
    HouseVotes84[c1,i] <-
      ifelse(runif(na_by_col_class(i,'democrat'))<p_y_col_class(i,'democrat'),'y','n')
    HouseVotes84[c2,i] <-
      ifelse(runif(na_by_col_class(i,'republican'))<p_y_col_class(i,'republican'),'y','n'
      )}
}
#divide into test and training sets
#create new col "train" and assign 1 or 0 in 80/20 proportion via random uniform
dist
HouseVotes84[,"train"] <- ifelse(runif(nrow(HouseVotes84))<0.80,1,0)
#get col number of train / test indicator column (needed later)
trainColNum <- grep("train",names(HouseVotes84))
#separate training and test sets and remove training column before modeling
trainHouseVotes84 <- HouseVotes84[HouseVotes84$train==1,-trainColNum]
testHouseVotes84 <- HouseVotes84[HouseVotes84$train==0,-trainColNum]
#train model
nb_model <- naiveBayes(Class~.,data = trainHouseVotes84)
nb_model
summary(nb_model)
str(nb_model)
#...and the moment of reckoning
nb_test_predict <- predict(nb_model,testHouseVotes84[,-1])
#confusion matrix
table(pred=nb_test_predict,true=testHouseVotes84$Class)


#fraction of correct predictions
mean(nb_test_predict==testHouseVotes84$Class)
#function to create, run and record model results
nb_multiple_runs <- function(train_fraction,n){
  fraction_correct <- rep(NA,n)
  for (i in 1:n){
    HouseVotes84[,"train"] <- ifelse(runif(nrow(HouseVotes84))<train_fraction,1,0)
    trainColNum <- grep("train",names(HouseVotes84))
    trainHouseVotes84 <- HouseVotes84[HouseVotes84$train==1,-trainColNum]
    testHouseVotes84 <- HouseVotes84[HouseVotes84$train==0,-trainColNum]
    nb_model <- naiveBayes(Class~.,data = trainHouseVotes84)
    nb_test_predict <- predict(nb_model,testHouseVotes84[,-1])
    fraction_correct[i] <- mean(nb_test_predict==testHouseVotes84$Class)
  }
  return(fraction_correct)
}
#20 runs, 80% of data randomly selected for training set in each run
fraction_correct_predictions <- nb_multiple_runs(0.8,20)
fraction_correct_predictions
#summary of results
summary(fraction_correct_predictions)
#standard deviation
sd(fraction_correct_predictions)