setwd("/Users/nakessien/Downloads/New_data/5-min volume/")
temp=list.files("/Users/nakessien/Downloads/New_data/5-min volume//",pattern="*.csv")
myfiles=as.data.frame(lapply(temp, (read.csv)))

dim(myfiles)
myfiles<-myfiles[-c(1:8), ]

T1<-t(myfiles)
T1<-as.data.frame(T1)
T1<-cbind(rownames(T1),T1)
rownames(T1)<-NULL

T1<-T1[,-c(1:4)]

data_f <- T1[,-c(291:912)]
data_f <- data_f[,-c(1)]

test <- data_f[-grep("Workday", data_f$`13`),]
test <- test[-grep("7 Day", test$`13`),]
test <- test[-grep("Count", test$`13`),]
test <- test[-grep(":", test$`14`),]


#Create timeseries#
time_index <- seq(from = as.POSIXct("2013-01-01 00:00"), 
                  to = as.POSIXct("2018-10-31 23:55"), by = "mins")

Final<-separate(test, `13`, into =c("Year", "Month","Day"))



Final<-data_f

#Omit missing values#
Final<-na.omit(X5_minute_vol)


sapply(Final,mode)
Final <- Final[,-c(1)]


#Normalise data#
maxs<- apply(Final,2,max)
mins<- apply(Final,2,min)
scaled.data<-as.data.frame(scale(Final,center = mins, scale = maxs-mins))


#Split into train and test datasets#

nrow<-nrow(Final)
train<-nrow*0.8
test<-nrow*0.2
train_<-scaled.data[1:train,]


test_<-scaled.data[(train+1):nrow,]
test<-Final[(train+1):nrow,]

sapply(train_, mode)

train_ <- as.matrix(train_[,-c(1)])
train_x <- array_reshape(train_, dim = list(1481, 1, 287))

tic <- sys
model <- keras_model_sequential() %>%
  layer_conv_1d(filters = 32, kernel_size = 1, activation = "relu", input_shape = list(1,287))%>%
  layer_conv_1d(filters = 32, kernel_size = 1, activation = "relu", input_shape = list(1, 287))%>%
  layer_conv_1d(filters = 32, kernel_size = 1, activation = "relu", input_shape = list(1, 287))%>%
  layer_lstm(units=64, return_sequences=TRUE, activation="relu", input_shape = list(1, 287)) %>% 
  layer_lstm(units=64, dropout=0.2, recurrent_dropout = 0.5, return_sequences=TRUE, activation="relu") %>% 
  layer_lstm(units=64, dropout=0.2, recurrent_dropout = 0.5, return_sequences=TRUE, activation="relu") %>%
  layer_lstm(units=64, dropout=0.2, recurrent_dropout = 0.5, return_sequences=TRUE, activation="relu") %>%
  layer_lstm(units = 64, dropout=0.2, recurrent_dropout = 0.5) %>% 
  layer_dense(units = 1, activation = "softmax")


model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae", metrics= c("mse")
) 


test_ <- as.matrix(test_[,-c(1)])
test_x <- array_reshape(test_, dim = list(370, 1, 287))

train_y <- train_[,c(1)]
test_y <- test_[,c(1)]







history <- fit(model, 
                                x                = (train_x), 
                                y                = train_y,
                                batch_size       = 128, 
                                epochs           = 1000,
                                validation_split = 0.30
)
plot(history)

metrics1 <- model %>% evaluate(test_x, test_y)
metrics1



metrics2 <- model %>% evaluate(test_x, test_y)
metrics2


std <- apply(train_, 2, sd)
speed_error1<-metrics1$loss*std[[1]]
speed_accuracy1 <- (1-metrics1$loss)*100


speed_error2<-metrics2$loss*std[[1]]
speed_accuracy2 <- (1-metrics2$loss)*100


pred <- model %>%predict(test_x)

pred1 <- as.data.frame(pred)
compare <- bind_cols(pred1, as.data.frame(test_y))








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


