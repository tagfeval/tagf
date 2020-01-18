setwd("/users/nakessien/downloads/data/")
temp=list.files("/users/nakessien/downloads/data/",pattern="*.csv")
myfiles=as.data.frame(lapply(temp, (read.csv)))
myfiles<-myfiles[-c(1:8,34:38),]

myfiles$GM_JOURNEY_TIME...Site.1034.to.Site.1033...Multi.Day.Speed<-NULL
myfiles$GM_JOURNEY_TIME...Site.1034.to.Site.1033...Multi.Day.Speed.1<-NULL
myfiles$GM_JOURNEY_TIME...Site.1034.to.Site.1033...Multi.Day.Speed.2<-NULL
myfiles$GM_JOURNEY_TIME...Site.1034.to.Site.1033...Multi.Day.Speed.3<-NULL
myfiles$GM_JOURNEY_TIME...Site.1034.to.Site.1033...Multi.Day.Speed.4<-NULL
myfiles$GM_JOURNEY_TIME...Site.1034.to.Site.1033...Multi.Day.Speed.5<-NULL
myfiles$GM_JOURNEY_TIME...Site.1034.to.Site.1033...Multi.Day.Speed.6<-NULL
myfiles$GM_JOURNEY_TIME...Site.1034.to.Site.1033...Multi.Day.Speed.7<-NULL
myfiles$GM_JOURNEY_TIME...Site.1034.to.Site.1033...Multi.Day.Speed.8<-NULL
myfiles$GM_JOURNEY_TIME...Site.1034.to.Site.1033...Multi.Day.Speed.9<-NULL
myfiles$GM_JOURNEY_TIME...Site.1034.to.Site.1033...Multi.Day.Speed.10<-NULL
myfiles$GM_JOURNEY_TIME...Site.1034.to.Site.1033...Multi.Day.Speed.11<-NULL

T1<-t(myfiles)
T1<-as.data.frame(T1)
T1<-cbind(rownames(T1),T1)
rownames(T1)<-NULL
T1<-select(T1,-`rownames(T1)`)
myfiles1<-gather(T1,"Date","Speed",2:25)
Ar_tbl<-select(myfiles1,Speed)


Final<-separate(myfiles1, `9`, into =c("Weekday","Day","Month","Year"))
Weekday_tbl<-filter(Final, Weekday %in% c("Mon","Tue","Wed","Thu","Fri"))  
Weekend_tbl<-filter(Final, Weekday %in% c("Sat","Sun"))
Sunday_tbl<-filter(Final, Weekday %in% "Sun")  
TimeSeries<-ts(Ar_tbl,frequency = 24,start=1, end=374)
TimeSeries<-na.remove(TimeSeries)
plot(TimeSeries,xlab="Day of Month",ylab="Average Speed")

#Stationarise Data#
plot(diff(TimeSeries[1:50]), type="l",xlab="Day of Month", ylab="Differenced Avg. Speed")
plot(log10(TimeSeries),xlab="Day",ylab="Log10 Avg. Speed")
plot(diff(log10(TimeSeries)),ylab="Diff. Log(Average Speed)")
plot(diff(diff(log10(TimeSeries))),ylab="Double Diff. Average Speed")

#Test for Stationarity#
adf.test(diff(diff(log10(TimeSeries))),alternative = "stationary")
Box.test(diff(diff(log10(TimeSeries))),lag=20,type = "Ljung-Box")
kpss.test(diff(diff(log10(TimeSeries))))
#Is data stationary?#

#Plot ACF and PACF to identify potential AR and MA model
par(mfrow=c(1,2))
acf(ts((((TimeSeries)))),main="ACF Avg. Speed")
pacf(ts((((TimeSeries)))),main="PACF Avg. Speed")
#ACF and PACF Plots should make you confirm stationarity#

#Identify best fit ARIMA model#
ARIMAfit<-auto.arima((TimeSeries),approximation = FALSE,trace = FALSE)
summary(ARIMAfit)

#Make the forecasts using the best fit ARIMA#
pred <-predict(ARIMAfit, n.ahead = 48)
pred

#Plot the forecasts#
plot(TimeSeries,type = "l", xlim = c(370,380),xlab="Day",ylab="Avg. Speed")
lines(10^(pred$pred), col = "blue")
lines(10^(pred$pred+1*pred$se), col = "orange")
lines(10^(pred$pred-1*pred$se), col = "orange")

#Plot ACF and PACF for residuals to be sure all information is extracted#
par(mfrow=c(1,2))
acf(ts(ARIMAfit$residuals),main="ACF Residual")
pacf(ts(ARIMAfit$residuals),main="PACF Residual")

#Plots for Analysis#
plot(Sunday_tbl$X, Sunday_tbl$`Average Speed`,main="Plot of Sunday Average Speeds",xlab="Time of Day",ylab="Average Speed",type="h")
plot(Weekday_tbl$X, Weekday_tbl$`Average Speed`,main="Plot of Weekday Average Speeds",xlab="Time of Day",ylab="Average Speed",type="h")
plot(Weekend_tbl$X, Weekend_tbl$`Average Speed`,main="Plot of Weekend Average Speeds",xlab="Time of Day",ylab="Average Speed",type="h")

