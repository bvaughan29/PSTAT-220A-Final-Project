########## FINAL PROJECT ##########
property <- read.table(text=" 
   size   age	  dc	  dt	 price
	 102.2 	 4.1 	 0.8 	 50.2 	 472.0 
	 102.7 	 16 	 5.5 	 51.4 	 660.1 
                       101.1 	 6.8 	 14.7 	 29.1 	 683.0 
                       121.2 	 18 	 6.5 	 66.2 	 473.2 
                       102.9 	 17 	 10.4 	 43.8 	 593.0 
                       94.4 	 12.4 	 15.1 	 54.2 	 570.6 
                       102.3 	 4.7 	 17.7 	 26.7 	 790.3 
                       72.9 	 18.6 	 13.7 	 56.8 	 364.8 
                       81.7 	 18.2 	 10.4 	 24.3 	 513.4 
                       107.0 	 3.6 	 18.1 	 13.2 	 644.6 
                       101.6 	 5.0 	 10.8 	 11.0 	 753.2 
                       127.8 	 6.8 	 2.7 	 43.4 	 815.7 
                       115.8 	 3.1 	 11.8 	 45.9 	 564.4 
                       88.5 	 20.2 	 13.2 	 21.8 	 625.5 
                       72.8 	 6.1 	 23 	 31.2 	 474.8 
                       105.6 	 3.2 	 13.3 	 19.2 	 643.5 
                       89.7 	 7.7 	 14.3 	 21.4 	 621.5 
                       79.5 	 12.8 	 16.3 	 30.7 	 635.7 
                       107.5 	 11.5 	 5.2 	 14.6 	 722.6 
                       91.6 	 4.9 	 15.3 	 29.4 	 599.7 
                       72.4 	 2.4 	 22.4 	 36.5 	 438.5 
                       81.1 	 39.7 	 13.4 	 39.5 	 317.8 
                       112.9 	 14.9 	 10.1 	 30.7 	 866.1 
                       113.3 	 21.5 	 16.4 	 27.2 	 741.7 
                       115.7 	 35.3 	 7.6 	 55.2 	 420.0 
                       87.9 	 3.6 	 29 	 61.6 	 467.0
                       109.3 	 11.5 	 18 	 61.8 	 502.5 
                       115.7 	 11.9 	 11 	 25.2 	 742.4 
                       82.3 	 22.2 	 21.9 	 10.9 	 496.0 
                       85.6 	 32.2 	 9.4 	 15.7 	 574.8 
                       113 	 2.2 	 7.3 	 38 	 699.7 
                       69.3 	 4.7 	 23.6 	 23.1 	 504.7 
                       79.6 	 26.5 	 10.9 	 48.4 	 387.5 
                       84.1 	 13.8 	 18.5 	 17.8 	 578.8 
                       100.2 	 4.3 	 11 	 44.5 	 469.3 
                       71 	 8.8 	 25 	 14.3 	 668.0
                       110.6 	 19 	 17 	 14.4 	 809.3 
                       107 	 10.9 	 20.4 	 36.3 	 804.3 
                       110.1 	 17.3 	 14.3 	 24.3 	 642.7 
                       109.7 	 14.7 	 7.6 	 54.3 	 563.0 
                       116.6 	 5.3 	 12.5 	 37.9 	 722.1 
                       105.1 	 16.5 	 24.1 	 37 	 450.2 
                       130.7 	 15.4 	 1.4 	 36.4 	 674.0 
                       118.3 	 8.8 	 13.9 	 28 	 906.0 
                       84.8 	 2.5 	 17.5 	 29.1 	 802.8 
                       70.6 	 3.6 	 27.6 	 28.7 	 402.4 
                       122.4 	 19 	 7.2 	 74.2 	 728.0 
                       101.6 	 3.1 	 6.0 	 32.4 	 505.6 
                       85.4 	 7.0 	 18.0 	 36.4 	 343.8 
                       86.6 	 3.6 	 24.2 	 6.9 	 537.0 
                       109.6 	 49.7 	 9.1 	 6.6 	 589.9 
                       79.1 	 25.5 	 16.1 	 65.5 	 453.9 
                       100.8 	 15.2 	 25.9 	 5.9 	 629.1 
                       95.6 	 20.5 	 8.1 	 53.6 	 547.4 
                       93.4 	 9.5 	 18.5 	 22.6 	 574.8 
                       87.7 	 7.1 	 14.6 	 31.3 	 555.0 
                       74.7 	 2.6 	 28 	 18.5 	 706.3 
                       118.1 	 14.9 	 0.3 	 69.9 	 590.0 
                       96 	 7.9 	 11.3 	 54.8 	 311.6 
                       90.5 	 2.4 	 16.7 	 10.9 	 483.8 
                       101.7 	 3.3 	 14.9 	 13.7 	 664.6 
                       102.7 	 5.3 	 16.6 	 56.5 	 537.0 
                       113.8 	 6.2 	 13.8 	 5.0 	 885.0 
                       115.9 	 19.4 	 13.2 	 37.6 	 593.0 
                       134.8 	 15.6 	 5.7 	 29.2 	 933.5 
                       124.5 	 8.1 	 8.0 	 34.4 	 807.3 
                       103.6 	 3.8 	 20.6 	 65.6 	 485.9 
                       85 	 7.9 	 12 	 64.5 	 528.5 
                       96.9 	 18.8 	 18.1 	 41.6 	 608.0 
                       93.5 	 3.6 	 11.1 	 56.7 	 452.2 
                       93.9 	 4.3 	 6.0 	 59.8 	 591.6 
                       97 	 16.6 	 9.3 	 73.1 	 487.5 
                       117.2 	 2.8 	 16.6 	 5.9 	 745.4 
                       69.7 	 12.2 	 24.8 	 31.2 	 375.6 
                       77.4 	 3.9 	 3.7 	 18.4 	 408.6 
                       94.2 	 3.3 	 18 	 19.5 	 615.5 
                       111.4 	 3.7 	 13.4 	 5.2 	 1005.5 
                       82.8 	 19 	 16.2 	 20.9 	 570.9 
                       116.8 	 8.5 	 14.1 	 74.2 	 416.4 
                       107.7 	 16.7 	 19.7 	 3.1 	 917.6 
                       94.1 	 5.1 	 21.9 	 9.0 	 583.8 
                       106.5 	 10.2 	 9.5 	 12.3 	 595.1 
                       113.9 	 34.6 	 12 	 35.7 	 565.2", 
   stringsAsFactors = FALSE,
   header = TRUE)

attach(property)

#preliminary exploratory data analysis
summary(property)

#plot all covariates against price
par(mfrow=c(2,2))
#size v price
plot(size,price,xlab="Property Size",ylab="Price")
#size v age
plot(age,price,xlab="Property Age",ylab="Price")
#size v dc
plot(dc,price,xlab="Property Dc",ylab="Price")
#size v dt
plot(dt,price,xlab="Property Dt",ylab="Price")

cor(size,price)
cor(age,price)
cor(dc,price)
cor(dt,price)

#QQ plots to check normality of each variable
par(mfrow=c(2,3))
qqnorm(price,main = "QQ plot for Price")
qqline(price)

qqnorm(size,main = "QQ plot for Size")
qqline(size)

qqnorm(age,main = "QQ plot for Age")
qqline(age)

qqnorm(dt,main = "QQ plot for Dt")
qqline(dt)

qqnorm(dc,main= "QQ plot for Dc")
qqline(dc)

par(mfrow=c(1,1))

shapiro.test(price)
shapiro.test(size)
shapiro.test(age)
shapiro.test(dc)
shapiro.test(dt)









#fit1 as simple combination of covariates
fit1 <- lm(price ~ size + age + dt + dc)

#check fit1 standardized residuals (since residuals are 
#correlated with non-constant variance)
fit1.stand <- rstandard(fit1)

#check fit1 deletion/studentized residuals (to reduce influence 
#of single obersvations that may be outliers; should be careful
#because we may miss outliers that are close together -> need
#to check outlier test, influences, and Cook's Distances)
fit1.delet <- rstudent(fit1)

#check fit1 QQ-plot (remember robustness against the Normality 
#assumption, this checks the normality of the errors and can alert
#us of possible outliers)
#regular residual QQ-plot
par(mfrow=c(1,3),cex=1)
qqnorm(residuals(fit1),ylab="Fit 1 Residuals",
       main = "QQ-plot of Fit 1 Residuals")
qqline(residuals(fit1))

#standardized residual QQ-plot
qqnorm(fit1.stand,ylab="Fit 1 Standardized Residuals",
       main = "QQ-plot of Fit 1 Standardized Residuals")
qqline(fit1.stand)

#deleted/studentized residual QQ-plot
qqnorm(fit1.delet,ylab="Fit 1 Deleted Residuals",
       main = "QQ-plot of Fit 1 Deleted Residuals")
qqline(fit1.delet)

#check residuals vs. fitted values (checks the constant variance 
#assumption and can alert us of possible outliers)
#regular residuals
plot(fitted(fit1), residuals(fit1), xlab="Fitted",
     ylab="Residuals", main = "Fit 1 Residuals vs
     Fitted Values")
abline(h=0)
#identify(fitted(fit1), residuals(fit1),n=2)
#standardized residuals
plot(fitted(fit1), fit1.stand, xlab="Fitted",
     ylab="Standardized Residuals", main = "Fit 1 Std Residuals vs
     Fitted Values")
abline(h=0)
#identify(fitted(fit1), fit1.stand,n=2)
#deleted residuals
plot(fitted(fit1), fit1.delet, xlab="Fitted",
     ylab="Deleted Residuals", main = "Fit 1 Deleted Residuals vs
     Fitted Values")
abline(h=0)
#identify(fitted(fit1), fit1.delet,n=2)

#test fit1 for outlier (far away from the model)
#(Bonferroni adjusted due to simultaneous testing of all
#observations, so we have multiple comparison issue)
library(car)
outlierTest(fit1)

#check leverages and Cook's distances (checks for influential points)
library(boot)
fit1.1 <- glm(price ~ size + age + dt + dc)
h.1 <- hatvalues(fit1)
cd.1 <- cooks.distance(fit1)
par(mfrow=c(1,2))
plot(cd.1,ylab="Fit 1 Cook statistic")
#identify(cd.1,n=2)
plot(h.1/(1-h.1),cd.1,ylab="Fit 1 Cook statistic",
     xlab="Leverage/(1-Leverage)")
identify(h.1/(1-h.1),cd.1,n=2)

#plot effect each observation has on each coefficient
fit1inf <- influence(fit1)
par(mfrow=c(2,2))
#size coefficient
plot(fit1inf$coef[,2],ylab="Change in Size coef")
#identify(fit1inf$coef[,2],n=1)
#age coefficient
plot(fit1inf$coef[,3],ylab="Change in Age coef")
#identify(fit1inf$coef[,3],n=3)
#dt coefficient
plot(fit1inf$coef[,4],ylab="Change in Dt coef")
#identify(fit1inf$coef[,4],n=3)
#dc coefficient
plot(fit1inf$coef[,5],ylab="Change in Dc coef")
#identify(fit1inf$coef[,5],n=2)

#check residuals vs. ALL covariates (alerts us to missing higher
#order terms)
#age clearly requires some higher order term
par(mfrow=c(2,2))
for(i in 1:4){
 plot(property[,i],residuals(fit1),
      xlab=names(property)[i],ylab="Residuals")
 abline(h=0)
}
par(mfrow=c(1,1))

#check added variable plots for each covariate
#(alerts us to the need of higher order terms
#for any covariates, outliers and/or influential points)
#added variable plot for size
par(mfrow=c(2,2))
d.1.size <- residuals(lm(price ~ age + dt + dc))
m.1.size <- residuals(lm(size ~ age + dt + dc))
plot(m.1.size,d.1.size,xlab="Size Residuals",ylab="Price Residuals",
     main = "Added Variable plot for Size")
abline(0,coef(fit1)[2])
lines(lowess(m.1.size,d.1.size),col="red",lty=2)
#added variable plot for age
#this plot shows possible need for square term for age
d.1.age <- residuals(lm(price ~ size + dt + dc))
m.1.age <- residuals(lm(age ~ size + dt + dc))
plot(m.1.age,d.1.age,xlab="Age Residuals",ylab="Price Residuals",
     main = "Added Variable plot for Age")
abline(0,coef(fit1)[3])
lines(lowess(m.1.age,d.1.age),col="red",lty=2)
#added variable plot for dt
d.1.dt <- residuals(lm(price ~ size + age + dc))
m.1.dt <- residuals(lm(dt ~ size + age + dc))
plot(m.1.dt,d.1.dt,xlab="Dt Residuals",ylab="Price Residuals",
     main = "Added Variable plot for Dt")
abline(0,coef(fit1)[4])
lines(lowess(m.1.dt,d.1.dt),col="red",lty=2)
#added variable plot for dc
#this plot shows possible need for square term for dc
d.1.dc <- residuals(lm(price ~ size + age + dt))
m.1.dc <- residuals(lm(dc ~ size + age + dt))
plot(m.1.dc,d.1.dc,xlab="Dc Residuals",ylab="Price Residuals",
     main = "Added Variable plot for Dc")
abline(0,coef(fit1)[5])
lines(lowess(m.1.dc,d.1.dc),col="red",lty=2)

#check partial residual plots for each covariate (alerts us to the 
#need for higher order terms or a transformation)
#partial residual plot for size
pr.1.size <- residuals(fit1)+coef(fit1)[2]*size
plot(size,pr.1.size,xlab="Size",ylab="Partial Residuals",
     main="Partial residual plot for Size")
abline(0,coef(fit1)[2])
lines(lowess(size,pr.1.size),col="red",lty=2)
#partial residual plot for age
#this plot shows possible need for square term for age
pr.1.age <- residuals(fit1)+coef(fit1)[3]*age
plot(age,pr.1.age,xlab="Age",ylab="Partial Residuals",
     main="Partial residual plot for Age")
abline(0,coef(fit1)[3])
lines(lowess(age,pr.1.age),col="red",lty=2)
#partial residual plot for dt
pr.1.dt <- residuals(fit1)+coef(fit1)[4]*dt
plot(dt,pr.1.dt,xlab="Dt",ylab="Partial Residuals",
     main="Partial residual plot for Dt")
abline(0,coef(fit1)[4])
lines(lowess(dt,pr.1.dt),col="red",lty=2)
#partial residual plot for dc
#this plot shows possible need for square term for dc
pr.1.dc <- residuals(fit1)+coef(fit1)[5]*dc
plot(dc,pr.1.dc,xlab="Dc",ylab="Partial Residuals",
     main="Partial residual plot for Dc")
abline(0,coef(fit1)[5])
lines(lowess(dc,pr.1.dc),col="red",lty=2)
par(mfrow=c(1,1))

#box-cox transformation
#Used in case the data are not normal (our previous QQ plot
#looked good, but try anyway to see if it make a difference)
#lambda=1 is in 95% CI for lambda, therefore, for the sake
#of symplicity, I do not believe there is a need for transformation
library(MASS)
boxcox(fit1,plotit=T,lambda=seq(-0.5,1.5,len=100))
title("Box-Cox Transformation for Fit 1")

#check collinearity (the case in which one or more covariates can be
#predicted by another covariate, causing small changes in the model
#or data to result in imprecise and unstable estimates of coeffs)
#check R^2 between each variable (regress each covariate on others
#R^2 close to 1 indicates problem)
round(cor(property[,1:4]),3)
#check VIF(function of R^2; VIF>5 considered an issue)
vif(fit1)
#check conditional number (≥30 considered an issue)
x.1 <- model.matrix(fit1)[,-1]
e.1 <- eigen(t(x.1)%*%x.1)
sqrt(e.1$val[1]/e.1$val)


















#after seeing added variable and partial residual plots for fit1
#fit new model, adding age^2 and dc^2 terms
fit2 <- lm(price ~ size + age + dt + dc + I(age^2) + I(dc^2))
summary(fit2)

#check fit2 standardized residuals (since residuals are 
#correlated with non-constant variance)
fit2.stand <- rstandard(fit2)

#check fit2 deletion/studentized residuals (to reduce influence 
#of single obersvations that may be outliers; should be careful
#because we may miss outliers that are close together -> need
#to check outlier test, influences, and Cook's Distances)
fit2.delet <- rstudent(fit2)

#check fit2 QQ-plot (remember robustness against the Normality 
#assumption, this checks the normality of the errors and can alert
#us of possible outliers)
#regular residual QQ-plot
par(mfrow=c(1,3),cex=1)
qqnorm(residuals(fit2),ylab="Fit 2 Residuals",
       main = "QQ-plot of Fit 2 Residuals")
qqline(residuals(fit2))

#standardized residual QQ-plot
qqnorm(fit2.stand,ylab="Fit 2 Standardized Residuals",
       main = "QQ-plot of Fit 2 Standardized Residuals")
qqline(fit2.stand)

#deleted/studentized residual QQ-plot
qqnorm(fit2.delet,ylab="Fit 2 Deleted Residuals",
       main = "QQ-plot of Fit 2 Deleted Residuals")
qqline(fit2.delet)

#check residuals vs. fitted values (checks the constant variance 
#assumption and can alert us of possible outliers)
#regular residuals
plot(fitted(fit2), residuals(fit2), xlab="Fitted",
     ylab="Residuals", main = "Fit 2 Residuals vs
     Fitted Values")
abline(h=0)
#identify(fitted(fit2), residuals(fit2),n=2)
#standardized residuals
plot(fitted(fit2), fit2.stand, xlab="Fitted",
     ylab="Standardized Residuals", main = "Fit 2 Std Residuals vs
     Fitted Values")
abline(h=0)
#identify(fitted(fit2), fit2.stand,n=2)
#deleted residuals
plot(fitted(fit2), fit2.delet, xlab="Fitted",
     ylab="Deleted Residuals", main = "Fit 2 Deleted Residuals vs
     Fitted Values")
abline(h=0)
#identify(fitted(fit2), fit2.delet,n=2)

#test fit2 for outlier (far away from the model)
#(Bonferroni adjusted due to simultaneous testing of all
#observations, so we have multiple comparison issue)
library(car)
outlierTest(fit2)

#check leverages and Cook's distances (checks for influential points)
library(boot)
fit2.1 <- glm(price ~ size + age + dt + dc + I(age^2) + I(dc^2))
h.2 <- hatvalues(fit2)
cd.2 <- cooks.distance(fit2)
par(mfrow=c(1,2))
plot(cd.2,ylab="Fit 2 Cook statistic")
identify(cd.2,n=2)
plot(h.2/(1-h.2),cd.2,ylab="Fit 2 Cook statistic",
     xlab="Leverage/(1-Leverage)")
identify(h.2/(1-h.2),cd.2,n=2)

#plot effect each observation has on each coefficient
par(mfrow=c(2,3))
fit2inf <- influence(fit2)
#size coefficient
plot(fit2inf$coef[,2],ylab="Change in Size coef")
identify(fit2inf$coef[,2],n=1)
#age coefficient
plot(fit2inf$coef[,3],ylab="Change in Age coef")
identify(fit2inf$coef[,3],n=1)
#dt coefficient
plot(fit2inf$coef[,4],ylab="Change in Dt coef")
identify(fit2inf$coef[,4],n=4)
#dc coefficient
plot(fit2inf$coef[,5],ylab="Change in Dc coef")
identify(fit2inf$coef[,5],n=1)
#age^2 coefficient
plot(fit2inf$coef[,6],ylab="Change in Age^2 coef")
identify(fit2inf$coef[,6],n=1)
#dc^2 coefficient
plot(fit2inf$coef[,7],ylab="Change in Dc^2 coef")
identify(fit2inf$coef[,7],n=1)

#box-cox transformation
#Used in case the data are not normal (our previous QQ plot
#looked good, but we see some outliers in diagnostic plots)
#lambda=1 is in the 95% CI but it's close, so I try lambda=0.5
#to see if we get a better fit (fit3)
library(MASS)
par(mfrow=c(1,1))
boxcox(fit2,plotit=T,lambda=seq(-0.5,1.5,len=100))
#did not help to reduce outliers

#check collinearity (the case in which one or more covariates can be
#predicted by another covariate, causing small changes in the model
#or data to result in imprecise and unstable estimates of coeffs)
#check R^2 between each variable (regress each covariate on others
#R^2 close to 1 indicates problem)
property.2 <- data.frame(size,age,dc,dt,age^2,dc^2)
round(cor(property.2[,1:6]),3)
#check VIF(function of R^2; VIF>5 considered an issue)
library(car)
vif(fit2)
#check conditional number (≥30 considered an issue)
x.2 <- model.matrix(fit2)[,-1]
e.2 <- eigen(t(x.2)%*%x.2)
sqrt(e.2$val[1]/e.2$val)
#no sign of collinearity

#VARIABLE SELECTION
#compare nested models (tests that the added square
#are signficant -> p=0.07414 -> close, continue)
anova(fit1,fit2)

#p-value method
summary(fit2)
#dc^2 has largest non-significant p-value
p1.fit2 <- lm(price ~ size + age + dt + dc + I(age^2))
summary(p1.fit2)
#dc has largest non-significant p-value
p2.fit2 <- lm(price ~ size + age + dt + I(age^2))
summary(p2.fit2)
#age not significant, but age^2 is significant
#-> leave age and age^2 in model -> stop

#adjusted R squared method
library(leaps)
a.2 <- regsubsets(formula(fit2), data = property, method = "exhaustive")
(rs.2 <- summary(a.2))
rs.2$adjr2
plot(2:7,rs.2$adjr2,xlab="Number of Parameters",
     ylab="Adjusted R^2")

#AIC forward method
step(lm(price ~ 1), direction = "forward",
     scope=list(upper=formula(fit2)))
#AIC backward method
step(fit2, direction = "backward")
#AIC both directions method
step(fit2, direction = "both")
#AIC and BIC graphs
par(mfrow=c(2,2))
n <- nrow(property)
AIC <- n*log(rs.2$rss) + 2*(2:7)
BIC <- n*log(rs.2$rss) + log(n)*(2:7)
plot(2:7,AIC,xlab="Number of Parameters",ylab="AIC")
plot(2:7,BIC,xlab="Number of Parameters",ylab="BIC")

#Mallow's Cp method
rs.2$cp
plot(2:7,rs.2$cp,xlab="Number of Parameters",
     ylab="Mallow's Cp")
abline(0,1)

#10-fold cross-validation method
library(boot)
X.2 <- model.matrix(fit2)
fold.2 <- sample(rep(1:10,8))
pse.cv.2 <- matrix(NA,6,10)
for(i in 1:6){
  for(j in 1:10){
    tmp <- lm(price~X.2[,rs.2$which[i,]]-1,subset=fold.2!=j)
    pred <- X.2[fold.2==j,rs.2$which[i,]]%*%coef(tmp)
    pse.cv.2[i,j]<-mean((price[fold.2==j]-pred)^2)
  }
}
plot(2:7,apply(pse.cv.2,1,mean),xlab="Number of parameters",
     ylab="CV estimates of prediction errors")
par(mfrow=c(1,1))

#LASSO method
install.packages("glmnet",repos = "http://cran.us.r-project.org")
library(glmnet)
X.lasso.2<-model.matrix(fit2)[,-1]
set.seed(1)
fit.lasso.2 <- glmnet(X.lasso.2, price, lambda.min=0,nlambda=101,
                     alpha = 1)
plot(fit.lasso.2,xvar="lambda",xlim=c(-4,5))
text(-3.5,coef(fit.lasso.2)[-1,length(fit.lasso.2$lambda)],
     labels=colnames(X.lasso.2),cex=0.6)
fit.lasso.cv.2 <- cv.glmnet(X.lasso.2, price, lambda.min = 0,
                           nlambda = 101)
abline(v=log(fit.lasso.cv.2$lambda.min),col="red")
mtext("CV estimate",side=1,at=log(fit.lasso.cv.2$lambda.min),cex=.6)
title("LASSO graphical summary for Fit 2")
(coeff.lasso.2 <- predict(fit.lasso.2, type = "coefficients",
                         s = fit.lasso.cv.2$lambda.min))













#after seeing outliers in diagnostic plots for fit2
fit3 <- lm(I((sqrt(price)-1)/0.5) ~ size + age + dt + dc + I(age^2) + I(dc^2))

library(boot)
fit3.1 <- glm(I((sqrt(price)-1)/0.5) ~ size + age + dt + dc + I(age^2) + I(dc^2))
glm.diag.plots(fit3.1)
#transformation makes outliers more apparent -> do not use
















#after seeing outliers in diagnostic plots and coefficient
#effect plots of fit 2, do sensitivity analysis
prop.new <- data.frame(property[-c(51,57),])
detach(property)
attach(prop.new)

fit4 <- lm(price ~ size + age + dt + dc + I(age^2) + I(dc^2))
summary(fit4)

#check fit2 standardized residuals (since residuals are 
#correlated with non-constant variance)
fit4.stand <- rstandard(fit4)

#check fit4 deletion/studentized residuals (to reduce influence 
#of single obersvations that may be outliers; should be careful
#because we may miss outliers that are close together -> need
#to check outlier test, influences, and Cook's Distances)
fit4.delet <- rstudent(fit4)

#check fit2 QQ-plot (remember robustness against the Normality 
#assumption, this checks the normality of the errors and can alert
#us of possible outliers)
#regular residual QQ-plot
qqnorm(residuals(fit4),ylab="Fit 4 Residuals",
       main = "QQ-plot of Fit 4 Residuals")
qqline(residuals(fit4))

#standardized residual QQ-plot
qqnorm(fit4.stand,ylab="Fit 4 Standardized Residuals",
       main = "QQ-plot of Fit 4 Standardized Residuals")
qqline(fit4.stand)

#deleted/studentized residual QQ-plot
qqnorm(fit4.delet,ylab="Fit 4 Deleted Residuals",
       main = "QQ-plot of Fit 4 Deleted Residuals")
qqline(fit4.delet)

#check residuals vs. fitted values (checks the constant variance 
#assumption and can alert us of possible outliers)
#regular residuals
par(mfrow=c(1,3))
plot(fitted(fit4), residuals(fit4), xlab="Fitted",
     ylab="Residuals", main = "Fit 4 Residuals vs
     Fitted Values")
abline(h=0)
#standardized residuals
plot(fitted(fit4), fit4.stand, xlab="Fitted",
     ylab="Standardized Residuals", main = "Fit 4 Std Residuals vs
     Fitted Values")
abline(h=0)
#identify(fitted(fit4), fit4.stand,n=2)
#deleted residuals
plot(fitted(fit4), fit4.delet, xlab="Fitted",
     ylab="Deleted Residuals", main = "Fit 4 Deleted Residuals vs
     Fitted Values")
abline(h=0)
#identify(fitted(fit4), fit4.delet,n=2)

#test fit2 for outlier (far away from the model)
#(Bonferroni adjusted due to simultaneous testing of all
#observations, so we have multiple comparison issue)
library(car)
outlierTest(fit4)

#check leverages and Cook's distances (checks for influential points)
library(boot)
fit4.1 <- glm(price ~ size + age + dt + dc + I(age^2) + I(dc^2))
h.4 <- hatvalues(fit4)
cd.4 <- cooks.distance(fit4)
par(mfrow=c(1,2))
plot(cd.4,ylab="Fit 4 Cook stat")
#identify(cd.4,n=2)
plot(h.4/(1-h.4),cd.4,ylab="Fit 4 Cook statistic")
#identify(h.4/(1-h.4),cd.4,n=2)

#plot effect each observation has on each coefficient
fit4inf <- influence(fit4)
par(mfrow=c(2,3))
#size coefficient
plot(fit4inf$coef[,2],ylab="Change in Size coef")
#identify(fit4inf$coef[,2],n=1)
#age coefficient
plot(fit4inf$coef[,3],ylab="Change in Age coef")
#identify(fit4inf$coef[,3],n=1)
#dt coefficient
plot(fit4inf$coef[,4],ylab="Change in Dt coef")
#identify(fit4inf$coef[,4],n=4)
#dc coefficient
plot(fit4inf$coef[,5],ylab="Change in Dc coef")
#identify(fit4inf$coef[,5],n=1)
#age^2 coefficient
plot(fit4inf$coef[,6],ylab="Change in Age^2 coef")
#identify(fit4inf$coef[,6],n=1)
#dc^2 coefficient
plot(fit4inf$coef[,7],ylab="Change in Dc^2 coef")
#identify(fit4inf$coef[,7],n=1)

#box-cox transformation
#Used in case the residuals are not normal or have non-constant variance
#(the residual plot for fit4 is iffy, can see slight U pattern)
#lambda=1 is not in the 95% CI so I try lambda=0.5 in fit5
library(MASS)
par(mfrow=c(1,1))
boxcox(fit4,plotit=T,lambda=seq(-0.5,1.5,len=100))















#after seeing residual and boxcox plots and coefficient effect plots of fit4
fit5 <- lm(I((sqrt(price)-1)/0.5) ~ size + age + dt + dc + I(age^2) + I(dc^2),data=prop.new)
summary(fit5)

#check fit5 standardized residuals (since residuals are 
#correlated with non-constant variance)
fit5.stand <- rstandard(fit5)

#check fit5 deletion/studentized residuals (to reduce influence 
#of single obersvations that may be outliers; should be careful
#because we may miss outliers that are close together -> need
#to check outlier test, influences, and Cook's Distances)
fit5.delet <- rstudent(fit5)

#check fit5 QQ-plot (remember robustness against the Normality 
#assumption, this checks the normality of the errors and can alert
#us of possible outliers)
par(mfrow=c(1,3))
#regular residual QQ-plot
qqnorm(residuals(fit5),ylab="Fit 5 Residuals",
       main = "QQ-plot of Fit 5 Residuals")

#standardized residual QQ-plot
qqnorm(fit5.stand,ylab="Fit 5 Standardized Residuals",
       main = "QQ-plot of Fit 5 Standardized Residuals")
abline(0,1)

#deleted/studentized residual QQ-plot
qqnorm(fit5.delet,ylab="Fit 5 Deleted Residuals",
       main = "QQ-plot of Fit 5 Deleted Residuals")
abline(0,1)

#check residuals vs. fitted values (checks the constant variance 
#assumption and can alert us of possible outliers)
#regular residuals
par(mfrow=c(1,3))
plot(fitted(fit5), residuals(fit5), xlab="Fitted",
     ylab="Residuals", main = "Fit 5 Residuals vs
     Fitted Values")
abline(h=0)
#standardized residuals
plot(fitted(fit5), fit5.stand, xlab="Fitted",
     ylab="Standardized Residuals", main = "Fit 5 Std Residuals vs
     Fitted Values")
abline(h=0)
#identify(fitted(fit5), fit5.stand,n=2)
#deleted residuals
plot(fitted(fit5), fit5.delet, xlab="Fitted",
     ylab="Deleted Residuals", main = "Fit 5 Deleted Residuals vs
     Fitted Values")
abline(h=0)
#identify(fitted(fit5), fit5.delet,n=2)

#test fit5 for outlier (far away from the model)
#(Bonferroni adjusted due to simultaneous testing of all
#observations, so we have multiple comparison issue)
library(car)
outlierTest(fit5)

#check leverages and Cook's distances (checks for influential points)
library(boot)
fit5.1 <- glm(I((sqrt(price)-1)/0.5) ~ size + age + dt + dc + I(age^2) + I(dc^2))
h.5 <- hatvalues(fit5)
cd.5 <- cooks.distance(fit5)
par(mfrow=c(1,2))
plot(cd.5,ylab="Fit 5 Cook stat")
#identify(cd.5,n=1)
plot(h.5/(1-h.5),cd.5,ylab="Fit 5 Cook statistic")
#identify(h.5/(1-h.5),cd.5,n=2)

#plot effect each observation has on each coefficient
par(mfrow=c(2,3))
fit5inf <- influence(fit5)
#size coefficient
plot(fit5inf$coef[,2],ylab="Change in Size coef")
#identify(fit5inf$coef[,2],n=1)
#age coefficient
plot(fit5inf$coef[,3],ylab="Change in Age coef")
#identify(fit5inf$coef[,3],n=1)
#dt coefficient
plot(fit5inf$coef[,4],ylab="Change in Dt coef")
#identify(fit5inf$coef[,4],n=4)
#dc coefficient
plot(fit5inf$coef[,5],ylab="Change in Dc coef")
#identify(fit5inf$coef[,5],n=1)
#age^2 coefficient
plot(fit5inf$coef[,6],ylab="Change in Age^2 coef")
#identify(fit5inf$coef[,6],n=1)
#dc^2 coefficient
plot(fit5inf$coef[,7],ylab="Change in Dc^2 coef")
#identify(fit5inf$coef[,7],n=1)

#box-cox transformation
#Used in case the residuals are not normal or non-constant variance
#lambda=1 is in the 95% CI, so no need to transform again
library(MASS)
par(mfrow=c(1,1))
boxcox(fit5,plotit=T,lambda=seq(-1,2.5,len=100))

#check collinearity (the case in which one or more covariates can be
#predicted by another covariate, causing small changes in the model
#or data to result in imprecise and unstable estimates of coeffs)
#check R^2 between each variable (regress each covariate on others
#R^2 close to 1 indicates problem)
property.5 <- data.frame(size,age,dc,dt,age^2,dc^2)
round(cor(property.5[,1:6]),3)
#check VIF(function of R^2; VIF>5 considered an issue)
library(car)
vif(fit5)
#check conditional number (≥30 considered an issue)
x.5 <- model.matrix(fit5)[,-1]
e.5 <- eigen(t(x.5)%*%x.5)
sqrt(e.5$val[1]/e.5$val)
#no sign of collinearity

#VARIABLE SELECTION

#p-value method
summary(fit5)
#dc^2 has largest non-significant p-value
p1.fit5 <- lm(I((sqrt(price)-1)/0.5) ~ size + age + dt + dc + I(age^2))
summary(p1.fit5)
#dc has largest non-significant p-value
p2.fit5 <- lm(I((sqrt(price)-1)/0.5) ~ size + age + dt + I(age^2))
summary(p2.fit5)
#age not significant, but age^2 is significant
#-> leave age and age^2 in model -> stop

#adjusted R squared method
library(leaps)
a.5 <- regsubsets(formula(fit5), data = prop.new, method = "exhaustive")
(rs.5 <- summary(a.5))
rs.5$adjr2
plot(2:7,rs.5$adjr2,xlab="Number of Parameters",
     ylab="Adjusted R^2")

#AIC forward method
step(lm(I((sqrt(price)-1)/0.5) ~ 1), direction = "forward",
     scope=list(upper=formula(fit5)))
#AIC backward method
step(fit5, direction = "backward")
#AIC both directions method
step(fit5, direction = "both")
#AIC and BIC graphs
n <- nrow(prop.new)
AIC <- n*log(rs.5$rss) + 2*(2:7)
BIC <- n*log(rs.5$rss) + log(n)*(2:7)
par(mfrow=c(2,2))
plot(2:7,AIC,xlab="Number of Parameters",ylab="AIC")
plot(2:7,BIC,xlab="Number of Parameters",ylab="BIC")

#Mallow's Cp method
rs.5$cp
plot(2:7,rs.5$cp,xlab="Number of Parameters",
     ylab="Mallow's Cp")
abline(0,1)

#10-fold cross-validation method
library(boot)
X.5 <- model.matrix(fit5)
fold.5 <- sample(rep(1:10,8))
pse.cv.5 <- matrix(NA,6,10)
for(i in 1:6){
  for(j in 1:10){
    tmp <- lm(price~X.5[,rs.5$which[i,]]-1,subset=fold.5!=j)
    pred <- X.5[fold.5==j,rs.5$which[i,]]%*%coef(tmp)
    pse.cv.5[i,j]<-mean((price[fold.5==j]-pred)^2)
  }
}
par(mfrow=c(1,1))
plot(2:7,apply(pse.cv.5,1,mean),xlab="Number of parameters",
     ylab="CV estimates of prediction errors")


#LASSO method
install.packages("glmnet",repos = "http://cran.us.r-project.org")
library(glmnet)
X.lasso.5<-model.matrix(fit5)[,-1]
set.seed(1)
fit.lasso.5 <- glmnet(X.lasso.5, price, lambda.min=0,nlambda=101,
                      alpha = 1)
plot(fit.lasso.5,xvar="lambda",xlim=c(-4,5))
text(-3.5,coef(fit.lasso.5)[-1,length(fit.lasso.5$lambda)],
     labels=colnames(X.lasso.5),cex=0.6)
fit.lasso.cv.5 <- cv.glmnet(X.lasso.5, price, lambda.min = 0,
                            nlambda = 101)
abline(v=log(fit.lasso.cv.5$lambda.min),col="red")
mtext("CV estimate",side=1,at=log(fit.lasso.cv.5$lambda.min),cex=.6)
(coeff.lasso.5 <- predict(fit.lasso.5, type = "coefficients",
                          s = fit.lasso.cv.5$lambda.min))















#after seeing variable selection methods in fit5
fit6 <- lm(I((sqrt(price)-1)/0.5) ~ size + age + dt + I(age^2))
summary(fit6)

#check fit6 standardized residuals (since residuals are 
#correlated with non-constant variance)
fit6.stand <- rstandard(fit6)

#check fit6 deletion/studentized residuals (to reduce influence 
#of single obersvations that may be outliers; should be careful
#because we may miss outliers that are close together -> need
#to check outlier test, influences, and Cook's Distances)
fit6.delet <- rstudent(fit6)

#check fit6 QQ-plot (remember robustness against the Normality 
#assumption, this checks the normality of the errors and can alert
#us of possible outliers)
#regular residual QQ-plot
par(mfrow=c(1,3))
qqnorm(residuals(fit6),ylab="Fit 6 Residuals",
       main = "QQ-plot of Fit 6 Residuals")

#standardized residual QQ-plot
qqnorm(fit6.stand,ylab="Fit 6 Standardized Residuals",
       main = "QQ-plot of Fit 6 Standardized Residuals")
abline(0,1)

#deleted/studentized residual QQ-plot
qqnorm(fit6.delet,ylab="Fit 6 Deleted Residuals",
       main = "QQ-plot of Fit 6 Deleted Residuals")
abline(0,1)

#check residuals vs. fitted values (checks the constant variance 
#assumption and can alert us of possible outliers)
#no large difference noted in QQ-plots between different
#residuals, so only doing regular residuals here and following
#regular residuals
par(mfrow=c(1,3))
plot(fitted(fit6), residuals(fit6), xlab="Fitted",
     ylab="Residuals", main = "Fit 6 Residuals vs
     Fitted Values")
abline(h=0)
#standardized residuals
plot(fitted(fit6), fit6.stand, xlab="Fitted",
     ylab="Standardized Residuals", main = "Fit 6 Std Residuals vs
     Fitted Values")
abline(h=0)
#identify(fitted(fit6), fit6.stand,n=2)
#deleted residuals
plot(fitted(fit6), fit6.delet, xlab="Fitted",
     ylab="Deleted Residuals", main = "Fit 6 Deleted Residuals vs
     Fitted Values")
abline(h=0)
#identify(fitted(fit6), fit6.delet,n=2)


#test fit6 for outlier (far away from the model)
#(Bonferroni adjusted due to simultaneous testing of all
#observations, so we have multiple comparison issue)
library(car)
outlierTest(fit6)

#check leverages and Cook's distances (checks for influential points)
library(boot)
fit6.1 <- glm(I((sqrt(price)-1)/0.5) ~ size + age + dt + I(age^2))
h.6 <- hatvalues(fit6)
cd.6 <- cooks.distance(fit6)
par(mfrow=c(1,2))
plot(cd.6,ylab="Fit 6 Cook stat")
#identify(cd.6,n=1)
plot(h.6/(1-h.6),cd.6,ylab="Fit 6 Cook statistic",
     xlab="Leverage/(1-Leverage)")
#identify(h.6/(1-h.6),cd.6,n=2)

#plot effect each observation has on each coefficient
fit6inf <- influence(fit6)
par(mfrow=c(2,2))
#size coefficient
plot(fit6inf$coef[,2],ylab="Change in Size coef")
#identify(fit6inf$coef[,2],n=1)
#age coefficient
plot(fit6inf$coef[,3],ylab="Change in Age coef")
#identify(fit6inf$coef[,3],n=1)
#dt coefficient
plot(fit6inf$coef[,4],ylab="Change in Dt coef")
#identify(fit6inf$coef[,4],n=4)
#age^2 coefficient
plot(fit6inf$coef[,5],ylab="Change in Age^2 coef")
#identify(fit6inf$coef[,6],n=1)

#box-cox transformation
#Used in case the residuals are not normal or non-constant variance
#lambda=1 is in the 95% CI, so no need to transform again
library(MASS)
boxcox(fit6,plotit=T,lambda=seq(-1,2.5,len=100))

#check collinearity (the case in which one or more covariates can be
#predicted by another covariate, causing small changes in the model
#or data to result in imprecise and unstable estimates of coeffs)
#check R^2 between each variable (regress each covariate on others
#R^2 close to 1 indicates problem)
property.6 <- data.frame(size,age,dt,age^2)
round(cor(property.6[,1:4]),3)
#check VIF(function of R^2; VIF>5 considered an issue)
library(car)
vif(fit6)
#check conditional number (≥30 considered an issue)
x.6 <- model.matrix(fit6)[,-1]
e.6 <- eigen(t(x.6)%*%x.6)
sqrt(e.6$val[1]/e.6$val)
#no sign of collinearity

#test nested models
anova(fit5,fit6)

summary(fit6)
























