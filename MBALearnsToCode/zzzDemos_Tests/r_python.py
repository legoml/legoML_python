---
title: 'Midterm Exam: Bikeshare Analysis'
output: pdf_document
fontsize: 12
geometry: margin=0.5in
---
(Student: Vinh Luong - 442069)


``` {r echo=FALSE, message=FALSE, warning=FALSE, results='hide'}
# Set workding directory
setwd('C:/Cloud/Box Sync/WORK/Chicago Booth/COURSES/3. Big Data/Assignments/05. Bikeshare (Mid-Term)')
# Load key packages
library(data.table)
library(plyr)
library(reshape2)
library(ggplot2)
library(caret)
library(gamlr)
library(lubridate)
# Start parallel computing cluster over multi cores
library(doParallel)
cl <- makeCluster(detectCores()-1)
registerDoParallel(cl)
getDoParWorkers()
```


# Data Pre-Processing

We first read the Bikeshare data into a *biketab* data table and perform the following pre-processing steps:

1. Rename the levels of the ***season*** factor variable to "*(01) Spring*", "*(02) Summer*", "*(03) Fall*" and "*(04) Winter*"
2. Rename the levels of the ***yr*** factor variable to *2011* and *2012*
3. Rename the levels of the ***mnth*** factor variable to "*(01) Jan*", "*(02) Feb*", "*(03) Mar*", "*(04) Apr*", "*(05) May*", "*(06) Jun*", "*(07) Jul*", "*(08) Aug*", "*(09) Sep*", "*(10) Oct*", "*(11) Nov*" and "*(12) Dec*"
4. Rename the levels of the ***weekday*** factor variable to "*(01) Sun*", "*(02) Mon*", "*(03) Tue*", "*(04) Wed*", "*(05) Thu*", "*(06) Fri*" and "*(07) Sat*"
5. Rename the levels of the ***weathersit*** factor variables to "*(01) Good*", "*(02) Cloudy*", "*(03) Bad*" and "*(04) Very Bad*"
6. Extract the day from the ***dteday*** variable and turn this variable into a factor

```{r echo=FALSE, results='hide'}
# read data into data table
biketab <- fread("bikeshare.csv")
# indicate factor variables
biketab[, `:=`(season = relevel(factor(mapvalues(season, from = 1 : 4,
                                                 to = c("(01) Spring", "(02) Summer", "(03) Fall", "(04) Winter"))),
                                ref = "(01) Spring"),
               yr = factor(mapvalues(yr,
                                     from = 0 : 1,
                                     to = 2011 : 2012)),
               mnth = relevel(factor(mapvalues(mnth, from = 1 : 12,
                                               to = c("(01) Jan", "(02) Feb", "(03) Mar", "(04) Apr",
                                                      "(05) May", "(06) Jun", "(07) Jul", "(08) Aug",
                                                      "(09) Sep", "(10) Oct", "(11) Nov", "(12) Dec"))),
                              ref = "(01) Jan"),
               hr = factor(hr),
               holiday = factor(holiday),
               weekday = relevel(factor(mapvalues(weekday, from = 0 : 6,
                                                  to = c("(01) Sun", "(02) Mon", "(03) Tue", "(04) Wed",
                                                         "(05) Thu", "(06) Fri", "(07) Sat"))),
                                 ref = "(01) Sun"),
               notbizday = factor(notbizday),
               weathersit = relevel(factor(mapvalues(weathersit, from = 1 : 4,
                                                     to = c("(01) Good", "(02) Cloudy", "(03) Bad", "(04) Very Bad"))),
                                    ref = "(01) Good"),
               dteday = factor(dteday))]
```


# QUESTION 1: Models, Outliers and False Discovery


## QUESTION 1.1:

We first consider a simple linear regression of the daily rental bike totals (*total*) on an interaction between *yr* and *mnth*:

```{r echo=FALSE}
# calculate total cnt by day, keeping track of the corresponding yr and mnth id.
daytots <- biketab[, .(total = sum(cnt),
                       mean_temp = mean(temp),
                       mean_wind = mean(windspeed),
                       mean_hum = mean(hum)),
                   by=c("dteday", "yr", "mnth",
                        "weekday", "holiday", "notbizday", "season")]
row.names(daytots) <- daytots$dteday
```

```{r}
daylm <- glm(total ~ yr*mnth, data=daytots)
```

This regression has an in-sample SSE (deviance) of **`r formatC(daylm$deviance, format = "g")`**, compared with a SST (null deviance) of `r formatC(daylm$null.deviance, format = "g")`. Its *R*^2^ statistic is hence **`r 1 - daylm$deviance / daylm$null.deviance`**.


## QUESTION 1.2:

The mathematical formula for this simple regression is:

$$
\begin{aligned}
  & total_t = \beta_0 + \beta_1 \cdot yr_t + \beta_2 \cdot mnth_t + \beta_3 \cdot yr_t \cdot mnth_t + \epsilon_t,
  \\
  & \text{where: } \epsilon_t \sim \mathcal N(0, \sigma^2)
\end{aligned}
$$

where *yr* and *mnth* are factor variables indicating the year (2011 or 2012) and the month (Jan - Dec), and *t* is the indicator of the sample date. The $\beta_0$ coefficient represents the average rental bike total for a typical day in Jan 2011. The $\beta_1$, $\beta_2$ and $\beta_3$ coefficients measure how the average daily rental bike total varies as the month and year differs from Jan and 2011.

In estimating this model, we are maximizing the log likelihood:

$$
l = \text{log} \prod_{t = 1}^T \text{p}(y_t | \mathbf x_t)
  = \sum_{t = 1}^T \text{log } \text{p}_{\mathcal N(\mathbf E[y_t|\mathbf x_t], \sigma^2)}(y_t)
  = \text{constant} - \sum_{t = 1}^T (y_t - \hat{y}_t)^2
$$

where $y$ is the dependent variable *total*, $\mathbf x$ captures the independent variables *yr* and *mnth*, and $\hat{y}$ indicates the predicted values $\mathbf E[y_t | x _t]$. This log likelihood maximization is equivalent to minimizing the sum of squared errors (SSE) $\sum_{t = 1}^T (y_t - \hat{y}_t)^2$, which is the model's deviance in this case.

This model is likely to be too simplistic to be a good predictor of bike rental demand, as its granularity is only at the monthly level and it excludes day-to-day variables that affect demand, such as weather and holidays.


## QUESTION 1.3:

We now consider the standardized residuals $r_t = \frac {y_t - \hat{y}_t} {\hat{\sigma}}$ from this model, and the corresponding outlier p-values:

```{r echo=FALSE, results='hide'}
daytots$predicted_total <- predict(daylm, data = daytots)
daytots[, resids := total - predicted_total]
daytots[, std_resids := resids / sd(resids)]
daytots[, outlier_p_values := 2 * pnorm(-abs(std_resids))]
ggplot(daytots) +
  aes(x = outlier_p_values) +
  geom_histogram(binwidth = 0.05) +
  ggtitle("Histogram of P-Values") +
  xlab("P-Value") + ylab("Count") +
  theme(axis.text.x = element_text(colour = "black", size = rel(1.2)),
        axis.text.y = element_text(colour = "black", size = rel(1.2)),
        axis.title.x = element_text(colour = "black", size = rel(1.2)),
        axis.title.y = element_text(colour = "black", size = rel(1.2)),
        plot.title = element_text(colour = "black", size = rel(1.2)))
```

The null hypothesis here for each day is **that day's actual bike rental total is indeed generated by the normal distribution $\mathcal N(\mathbf E[y_t|\mathbf x_t], \sigma^2)$ (which is being approximated by $\mathcal N(\hat{y}_t, \hat{\sigma}^2)$)**. A low p-value indicates that the probability of this being true under the null hypothesis is low, i.e. the actual bike rental total is unlikely to have been generated by the concerned normal distribution.


## QUESTION 1.4:

```{r echo=FALSE}
source("fdr.R")
alpha <- fdr_cut(daytots$outlier_p_values, 0.05)
```

The p-value rejection cut-off associated with a 5% False Discovery Rate here is **`r formatC(alpha, format = "g")`**. The days that are in this rejection region are:

```{r echo=FALSE}
outliers <- daytots[outlier_p_values <= alpha, ]
outliers[, .(dteday, weekday, holiday, season, predicted_total = round(predicted_total), total)]
```

The latter two outliers correspond to the dates on which Hurricane Sandy hit the Northeastern U.S. With a hurricane capable of blowing anything in its path up to the sky, it is probably even surprising that on October 29, 2012 there were still people renting bicycles... 


## QUESTION 1.5:

We refer to the histogram of the outlier p-values plotted under Question 1.4.

Under the null hypothesis that the actual rental bike totals are generated by the normal distribution specified earlier, the p-values ought to be uniformly distributed over the (0, 1) interval.

The plotted histogram looks somewhat uniform, although certainly not perfect, suggesting that modelling the *total* variable on a linear scale as a normally-distributed variable is probably reasonable. This is justified by the following histogram of the *total* variable, which looks sufficiently symmetric:

```{r echo=FALSE}
ggplot(daytots) +
  aes(x = total) +
  geom_histogram() +
  ggtitle("Histogram of Daily Totals") +
  xlab("Daily Total") + ylab("Count") +
  theme(axis.text.x = element_text(colour = "black", size = rel(1.2)),
        axis.text.y = element_text(colour = "black", size = rel(1.2)),
        axis.title.x = element_text(colour = "black", size = rel(1.2)),
        axis.title.y = element_text(colour = "black", size = rel(1.2)),
        plot.title = element_text(colour = "black", size = rel(1.2)))
```


# QUESTION 2: LASSO Linear Regression and Model Selection


## Question 2.1:

We now consider a cross-validated LASSO regression of the log of the *cnt* variable on other variables plus interactions between *yr* and *mth* and between *hr* and *notbizday*.

```{r echo=FALSE}
source("naref.R")
mmbike <- sparse.model.matrix(
  cnt ~ . + yr*mnth + hr*notbizday, 
	data=naref(biketab))[,-1]
y <- log(biketab$cnt)
n = length(y)
null_deviance = n * var(y)
fitlin <- gamlr(mmbike, y, lmr=1e-5)
cv.fitlin <- cv.gamlr(mmbike, y, lmr=1e-5, cl = cl)
```

The response variable log*(cnt)* means that we are considering multiplicative changes instead of linear changes in *cnt*. This is advisable given that the *cnt* variable at the hourly granularity, unlike the daily *total* in Question 1, displays a clear exponential pattern as presented below:

```{r echo=FALSE}
ggplot(biketab) +
  aes(x = cnt) +
  geom_histogram() +
  ggtitle("Histogram of Hourly Bike Rental Counts") +
  xlab("Hourly Count") + ylab("Count") +
  theme(axis.text.x = element_text(colour = "black", size = rel(1.2)),
        axis.text.y = element_text(colour = "black", size = rel(1.2)),
        axis.title.x = element_text(colour = "black", size = rel(1.2)),
        axis.title.y = element_text(colour = "black", size = rel(1.2)),
        plot.title = element_text(colour = "black", size = rel(1.2)))
```

The model matrix is a sparse matrix, with columns of factor variables *season*, *yr*, *mnth*, *hr*, *holiday*, *weekday*, *notbizday* and *weathersit*, plus those of the interaction terms, having values of 1 only when the concerned factor value is "on" for a certain case, and having values 0 everywhere else.

This model addresses the outliers detected in Questions 1 by including variables indicating weather conditions and holidays, which were not considered by the previous model.


## QUESTION 2.2:

```{r echo=FALSE}
plot(cv.fitlin)
title("Model Selection by CV")
```

In the cross-validated model, the model selection criterion **select="min"** chooses the LASSO regularization parameter $\lambda$ that minimizes the average out-of-sample residual deviance. In this case, the estimated average out-of-sample deviance is `r formatC(cv.fitlin$cvm[cv.fitlin$seg.min], format = "g")`, corresponding to estimated *R*^2^ of **`r formatC(1 - cv.fitlin$cvm[cv.fitlin$seg.min]/var(y), format = "g")`**.

If we choose the model selection criterion **select="1se"** instead, we'll end up with a simpler model (regularized by a larger $\lambda$) whose average out-of-sample residual deviance not more than 1 standard error away from the minimum. In this case, the estimated average out-of-sample deviance is `r formatC(cv.fitlin$cvm[cv.fitlin$seg.1se], format = "g")`, corresponding to estimated *R*^2^ of **`r formatC(1 - cv.fitlin$cvm[cv.fitlin$seg.1se]/var(y), format = "g")`**.


## QUESTION 2.3:

The below plots and table compares model selection by information criteria AICc, AIC and BIC and cross valiation criteria "min" and "1se":

```{r echo=FALSE}
t <- data.frame(DEVIANCE=rep(0, 5), 
                R2=rep(0, 5),
                AICc=rep(NA, 5),
                AIC=rep(NA, 5),
                BIC=rep(NA, 5),
                row.names=c("AICc Model", "AIC Model", "BIC Model", 
                            "CV.min Model", "CV.1se Model"))
t["AICc Model", "DEVIANCE"] = deviance(fitlin)[which.min(AICc(fitlin))] / n
t["AIC Model", "DEVIANCE"] = deviance(fitlin)[which.min(AIC(fitlin))] / n
t["BIC Model", "DEVIANCE"] = deviance(fitlin)[which.min(BIC(fitlin))] / n
t["CV.min Model", "DEVIANCE"] = cv.fitlin$cvm[cv.fitlin$seg.min]
t["CV.1se Model", "DEVIANCE"] = cv.fitlin$cvm[cv.fitlin$seg.1se]

t$R2 = 1 - t$DEVIANCE / var(y)

t["AICc Model", "AICc"] = AICc(fitlin)[which.min(AICc(fitlin))] / n
t["AIC Model", "AICc"] = AICc(fitlin)[which.min(AIC(fitlin))] / n
t["BIC Model", "AICc"] = AICc(fitlin)[which.min(BIC(fitlin))] / n
t["CV.min Model", "AICc"] = AICc(cv.fitlin$gamlr)[cv.fitlin$seg.min] / n
t["CV.1se Model", "AICc"] = AICc(cv.fitlin$gamlr)[cv.fitlin$seg.1se] / n

t["AICc Model", "AIC"] = AIC(fitlin)[which.min(AICc(fitlin))] / n
t["AIC Model", "AIC"] = AIC(fitlin)[which.min(AIC(fitlin))] / n
t["BIC Model", "AIC"] = AIC(fitlin)[which.min(BIC(fitlin))] / n
t["CV.min Model", "AIC"] = AIC(cv.fitlin$gamlr)[cv.fitlin$seg.min] / n
t["CV.1se Model", "AIC"] = AIC(cv.fitlin$gamlr)[cv.fitlin$seg.1se] / n

t["AICc Model", "BIC"] = BIC(fitlin)[which.min(AICc(fitlin))] / n
t["AIC Model", "BIC"] = BIC(fitlin)[which.min(AIC(fitlin))] / n
t["BIC Model", "BIC"] = BIC(fitlin)[which.min(BIC(fitlin))] / n
t["CV.min Model", "BIC"] = BIC(cv.fitlin$gamlr)[cv.fitlin$seg.min] / n
t["CV.1se Model", "BIC"] = BIC(cv.fitlin$gamlr)[cv.fitlin$seg.1se] / n

t
```

```{r echo=FALSE}
#plot CV results and the various IC
par(mfrow=c(1,2))
log_lambda <- log(fitlin$lambda) ## the sequence of lambdas

plot(log_lambda, BIC(fitlin)/n, pch=21, bg="green",
     xlab="log lambda", ylab="IC/n", ylim = c(-2, 1))
abline(v=log_lambda[which.min(AIC(fitlin))], col="orange", lty=3)
abline(v=log_lambda[which.min(BIC(fitlin))], col="green", lty=3)
abline(v=log_lambda[which.min(AICc(fitlin))], col="black", lty=3)
points(log_lambda, AIC(fitlin)/n, pch=21, bg="orange")
points(log_lambda, AICc(fitlin)/n, pch=21, bg="black")
legend("topleft", bty="n",
  fill=c("black","orange","green"),legend=c("AICc","AIC","BIC"))
title("Model Selection by IC")

plot(cv.fitlin)
title("Model Selection by CV")
```

```{r echo=FALSE}
# all metrics, together in a path plot.
par(mfrow=c(1,1))
plot(fitlin, col="grey")
abline(v=log_lambda[which.min(AICc(fitlin))], col="black", lty=2)
abline(v=log_lambda[which.min(AIC(fitlin))], col="orange", lty=2)
abline(v=log_lambda[which.min(BIC(fitlin))], col="green", lty=2)
abline(v=log(cv.fitlin$lambda.min), col="blue", lty=2)
abline(v=log(cv.fitlin$lambda.1se), col="purple", lty=2)
legend("topright", bty="n", lwd=1, 
  col=c("black","orange","green","blue","purple"),
	legend=c("AICc","AIC","BIC","CV.min","CV.1se"))
title("Comparison of all 5 Model Selection Methods")
```


## QUESTION 2.4:

It seems from the table above (Question 2.3) that the five model selection criteria select models with very similar performances. I decide to go for the one selected by the AICc criterion.

Let's look at the largest three *dteday* effects in the model:

```{r echo=FALSE, warning=FALSE, message=FALSE}
lin_betas_AICc <- coef(fitlin)
lin_betas_AICc_dteday <- lin_betas_AICc[grepl("dteday", rownames(lin_betas_AICc))]
top_3 <- order(abs(lin_betas_AICc_dteday), decreasing=TRUE)[1:3]
data.frame(date=levels(biketab$dteday)[top_3], date_effect=lin_betas_AICc_dteday[top_3])
```

On this three dates, the log of the hourly bike rental totals was down by about 1.0, meaning the number of bikes rented on average per hour was only about by `r formatC(1 / exp(1.0), format = "g")` as many was the baseline. This is easy to explain, as two of the dates corresponded to Christmas/Boxing Day festivities while there was a severe snow storm (the "2011 Halloween nor'easter") affecting Northeastern U.S. on Oct 29, 2011. 


## QUESTION 2.5:

We now bootstrap the regularization parameters $\lambda$ selected by the AICc and BIC criteria.

```{r echo=FALSE}
gamma_AICc <- c()
gamma_BIC <- c()
for (b in 1 : 30)
{
  ib <- sample(1 : n, n, replace=TRUE)
  mmbike_b <- mmbike[ib, ]
  y_b <- y[ib]
  fitlin_b <- gamlr(mmbike_b, y_b, lmr=1e-5)
  gamma_AICc <- fitlin_b$lambda[which.min(AICc(fitlin_b))]
  gamma_BIC <- fitlin_b$lambda[which.min(BIC(fitlin_b))]
}
```

The $\lambda$'s selected by AICc have a mean of `r formatC(mean(gamma_AICc), format = "g")`. The $\lambda$'s selected by BIC have a mean of `r formatC(mean(gamma_BIC), format = "g")`.

We can see clearly that BIC tends to select a larger $\lambda$ because its penalization of the coefficients is stricter than that by AICc.


# QUESTION 3. Logistic Regression and Classification

## QUESTION 3.1:

We now consider a logistic regresssion of *overload* on the same dependent variables as under Question 2.

```{r}
overload <- biketab$cnt > 500
fitlog <- gamlr(mmbike, overload, family = "binomial", lmr=1e-3)
plot(fitlog)
title("LASSO regularization path of Logisic Regression")
```


# QUESTION 3.2:

The hour-of-day effects on *overload* on business days are as follows:

```{r warning=FALSE, message=FALSE, echo=FALSE}
par(mfrow = c(1, 2))
logistic_betas_AICc <- coef(fitlog)
row_names_to_get <- paste("hr", 0 : 23, sep="")
logistic_hour_of_biz_day_effects <-
  logistic_betas_AICc[rownames(logistic_betas_AICc) %in% row_names_to_get]
lin_hour_of_biz_day_effects <-
  lin_betas_AICc[rownames(lin_betas_AICc) %in% row_names_to_get]
row_names_to_get <- paste(row_names_to_get, ":notbizday0", sep="")
logistic_hour_of_biz_day_effects <- logistic_hour_of_biz_day_effects +
  logistic_betas_AICc[rownames(logistic_betas_AICc) %in% row_names_to_get]
lin_hour_of_biz_day_effects <- lin_hour_of_biz_day_effects +
  lin_betas_AICc[rownames(lin_betas_AICc) %in% row_names_to_get]
plot(0 : 23, logistic_hour_of_biz_day_effects, type = "l",
     xlab = "Hour", ylab = "Hour of Biz Day Effect")
title("Logistic Regression")
plot(0 : 23, lin_hour_of_biz_day_effects, type = "l",
     xlab = "Hour", ylab = "Hour of Biz Day Effect")
title("Linear Regression")
```

We can see two clear peaks at 8am and 5pm on business days. Take 5pm, for instance: the demand for rental bikes is about `r formatC(exp(lin_hour_of_biz_day_effects[18]), format = "g")` times the baseline demand, making the odds of an overload be `r formatC(exp(logistic_hour_of_biz_day_effects[18]), format = "g")` times such odds in the baseline.


## Question 3.3:

If it costs $200/hr in overtime pay if we have an overload, and staffing an extra driver to move the bikes costs only $100/hr, then the we should staff an extra driver if the expected cost of overload exceeds the cost of staffing the driver, i.e. when probability of overload is greater than $100 / $200 = 0.5.


## Question 3.4:

Below is the ROC curve for the AICc-selected logistic regression model:

```{r echo=FALSE}
source("roc.R")
overload_probability <- predict(fitlog, newdata = mmbike, type = "response")
roc(overload_probability, overload)
title("ROC Curve")

threshold <- 0.5
num_pos <- sum(overload == 1)
num_true_pos <- sum((overload == 1) * (overload_probability >= 0.5))
num_neg <- sum(overload == 0)
num_true_neg <- sum((overload == 0) * (overload_probability < 0.5))
```

We can see that the classifier for *overload* seems to be a very good one, with very little trade-off between sensitivity and specificity. At the decision threshold of 0.5, sensitivity is **`r formatC(num_true_pos / num_pos, format = "g")`** and specificity is **`r formatC(num_true_neg / num_neg, format = "g")`**.


## Question 3.5:

We now refit the logisic regression on a Training set and evaluate its performance on a Test set, through a Test ROC curve:

```{r echo=FALSE}
set.seed(5807) 
test <- sample(1 : n, 3000)
mmbike_train <- mmbike[-test, ]
overload_train <- overload[-test]
mmbike_test <- mmbike[test, ]
overload_test <- overload[test]
fitlog <- gamlr(mmbike_train, overload_train, family = "binomial", lmr=1e-3)
overload_probability_test <- predict(fitlog, newdata = mmbike_test, type = "response")
roc(overload_probability_test, overload_test)
title("ROC Curve (Test Set)")
```

The shape of the Test ROC is very similar to what we see in Question 3.4, implying that the model selected by AICc in this case has very good out-of-sample performance.


# QUESTION 4. Treatment Effects Estimation

## QUESTION 4.1:

Based on the "naive" model in Question 1, one standard deviation increase in hudimity decreases the log of the hourly demand by `r coef(fitlin)["hum",]`, i.e. make the demand be `r exp(coef(fitlin)["hum",])` as much as in the baseline, i.e. fall  by 5%.


## QUESTION 4.2:

We now try to estimate the independent effect of humidity. First, we fit a LASSO regression of humidity on other independent variables:

```{r}
x <- mmbike[, -grep("hum",colnames(mmbike))]
hum <- mmbike[, "hum"] # pull humidity out as a separate vector
hum_reg <- gamlr(x, hum, lmr = 1e-4)
pred_hum <- as.vector(predict(hum_reg, newdata = x))
```

This regression has a average deviance of `r formatC(deviance(hum_reg)[which.min(AICc(hum_reg))], format = "g")` and *R*^2^ of `r formatC(1 - deviance(hum_reg)[which.min(AICc(hum_reg))] / n / var(hum), format = "g")`. This suggests a lot of variation in the *hum* variable is explained by other independent variables. This is relevant because we'll need to isolate the effect of the part of *hum* not captured by the other variables.


## QUESTION 4.3:

We now fit a LASSO regression including the fitted values for *hum* from the above regression, with no penalization on the coefficient of these fitted values: 

```{r}
hum_reg_for_treatment_effect <- gamlr(cBind(pred_hum, mmbike), y,
                                      free = 1, lmr = 1e-4)
```

The effect of *hum* independent from other variables is now estimated to be `r formatC(coef(hum_reg_for_treatment_effect)["hum",], format = "g")`, slightly different from the coefficient in the "naive" model.


## QUESTION 4.4:

We now extend the model to include the interaction between humidity and temperature:

```{r}
mmbike <- sparse.model.matrix(
  cnt ~ . + yr*mnth + hr*notbizday + hum:temp, 
  data=naref(biketab))[,-1]
fitlin <- gamlr(mmbike, y, lmr=1e-5)
```

The effect of one standard deviation increase *hum* on the log of rental bike demand is now `r formatC(coef(fitlin)["hum",], format = "g")` + `r formatC(coef(fitlin)["temp:hum",], format = "g")` *temp*. This effect is a positive linear function of temperature, which means, the hotter it is, the more positive / less negative the effect of humidity on rental bike demand becomes.

## QUESTION 4.5:

We now try to isolate the independent effect of humidity from the model in Question 4.4:

```{r}
x <- mmbike[, -grep("hum",colnames(mmbike))]
hum <- mmbike[, "hum"] # pull humidity out as a separate vector
hum_reg <- gamlr(x, hum, lmr = 1e-4)
pred_hum <- as.vector(predict(hum_reg, newdata = x))
pred_hum_times_temp <- pred_hum * biketab$temp
hum_reg_for_treatment_effect <- gamlr(cBind(pred_hum, pred_hum_times_temp, mmbike), y,
                                      free = 1 : 2, lmr = 1e-4)
```

The independent effect of one standard deviation increase *hum* on the log of rental bike demand is `r formatC(coef(hum_reg_for_treatment_effect)["hum",], format = "g")` + `r formatC(coef(hum_reg_for_treatment_effect)["temp:hum",], format = "g")` *temp*, which is now not dependent on *temp*.

```{r echo=FALSE}
stopCluster(cl)
```