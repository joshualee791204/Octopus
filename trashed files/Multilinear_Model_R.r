library(dplyr)
library(ggplot2)
library(car)
library(mice)

raw_df <- read.csv('train.csv', stringsAsFactors = TRUE, header = TRUE)
#create a copy of data frame with different memory location. Thus no mutation
df <- cbind(raw_df)
View(raw_df)
df['SalePrice']=log(df['SalePrice'])

###The following codes tried multiliean regression. with 'SalePrice', 'OverallQual', 'X1stFlrSF',
##'GrLivArea', 'YearBuilt' as variables
#select colums
df <- df %>% select(price='SalePrice', overall_qual='OverallQual', area_f1 = 'X1stFlrSF', area_all = 'GrLivArea', ms_zone = 'MSZoning', year='YearBuilt')
library(car)
scatterplotMatrix(~price + overall_qual + area_f1 + area_all + ms_zone + year, data=df,
                  col=c('purple','blue','black','red', 'orange', 'black'))
multi_ml <- lm(price ~., data = df)
mean((predict(multi_ml, data=df)-df$price)^2)
#mean square error
mean((exp(predict(multi_ml, data=df))-raw_df$SalePrice)^2)
summary(multi_ml)
#R-square is 0.8134
influencePlot(multi_ml)
##1299 is a outlier need to be addressed.


###The following codes tried Ridge regression
library(glmnet)
grid <- exp(seq(5,-4, length=500))
x <- model.matrix(price ~., df)[,-1]
y <- df$price
rid_model <- glmnet(x, y, lambda=grid, alpha=0)

train = sample(1:nrow(x), 8*nrow(x)/10)
test = (-train)

summary(rid_model)
coef(rid_model)
plot(rid_model, xvar = "lambda", label = TRUE, main= "ridge regression")
##use cross validation to find the best lambda
cv_rid_model <- cv.glmnet(x[train,],y[train],lambda = grid, nfolds = 10, alpha=0)
best_lambda_rid <- cv_rid_model$lambda.min
best_lambda_rid
cv_rid_out <- predict(cv_rid_model, x[test,], lambda=best_lambda_rid, alpha=0)
cv_rid_out_train <- predict(cv_rid_model, x[train,], lambda=best_lambda_rid, alpha=0)
##mean sqaure 
mean((cv_rid_out_train-y[train])^2)
mean((cv_rid_out-y[test])^2)


