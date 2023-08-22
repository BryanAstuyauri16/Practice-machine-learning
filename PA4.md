---
title: 'Testing models to predict the manner in which people do exercise'
output: 
  #pdf_document: default
  html_document:
    keep_md: yes
always_allow_html: yes
---
### BACKGROUND
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: 
http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har
 (see the section on the Weight Lifting Exercise Dataset).


### Data Cleaning
Importing the required libraries

```r
library(caret)
library(rattle)
library(randomForest)
library(gbm)
library(kableExtra)
```
Importing the data

```r
training <- read.csv("pml-training.csv")
testing <- read.csv("pml-testing.csv")
```
<!-- ![plot of chunk histogram](1.png) -->
Removing near zero variance columns

```r
nearZeroVar <- nearZeroVar(training)
training <- training[, -nearZeroVar]
testing <- testing[, -nearZeroVar]
```
Removing columns which have more than 90% NA values

```r
Na_values <- sapply(training, function(x) mean(is.na(x))) > 0.9
training <- training[, Na_values == FALSE]
testing <- testing[, Na_values == FALSE]
```
Removing the first 6 columns which contain superfluous information

```r
training <- training[, -c(1:6)]
testing <- testing[, -c(1:6)]
```
Partitioning the training data into a train set and a test set to be able to test our model

```r
intrain <- createDataPartition(training$classe, p = 0.6, list = FALSE)
training_set <- training[intrain, ]
testing_set <- training[-intrain, ]
```
### Buildings models
We are going to create a desicion three, random forest and Generalized boosted regression models

```r
model1 <- train(classe ~., data = training_set, method = 'rpart')
model2 <- train(classe ~., data = training_set, method = "rf", ntree = 5)
model3 <- train(classe ~., data = training_set, method = "gbm", verbose = FALSE, 
                trControl = trainControl(method = "CV", number = 2, allowParallel = TRUE))
```
testing our models on testing_set

```r
pred1 <- predict(model1, testing_set)
pred2 <- predict(model2, testing_set)
pred3 <- predict(model3, testing_set)
```
Visualizing the tree model 

```r
fancyRpartPlot(model1$finalModel)
```

![](PA4_files/figure-html/unnamed-chunk-9-1.png)<!-- -->

building the confusion matrix for the models

```r
CM1 <- confusionMatrix(pred1, as.factor(testing_set$classe))
CM2 <- confusionMatrix(pred2, as.factor(testing_set$classe))
CM3 <- confusionMatrix(pred3, as.factor(testing_set$classe))
```
Visualizing the accuracy

```r
df <- data.frame(CM1$overall, CM2$overall, CM3$overall)
colnames(df) <- c("Model 1", "Model 2", "Model 3")
kable(df, 'html') %>% kable_styling(full_width = F) 
```

<table class="table" style="width: auto !important; margin-left: auto; margin-right: auto;">
 <thead>
  <tr>
   <th style="text-align:left;">   </th>
   <th style="text-align:right;"> Model 1 </th>
   <th style="text-align:right;"> Model 2 </th>
   <th style="text-align:right;"> Model 3 </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> Accuracy </td>
   <td style="text-align:right;"> 0.4984706 </td>
   <td style="text-align:right;"> 0.9747642 </td>
   <td style="text-align:right;"> 0.9585776 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Kappa </td>
   <td style="text-align:right;"> 0.3448667 </td>
   <td style="text-align:right;"> 0.9680713 </td>
   <td style="text-align:right;"> 0.9475789 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> AccuracyLower </td>
   <td style="text-align:right;"> 0.4873456 </td>
   <td style="text-align:right;"> 0.9710487 </td>
   <td style="text-align:right;"> 0.9539312 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> AccuracyUpper </td>
   <td style="text-align:right;"> 0.5095966 </td>
   <td style="text-align:right;"> 0.9781211 </td>
   <td style="text-align:right;"> 0.9628794 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> AccuracyNull </td>
   <td style="text-align:right;"> 0.2844762 </td>
   <td style="text-align:right;"> 0.2844762 </td>
   <td style="text-align:right;"> 0.2844762 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> AccuracyPValue </td>
   <td style="text-align:right;"> 0.0000000 </td>
   <td style="text-align:right;"> 0.0000000 </td>
   <td style="text-align:right;"> 0.0000000 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> McnemarPValue </td>
   <td style="text-align:right;"> NaN </td>
   <td style="text-align:right;"> 0.0000000 </td>
   <td style="text-align:right;"> 0.0000000 </td>
  </tr>
</tbody>
</table>
### Predicting classe variable
Looking at the table above we can see that the random forest model provides the better results. Testing on the given test data

```r
Pred <- data.frame(predict(model2, testing))
kable(Pred, 'html') %>% kable_styling(full_width = F) 
```

<table class="table" style="width: auto !important; margin-left: auto; margin-right: auto;">
 <thead>
  <tr>
   <th style="text-align:left;"> predict.model2..testing. </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> B </td>
  </tr>
  <tr>
   <td style="text-align:left;"> A </td>
  </tr>
  <tr>
   <td style="text-align:left;"> B </td>
  </tr>
  <tr>
   <td style="text-align:left;"> A </td>
  </tr>
  <tr>
   <td style="text-align:left;"> A </td>
  </tr>
  <tr>
   <td style="text-align:left;"> E </td>
  </tr>
  <tr>
   <td style="text-align:left;"> D </td>
  </tr>
  <tr>
   <td style="text-align:left;"> B </td>
  </tr>
  <tr>
   <td style="text-align:left;"> A </td>
  </tr>
  <tr>
   <td style="text-align:left;"> A </td>
  </tr>
  <tr>
   <td style="text-align:left;"> B </td>
  </tr>
  <tr>
   <td style="text-align:left;"> C </td>
  </tr>
  <tr>
   <td style="text-align:left;"> B </td>
  </tr>
  <tr>
   <td style="text-align:left;"> A </td>
  </tr>
  <tr>
   <td style="text-align:left;"> E </td>
  </tr>
  <tr>
   <td style="text-align:left;"> E </td>
  </tr>
  <tr>
   <td style="text-align:left;"> A </td>
  </tr>
  <tr>
   <td style="text-align:left;"> B </td>
  </tr>
  <tr>
   <td style="text-align:left;"> B </td>
  </tr>
  <tr>
   <td style="text-align:left;"> B </td>
  </tr>
</tbody>
</table>

