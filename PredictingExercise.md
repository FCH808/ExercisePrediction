Predicting How Well Exercises are Done
========================================================


Many machine learning datasets attempt to predict what action is being performed by participants (Human Activity Recognition) through the use of devices like the Jawbone Up, Nike Fuelband, Fitbit, and even the ubiquitous smartphone. [1] A study by Velloso et al. asks the question of not whether we can predict which activity is being performed, but how well it is being done. [2] [3]

This paper attempts to create a fully reproducible multi-class classification supervised machine learning algorithm that accurately predicts if an exercise is being performed correctly using data from the Vellesco et al study. 

The original study took readings from several wearable devices containing sensors including belts, gloves, arm-bands, and dumbbells. Participants were instructed to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different ways. These are listed below.

A: exactly according to the specification

B: throwing the elbows to the front

C: lifting the dumbbell only halfway

D: lowering the dumbbell only halfway

E: and throwing the hips to the front

The exercises were performed by six male participants aged 20-28 years who had little weight lifting experience.

The goal of this paper is to correctly predict which way (how well) the participant was performing the exercise based solely on the sensor readings.

## Data Preparation

The data were downloaded via links provided by the Practical Machine Learning course taught by Jeff Leek via the Coursera platform.[4] The data is also available from the original author's webpage. [3]

Many entries in the summary statistics columns had #DIV/0! values. These were determined to be NA values trying to divide by 0 to get an average of a non-existant number. This is a typical string entry in Excel this is attempted. They are set to NA within the read.csv call.

The data had one training set with 19,622 observations and 160 columns of variable measurements with the "classe" variable (dependent variable) provided, and one hold-out test of 20 observations to be predicted and submitted.  The building set 

The training set was initially randomly split into a model building set (70%; 13,375 observations) and a hold-out validation set (30%; 5,887 observations). 

The building set was further randomly split into a training set (70%; 9,615 observations) and a testing set (30%; 4,120 observations). 

Timestamp variables were removed since exercises were done sequentially and these are 100% predictors on this training set (and the test set), but would not be accurate for OOB predictions if exercises are done in different orders or for different periods of time. Username was also excluded right away.

Columns of variable with near zero variance were then removed as they had no predictive value without any variance.

Columns with NA's had over 9000 NA's entries. These seemed to be summary statistics from the last num_ window chunks of time, signified by a "yes" from the new_ window column. These wouldn't be amenable to imputation (they are just derived from the preceding block of time). They are removed for this single-time-point prediction modeling.

The remaining variable (excluding the classe variable) were then checked for multi-colinearity with correlations over an 80% threshold. The extraneous columns were then removed. 

This brought the predictor variable count to 39, along with the classe variable.

Box-Cox power transformations were then applied to stabilize variance. Data was also centered and scaled at this point.

One set of data were saved at this point for later use in a random forest model with pre-Principal Component Analysis (PCA) data. The data were then processed through PCA for use in post-PCA model building.

## Model building

Four models in total were trained on the data. One random forest model (RF) was trained on the pre-PCA data, the other three were trained on the post-PCA data. They PCA models included RF with default settings, one RF model with 10-fold repeated cross-validation with 3 repetitions, and one gradient boosting machine (GBM) model.

## Predicting

Each of these models were trained on the training set and used to predict on the test set. The pre-PCA RF model showed 99% accuracy, both post-PCA RF models showed 96.9% accuracy, and the GBM model showed 81.6% accuracy.

## More model building (blending)

Probabilities for each models' predictions were then into a separate matrix, then combined into one data.frame along with the classe variable of the test set. This data.frame was used to blend all predictions into one predicting model for the validation set. 

The random forest model was used to combine all predictions since it showed very high accuracy. A final blending model was trained on the test set probabilities predicting its test set classe variable. This model achieved 100% accuracy in predicting its values. 

## Predicting Validation

The validation set was then predicted **once** by each of the 4 training set models. These probabilities were combined into a data.frame and used to predict **once** by the blending model built on the test set. 

### Out-of-bag Estimate.

This resulted in 99.2% accuracy on the validation set with an out-of-bag estimate of error rate of **0.8%**. 
This seemed highly accurate for a model built on blended predictions.

## Final build/test

The final model was then built and run once on the actual test set. The work flow was changed slightly since there was no validation set.

The final model was created by the 4 models predicting the outcomes of the training set, then using those training set probabilities to train the blended model.

The test set was predicted by each of those 4 training set models, then all the probabilities were run through that final training set blending model for the final prediction.

## Final Prediction Accuracy
All 20 unknown predictions were submitted and confirmed correct. 

The out-of-bag estimate of 0.8% help up for predicting 20 cases. If we make 100's or 1000's of predictions, we can expect about 0.8% to be misclassified.

With all predictions confirmed correct externally with a one-time submission, post mortem analysis could then be performed on each model to assess the how each model fared.

## Post-mortem Analysis

Interestingly, besides the blended model, only the **pre-PCA** random forest model correctly predicted all 20 unknown cases (100% accuracy.) Both PCA processed random forest models had trouble with question 3 (95% accuracy each), while the GBM model had trouble with multiple misses (75% accuracy.)


## Final model performance 

Performance on unknown 20 predictions.

### RF default options on PCA processed data.

Accuracy : 0.95 
95% CI : (0.751, 0.999)

### RF repeated cross-validation 10 folds, 3 repeats on PCA processed data.

Accuracy : 0.95 
95% CI : (0.751, 0.999)

### RF default options on Box-Cox transformed, scaled, and centered data but no PCA transformation.

Accuracy : 1

95% CI : (0.832, 1)

### GBM model on PCA processed data.

Accuracy : 0.75

95% CI : (0.509, 0.913)
 
### Final blended model accuracy.

Accuracy : 1

95% CI : (0.832, 1)

No. of variables tried at each split: 2

OOB estimate of  error rate: 0%

## Conclusion

In this case, reducing the dataset to principal components negatively impacted prediction performance. Model blending turned out to be good for coming to an agreement between models on predicting all 20 test cases. 

Random forest performed well in blending the models together as simple majority voting would have also chosen the wrong prediction for prediction #3 of 20. 

I would love to try other blending techniques and incorporating more models on slightly more intractable problems as this one seems to be modeled quite well by default-setting random forest modeling once the data have been processed a little bit (but not too much!) 


_Model plots and output are included in the source code but are not considered as a required part of this report write-up._
__________________________________________________________________________________________________________________________
### References

[1] Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. Human Activity Recognition on Smartphones using a Multiclass Hardware-Friendly Support Vector Machine. International Workshop of Ambient Assisted Living (IWAAL 2012). Vitoria-Gasteiz, Spain. Dec 2012 https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones

[2] Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.

[3] http://groupware.les.inf.puc-rio.br/har

[4] https://class.coursera.org/predmachlearn-002
__________________________________________________________________________________________________________________________

## Source Code


```r
library(data.table)

setwd("C:/Users/fch80_000/Dropbox/~Coursera/DataScienceSpec/Practical-Machine-Learning/Project")

# Download data files if the required files do not exist in the current working directory
if (! file.exists("pml-training.csv")) {
    print("Downloading training set data file")
    download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", "pml-training.csv")
}

if (! file.exists("pml-testing.csv")) {
  print("Downloading testing set data file")
  download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", "pml-testing.csv")
}
```

Many entries in the summary statistics columns had #DIV/0! values. These were determined to be NA values trying to divide by 0 to get an average of a non-existant number. This is a typical string entry in Excel this is attempted. They are set to NA within the read.csv call.


```r
train<-read.csv("pml-training.csv", na.strings = c("NA", "#DIV/0!"))
test<- read.csv("pml-testing.csv", na.strings = c("NA", "#DIV/0!"))
```



```r
library(caTools)
set.seed(808)
inBuild<- sample.split(train$classe, SplitRatio = .70)

validation<- train[!inBuild,]
buildData<- train[inBuild,]

set.seed(809)
inTrain<- sample.split(buildData$classe, SplitRatio = .70)

training<- buildData[inTrain,]
testing<- buildData[!inTrain,]
```




```r
names(training)[1:7]
```

```
## [1] "X"                    "user_name"            "raw_timestamp_part_1"
## [4] "raw_timestamp_part_2" "cvtd_timestamp"       "new_window"          
## [7] "num_window"
```

```r
##Remove timestamp variables since exercises were done sequentially and these are 100% predictors on this training set (and the test set..), but will not be accurate for OOB predictions if exercises are done in different orders or for different periods of time.
reducedTraining<- subset(training, select = -c(X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window))
reducedTesting<- subset(testing, select = -c(X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window))
reducedValidation<- subset(validation, select = -c(X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window))
```



```r
# Find variables with both zero variance ratios and zero variability. Dependent variable isnt removed since if it has near zero variability, it can be predicted without any modeling!
library(caret)

nzv <- nearZeroVar(reducedTraining)

# Use training ZeroVariance info to remove those variables(columns) from both the training and testing datasets. Reduced to 120 variables + classe.
filteredTraining<- reducedTraining[,-nzv] 
filteredTesting<- reducedTesting[,-nzv] 
filteredValidation<- reducedValidation[,-nzv] 
```



```r
## Columns with NA's have over 9000 NA's. These seem to be summary statistics from the last num_window chunks of time, signified by a "yes" from the new_window column. These wouldn't be amenable to imputation (they are just derived from the preceding block of time). They are removed for this single-time-point prediction modeling. 
#summary(filteredTraining)
#Not included since it is very verbose.
```


```r
## This function will return the column indices which contain NA's.
nacols <- function(df) {
  colnames(df)[unlist(lapply(df, function(x) any(is.na(x))))]
}

#Indices are made from the training data and applied to the testing and validation data. 
NAcols<-nacols(filteredTraining)

# The NA columns are filtered out 
filteredTraining<- filteredTraining[ , -which(names(filteredTraining) %in% NAcols)]
filteredTesting<- filteredTesting[ , -which(names(filteredTesting) %in% NAcols)]
filteredValidation<- filteredValidation[ , -which(names(filteredValidation) %in% NAcols)]
```



```r
## Find multicolinear variables. Use absolute values to find both highly positive and negative correlations. Remove the dependent classe varaible in the cor().
M<- abs(cor(filteredTraining[,-53]))
## All variables are correlated 100% with themselves. Set these to 0.
diag(M)<- 0
## Find all variables that over the threshold of 80%
which(M > 0.8, arr.ind = TRUE)
```

```
##                  row col
## yaw_belt           3   1
## total_accel_belt   4   1
## accel_belt_y       9   1
## accel_belt_z      10   1
## accel_belt_x       8   2
## magnet_belt_x     11   2
## roll_belt          1   3
## roll_belt          1   4
## accel_belt_y       9   4
## accel_belt_z      10   4
## pitch_belt         2   8
## magnet_belt_x     11   8
## roll_belt          1   9
## total_accel_belt   4   9
## accel_belt_z      10   9
## roll_belt          1  10
## total_accel_belt   4  10
## accel_belt_y       9  10
## pitch_belt         2  11
## accel_belt_x       8  11
## gyros_arm_y       19  18
## gyros_arm_x       18  19
## magnet_arm_x      24  21
## accel_arm_x       21  24
## magnet_arm_z      26  25
## magnet_arm_y      25  26
## accel_dumbbell_x  34  28
## accel_dumbbell_z  36  29
## gyros_dumbbell_z  33  31
## gyros_forearm_y   45  31
## gyros_forearm_z   46  31
## gyros_dumbbell_x  31  33
## gyros_forearm_y   45  33
## gyros_forearm_z   46  33
## pitch_dumbbell    28  34
## yaw_dumbbell      29  36
## gyros_dumbbell_x  31  45
## gyros_dumbbell_z  33  45
## gyros_forearm_z   46  45
## gyros_dumbbell_x  31  46
## gyros_dumbbell_z  33  46
## gyros_forearm_y   45  46
```



```r
#Find correlated predictor variables correlated over |0.8| , and remove. Use same indices from train toward test/validation.
descrCor <- cor(filteredTraining[,-53])
highlyCorDescr <- findCorrelation(descrCor, cutoff = 0.8)


filteredTraining <- filteredTraining[, -highlyCorDescr]
filteredTesting <- filteredTesting[, -highlyCorDescr]
filteredValidation <- filteredValidation[, -highlyCorDescr]

## This reduces the predictor variables to 39.
names(filteredTraining)[40]
```

```
## [1] "classe"
```



```r
## Box-Cox power transformation is applied to stabilize variance.
## 52 remaining predictor variables are centered and scaled. 
## The operations are applied in this order: Box-Cox/Yeo-Johnson transformation, centering, scaling, range, imputation, PCA, ICA then spatial sign. 
## There are no NA's to impute in this data set.
preObj<- preProcess(filteredTraining[,-40] , method = c("BoxCox","center", "scale"))
```



```r
## predict test and training scaled and centered values with the same training set settings/object
trainPred<- predict(preObj, filteredTraining[,-40])
testPred<- predict(preObj, filteredTesting[,-40])
validationPred<- predict(preObj, filteredValidation[,-40])

#*#*#*#*#
### A training/test/validation set from here will be called to a later random forest model. This will be to create one random forest model with pre-PCA data, and one with post-PCA data to blend together.
#*#*#*#*#

#Training set should be scaled to mean 0 and sd of 1, testing set should be somewhat close. summary output is verbose, but this is confirmed.

#summary(trainPred)
#summary(testPred)
#summary(validationPred)
```




```r
## Apply PCA to the new scaled datasets. Caret had a wildly different result when adding this in the earlier call with Box-Cox, centering, and scaling, so this was separated.
prComp<- preProcess(trainPred, method = "pca")
predPCA<- predict(prComp, trainPred)

#predict test PCA with training set values. DO NOT retrain using test variables.
testPredPCA<- predict(prComp, testPred)
validationPredPCA<- predict(prComp, validationPred)
```



```r
## Fit one RF model with the PCA transformed data.
set.seed(808)
rfFit<- train(filteredTraining$classe ~ ., method = "rf", data = predPCA, trControl = trainControl(savePredictions = TRUE))
```

```
## Loading required package: randomForest
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
rfFitCF<- confusionMatrix(filteredTesting$classe, predict(rfFit, testPredPCA))
rfFitCF
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1161    4    3    4    0
##          B   26  745   20    3    3
##          C    4    9  691   11    4
##          D    3    2   22  647    1
##          E    2    2   10    3  740
## 
## Overall Statistics
##                                         
##                Accuracy : 0.967         
##                  95% CI : (0.961, 0.972)
##     No Information Rate : 0.29          
##     P-Value [Acc > NIR] : < 2e-16       
##                                         
##                   Kappa : 0.958         
##  Mcnemar's Test P-Value : 0.000786      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.971    0.978    0.926    0.969    0.989
## Specificity             0.996    0.985    0.992    0.992    0.995
## Pos Pred Value          0.991    0.935    0.961    0.959    0.978
## Neg Pred Value          0.988    0.995    0.984    0.994    0.998
## Prevalence              0.290    0.185    0.181    0.162    0.182
## Detection Rate          0.282    0.181    0.168    0.157    0.180
## Detection Prevalence    0.284    0.193    0.175    0.164    0.184
## Balanced Accuracy       0.983    0.981    0.959    0.980    0.992
```

```r
ggplot(rfFit)
```

![plot of chunk unnamed-chunk-13](figure/unnamed-chunk-131.png) 

```r
rfFit$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 4.11%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 2687   22   12   10    3     0.01719
## B   48 1763   38    5    7     0.05266
## C   10   44 1588   25    9     0.05251
## D    8    5   81 1472   10     0.06599
## E    2   13   26   17 1710     0.03281
```

```r
plot(rfFit$finalModel)
```

![plot of chunk unnamed-chunk-13](figure/unnamed-chunk-132.png) 


```r
## Fit one random forest model using repeated cross-validation 10 folds, repeated 3 times.
set.seed(808)
rfFit1<- train(filteredTraining$classe ~ .,
               method = "rf",
               data = predPCA,
               trControl = trainControl(savePredictions = TRUE,
                                        method = "repeatedcv",
                                        number = 10,
                                        repeats = 3,
                                        allowParallel = TRUE))

rfFitCF1<- confusionMatrix(filteredTesting$classe, predict(rfFit1, testPredPCA))
rfFitCF1
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1161    3    3    5    0
##          B   21  749   22    3    2
##          C    2    8  692   12    5
##          D    4    1   20  650    0
##          E    2    2   10    3  740
## 
## Overall Statistics
##                                         
##                Accuracy : 0.969         
##                  95% CI : (0.963, 0.974)
##     No Information Rate : 0.289         
##     P-Value [Acc > NIR] : < 2e-16       
##                                         
##                   Kappa : 0.961         
##  Mcnemar's Test P-Value : 0.000853      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.976    0.982    0.926    0.966    0.991
## Specificity             0.996    0.986    0.992    0.993    0.995
## Pos Pred Value          0.991    0.940    0.962    0.963    0.978
## Neg Pred Value          0.990    0.996    0.984    0.993    0.998
## Prevalence              0.289    0.185    0.181    0.163    0.181
## Detection Rate          0.282    0.182    0.168    0.158    0.180
## Detection Prevalence    0.284    0.193    0.175    0.164    0.184
## Balanced Accuracy       0.986    0.984    0.959    0.979    0.993
```

```r
ggplot(rfFit1)
```

![plot of chunk unnamed-chunk-14](figure/unnamed-chunk-141.png) 

```r
rfFit1$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 4.26%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 2685   26   10   11    2     0.01792
## B   44 1770   38    3    6     0.04890
## C    7   49 1577   35    8     0.05907
## D    5    5   82 1472   12     0.06599
## E    2   14   28   23 1701     0.03790
```

```r
plot(rfFit1$finalModel)
```

![plot of chunk unnamed-chunk-14](figure/unnamed-chunk-142.png) 


```r
## Fit one random forest model with training data before PCA transformation.
set.seed(808)
rfFitCFnoPCA<- train(filteredTraining$classe ~., 
                     method = "rf",
                     data = trainPred,
                     trControl = trainControl(savePredictions = TRUE,
                                              allowParallel = TRUE))

ggplot(rfFitCFnoPCA)
```

![plot of chunk unnamed-chunk-15](figure/unnamed-chunk-151.png) 

```r
rfFitCFnoPCA$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 1.45%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 2729    3    1    0    1    0.001829
## B   21 1812   24    2    2    0.026330
## C    2   27 1642    5    0    0.020286
## D    0    0   36 1534    6    0.026650
## E    0    0    1    8 1759    0.005090
```

```r
plot(rfFitCFnoPCA$finalModel)
```

![plot of chunk unnamed-chunk-15](figure/unnamed-chunk-152.png) 


```r
## Fit one gradient boosting machine (GBM) model with post-PCA data.
set.seed(808)
gbmFit1 <- train(filteredTraining$classe ~ ., data = predPCA,
                 method = "gbm",
                 verbose = FALSE,
                 trControl = trainControl(savePredictions = TRUE))
```

```
## Loading required package: gbm
## Loading required package: survival
## Loading required package: splines
## 
## Attaching package: 'survival'
## 
## The following object is masked from 'package:caret':
## 
##     cluster
## 
## Loading required package: parallel
## Loaded gbm 2.1
## Loading required package: plyr
```

```r
gbmFitCF<- confusionMatrix(filteredTesting$classe, predict(gbmFit1, testPredPCA))
gbmFitCF
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1055   32   20   49   16
##          B   70  601   84   25   17
##          C   39   57  570   38   15
##          D   26   19   67  550   13
##          E   24   58   68   21  586
## 
## Overall Statistics
##                                         
##                Accuracy : 0.816         
##                  95% CI : (0.804, 0.828)
##     No Information Rate : 0.295         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.767         
##  Mcnemar's Test P-Value : <2e-16        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.869    0.784    0.705    0.805    0.906
## Specificity             0.960    0.942    0.955    0.964    0.951
## Pos Pred Value          0.900    0.754    0.793    0.815    0.774
## Neg Pred Value          0.946    0.950    0.930    0.961    0.982
## Prevalence              0.295    0.186    0.196    0.166    0.157
## Detection Rate          0.256    0.146    0.138    0.133    0.142
## Detection Prevalence    0.284    0.193    0.175    0.164    0.184
## Balanced Accuracy       0.914    0.863    0.830    0.884    0.928
```

```r
ggplot(gbmFit1)
```

![plot of chunk unnamed-chunk-16](figure/unnamed-chunk-161.png) 

```r
gbmFit1$finalModel
```

```
## A gradient boosted model with multinomial loss function.
## 150 iterations were performed.
## There were 24 predictors of which 24 had non-zero influence.
```

```r
plot(gbmFit1$finalModel)
```

![plot of chunk unnamed-chunk-16](figure/unnamed-chunk-162.png) 


```r
## Predict all 4 training set models onto the test set data. We will save the probabilities for each classe from model into a separate matrix, then combine them all into one dataframe along with the classe variable of the test set. This data.frame will be used to blend them all into one predicting model for the validation set.
pred1<- predict(rfFit, testPredPCA, type = "prob")
pred2<- predict(rfFit1, testPredPCA, type = "prob")
#This one uses pre-PCA data
pred3<- predict(rfFitCFnoPCA, testPred, type = "prob")
pred4<- predict(gbmFit1, testPredPCA, type = "prob")

predDF<- data.frame(pred1, pred2, pred3, pred4, classe = filteredTesting$classe)
table(predDF$classe)
```

```
## 
##    A    B    C    D    E 
## 1172  797  719  675  757
```

```r
head(predDF)
```

```
##        A     B     C     D     E   A.1   B.1   C.1   D.1   E.1   A.2 B.2
## 12 1.000 0.000 0.000 0.000 0.000 1.000 0.000 0.000 0.000 0.000 1.000   0
## 15 0.996 0.000 0.002 0.002 0.000 1.000 0.000 0.000 0.000 0.000 1.000   0
## 22 1.000 0.000 0.000 0.000 0.000 1.000 0.000 0.000 0.000 0.000 1.000   0
## 25 0.998 0.002 0.000 0.000 0.000 1.000 0.000 0.000 0.000 0.000 0.998   0
## 27 0.928 0.012 0.020 0.010 0.030 0.920 0.032 0.010 0.010 0.028 0.998   0
## 30 0.982 0.004 0.004 0.004 0.006 0.982 0.006 0.004 0.004 0.004 0.998   0
##      C.2 D.2 E.2    A.3     B.3     C.3      D.3     E.3 classe
## 12 0.000   0   0 0.9322 0.02079 0.01800 0.003067 0.02593      A
## 15 0.000   0   0 0.9326 0.02122 0.01935 0.002900 0.02393      A
## 22 0.000   0   0 0.9321 0.02361 0.01711 0.003258 0.02392      A
## 25 0.002   0   0 0.9347 0.02084 0.01716 0.003268 0.02399      A
## 27 0.002   0   0 0.9184 0.02338 0.02035 0.003362 0.03454      A
## 30 0.002   0   0 0.9204 0.02492 0.01909 0.003154 0.03241      A
```


```r
# Train one random forest model on all predicted probabilities to predict classe.
set.seed(808)
rfFitAll<- train(classe ~ ., method = "rf", data = predDF, trControl = trainControl(savePredictions = TRUE,
                                                                                    allowParallel = TRUE))
gbmFitCFAll<- confusionMatrix(predDF$classe, predict(rfFitAll, predDF))
gbmFitCFAll
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1172    0    0    0    0
##          B    0  797    0    0    0
##          C    0    0  719    0    0
##          D    0    0    0  675    0
##          E    0    0    0    0  757
## 
## Overall Statistics
##                                     
##                Accuracy : 1         
##                  95% CI : (0.999, 1)
##     No Information Rate : 0.284     
##     P-Value [Acc > NIR] : <2e-16    
##                                     
##                   Kappa : 1         
##  Mcnemar's Test P-Value : NA        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    1.000    1.000    1.000    1.000
## Specificity             1.000    1.000    1.000    1.000    1.000
## Pos Pred Value          1.000    1.000    1.000    1.000    1.000
## Neg Pred Value          1.000    1.000    1.000    1.000    1.000
## Prevalence              0.284    0.193    0.175    0.164    0.184
## Detection Rate          0.284    0.193    0.175    0.164    0.184
## Detection Prevalence    0.284    0.193    0.175    0.164    0.184
## Balanced Accuracy       1.000    1.000    1.000    1.000    1.000
```

OOB estimate of error rate: 0.8% with the final blended model.


```r
rfFitAll$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 11
## 
##         OOB estimate of  error rate: 0.8%
## Confusion matrix:
##      A   B   C   D   E class.error
## A 1166   5   0   1   0    0.005119
## B    4 788   5   0   0    0.011292
## C    0   7 709   3   0    0.013908
## D    0   0   7 667   1    0.011852
## E    0   0   0   0 757    0.000000
```

```r
################### Predict Validation ##########

rfFitCFVal<- confusionMatrix(filteredValidation$classe, predict(rfFit, validationPredPCA))
rfFitCFVal
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1653    8    6    7    0
##          B   43 1056   30    7    3
##          C   10   17  973   23    4
##          D    5    8   48  903    1
##          E    1    6    8    8 1059
## 
## Overall Statistics
##                                         
##                Accuracy : 0.959         
##                  95% CI : (0.953, 0.964)
##     No Information Rate : 0.291         
##     P-Value [Acc > NIR] : < 2e-16       
##                                         
##                   Kappa : 0.948         
##  Mcnemar's Test P-Value : 1.12e-06      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.966    0.964    0.914    0.953    0.993
## Specificity             0.995    0.983    0.989    0.987    0.995
## Pos Pred Value          0.987    0.927    0.947    0.936    0.979
## Neg Pred Value          0.986    0.992    0.981    0.991    0.998
## Prevalence              0.291    0.186    0.181    0.161    0.181
## Detection Rate          0.281    0.179    0.165    0.153    0.180
## Detection Prevalence    0.284    0.193    0.174    0.164    0.184
## Balanced Accuracy       0.980    0.974    0.951    0.970    0.994
```

```r
rfFitCFVal1<- confusionMatrix(filteredValidation$classe, predict(rfFit1, validationPredPCA))
rfFitCFVal1
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1653   10    5    6    0
##          B   47 1051   31    6    4
##          C    9   22  970   23    3
##          D    4    7   41  910    3
##          E    1    5    6    9 1061
## 
## Overall Statistics
##                                         
##                Accuracy : 0.959         
##                  95% CI : (0.954, 0.964)
##     No Information Rate : 0.291         
##     P-Value [Acc > NIR] : < 2e-16       
##                                         
##                   Kappa : 0.948         
##  Mcnemar's Test P-Value : 4.94e-05      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.964    0.960    0.921    0.954    0.991
## Specificity             0.995    0.982    0.988    0.989    0.996
## Pos Pred Value          0.987    0.923    0.944    0.943    0.981
## Neg Pred Value          0.986    0.991    0.983    0.991    0.998
## Prevalence              0.291    0.186    0.179    0.162    0.182
## Detection Rate          0.281    0.179    0.165    0.155    0.180
## Detection Prevalence    0.284    0.193    0.174    0.164    0.184
## Balanced Accuracy       0.980    0.971    0.955    0.971    0.993
```

```r
#This one uses pre-PCA data.
rfFitCFNOPCAVal1<- confusionMatrix(filteredValidation$classe, predict(rfFitCFnoPCA, validationPred))
rfFitCFNOPCAVal1
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B   16 1114    8    0    1
##          C    0   11 1014    2    0
##          D    0    0   17  948    0
##          E    0    0    0    0 1082
## 
## Overall Statistics
##                                         
##                Accuracy : 0.991         
##                  95% CI : (0.988, 0.993)
##     No Information Rate : 0.287         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.988         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.991    0.990    0.976    0.998    0.999
## Specificity             1.000    0.995    0.997    0.997    1.000
## Pos Pred Value          1.000    0.978    0.987    0.982    1.000
## Neg Pred Value          0.996    0.998    0.995    1.000    1.000
## Prevalence              0.287    0.191    0.176    0.161    0.184
## Detection Rate          0.284    0.189    0.172    0.161    0.184
## Detection Prevalence    0.284    0.193    0.174    0.164    0.184
## Balanced Accuracy       0.995    0.992    0.987    0.997    1.000
```

```r
gbmFitCFVal<- confusionMatrix(filteredValidation$classe, predict(gbmFit1, validationPredPCA))
gbmFitCFVal
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1495   54   40   72   13
##          B  113  833  105   54   34
##          C   66   73  827   47   14
##          D   39   16  107  791   12
##          E   26   78   85   50  843
## 
## Overall Statistics
##                                         
##                Accuracy : 0.813         
##                  95% CI : (0.803, 0.823)
##     No Information Rate : 0.295         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.764         
##  Mcnemar's Test P-Value : <2e-16        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.860    0.790    0.710    0.780    0.920
## Specificity             0.957    0.937    0.958    0.964    0.952
## Pos Pred Value          0.893    0.731    0.805    0.820    0.779
## Neg Pred Value          0.942    0.953    0.931    0.955    0.985
## Prevalence              0.295    0.179    0.198    0.172    0.156
## Detection Rate          0.254    0.141    0.140    0.134    0.143
## Detection Prevalence    0.284    0.193    0.174    0.164    0.184
## Balanced Accuracy       0.908    0.864    0.834    0.872    0.936
```


```r
## Make a data.frame for the validation predictions.
predVal1<- predict(rfFit, validationPredPCA, type = "prob")
predVal2<- predict(rfFit1, validationPredPCA, type = "prob")
#This one uses pre-PCA data.
predVal3<- predict(rfFitCFnoPCA, validationPred, type = "prob")
predVal4<- predict(gbmFit1, validationPredPCA, type = "prob")

predDFVal<- data.frame(predVal1, predVal2, predVal3, predVal4, classe = filteredValidation$classe)


FitCFAllVal<- confusionMatrix(predDFVal$classe, predict(rfFitAll, predDFVal))

FitCFAllVal
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1669    5    0    0    0
##          B    7 1128    4    0    0
##          C    0    9 1010    8    0
##          D    0    0   13  949    3
##          E    0    0    0    0 1082
## 
## Overall Statistics
##                                         
##                Accuracy : 0.992         
##                  95% CI : (0.989, 0.994)
##     No Information Rate : 0.285         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.989         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.996    0.988    0.983    0.992    0.997
## Specificity             0.999    0.998    0.997    0.997    1.000
## Pos Pred Value          0.997    0.990    0.983    0.983    1.000
## Neg Pred Value          0.998    0.997    0.997    0.998    0.999
## Prevalence              0.285    0.194    0.174    0.163    0.184
## Detection Rate          0.284    0.192    0.172    0.161    0.184
## Detection Prevalence    0.284    0.193    0.174    0.164    0.184
## Balanced Accuracy       0.997    0.993    0.990    0.994    0.999
```

99.2% accuracy on a one time run on the validation set with models built on the training set, then predicted out and a blended model built on those predictions on an independent test set.

This seems highly accurate for a model built on blended predictions.

Now, we will build the final model, and run it once on the actual test set. The work flow will be changed slightly since we don't have a test set and validation set. 

The final model will be created by the 4 models predicting the outcomes of the training set, then using those training set probabilities to train the blended model. 

The test set will be predicted by each of those 4 models, then all the probabilities will be run through that final model for the final prediction.


```r
########################################################################################################
##FINAL RUN
######################



trainingFINAL<-read.csv("pml-training.csv", na.strings = c("NA", "#DIV/0!"))
testingFINAL<- read.csv("pml-testing.csv", na.strings = c("NA", "#DIV/0!"))


reducedTrainingFINAL<- subset(trainingFINAL, select = -c(X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window))
reducedTestingFINAL<- subset(testingFINAL, select = -c(X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window))

nzv <- nearZeroVar(reducedTrainingFINAL)

# Use training ZeroVariance info to remove those variables(columns) from both the training and testing datasets. Reduced to 120 variables + classe.
filteredTrainingFINAL<- reducedTrainingFINAL[,-nzv] 
filteredTestingFINAL<- reducedTestingFINAL[,-nzv] 
```


```r
nacols <- function(df) {
  colnames(df)[unlist(lapply(df, function(x) any(is.na(x))))]
}

NAcols<-nacols(filteredTrainingFINAL)

filteredTrainingFINAL<- filteredTrainingFINAL[ , -which(names(filteredTrainingFINAL) %in% NAcols)]
filteredTestingFINAL<- filteredTestingFINAL[ , -which(names(filteredTestingFINAL) %in% NAcols)]

names(filteredTrainingFINAL)[53]
```

```
## [1] "classe"
```

```r
descrCorFINAL <- cor(filteredTrainingFINAL[,-53])
highlyCorDescrFINAL <- findCorrelation(descrCorFINAL, cutoff = 0.8)

filteredTrainingFINAL <- filteredTrainingFINAL[, -highlyCorDescrFINAL]
filteredTestingFINAL <- filteredTestingFINAL[, -highlyCorDescrFINAL]

# One less column was dropped when running on the full training set.
names(filteredTrainingFINAL)[41]
```

```
## [1] "classe"
```

```r
names(filteredTestingFINAL)[41]
```

```
## [1] "problem_id"
```


```r
preObjFINAL<- preProcess(filteredTrainingFINAL[,-41] , method = c("BoxCox","center", "scale"))

## predict test and training scaled and centered values with the same training set settings/object
trainPredFINAL<- predict(preObjFINAL, filteredTrainingFINAL[,-41])
testPredFINAL<- predict(preObjFINAL, filteredTestingFINAL[,-41])

#*#*#*#*#
### A training/test/validation set from here will be called to a later random forest model. This will be to create one random forest model with pre-PCA data, and one with post-PCA data to blend together.
#*#*#*#*#
```


```r
prCompFINAL<- preProcess(trainPredFINAL, method = "pca")
predPCAFINAL<- predict(prCompFINAL, trainPredFINAL)

#predict test PCA with training set values. DO NOT retrain using test variables.
testPredPCAFINAL<- predict(prCompFINAL, testPredFINAL)
```


```r
set.seed(808)
rfFitFINAL<- train(filteredTrainingFINAL$classe ~ ., method = "rf", data = predPCAFINAL, trControl = trainControl(savePredictions = TRUE))
```


```r
set.seed(808)
rfFitFINAL1<- train(filteredTrainingFINAL$classe ~ .,
               method = "rf",
               data = predPCAFINAL,
               trControl = trainControl(savePredictions = TRUE,
                                        method = "repeatedcv",
                                        number = 10,
                                        repeats = 3,
                                        allowParallel = TRUE))
```


```r
################ No PCA Random Forest
#Uses pre-PCA dataset
set.seed(808)
rfFitCFnoPCAFINAL<- train(filteredTrainingFINAL$classe ~., 
                     method = "rf",
                     data = trainPredFINAL,
                     trControl = trainControl(savePredictions = TRUE,
                                              allowParallel = TRUE))
```


```r
set.seed(808)
gbmFitFINAL1 <- train(filteredTrainingFINAL$classe ~ ., data = predPCAFINAL,
                 method = "gbm",
                 verbose = FALSE,
                 trControl = trainControl(savePredictions = TRUE))
```


```r
pred1FINAL<- predict(rfFitFINAL, predPCAFINAL, type = "prob")
pred2FINAL<- predict(rfFitFINAL1, predPCAFINAL, type = "prob")
#This uses pre-PCA data
pred3FINAL<- predict(rfFitCFnoPCAFINAL, trainPredFINAL, type = "prob")
pred4FINAL<- predict(gbmFitFINAL1, predPCAFINAL, type = "prob")


predDFFINAL<- data.frame(pred1FINAL, pred2FINAL, pred3FINAL, pred4FINAL, classe = filteredTrainingFINAL$classe)
```


```r
set.seed(808)
rfFitFINALAll<- train(classe ~ ., method = "rf", data = predDFFINAL, trControl = trainControl(savePredictions = TRUE,  
                                                                                    allowParallel = TRUE))
```


```r
# Final accuracy of the training set data predicting its own classe.
# It reached 100%, through blending models together. 
# This isn't too much of a surprise since it was highly accurate predicting training->test-validation (97.6%), so just predicting its own values should be even more accurate..

gbmFitCFFINALAll<- confusionMatrix(predDFFINAL$classe, predict(rfFitFINALAll, predDFFINAL))
gbmFitCFFINALAll
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 5580    0    0    0    0
##          B    0 3797    0    0    0
##          C    0    0 3422    0    0
##          D    0    0    0 3216    0
##          E    0    0    0    0 3607
## 
## Overall Statistics
##                                 
##                Accuracy : 1     
##                  95% CI : (1, 1)
##     No Information Rate : 0.284 
##     P-Value [Acc > NIR] : <2e-16
##                                 
##                   Kappa : 1     
##  Mcnemar's Test P-Value : NA    
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    1.000    1.000    1.000    1.000
## Specificity             1.000    1.000    1.000    1.000    1.000
## Pos Pred Value          1.000    1.000    1.000    1.000    1.000
## Neg Pred Value          1.000    1.000    1.000    1.000    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.194    0.174    0.164    0.184
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       1.000    1.000    1.000    1.000    1.000
```


```r
pred5FINAL<- predict(rfFitFINAL, testPredPCAFINAL, type = "prob")
pred6FINAL<- predict(rfFitFINAL1, testPredPCAFINAL, type = "prob")
#This uses no-PCA test set
pred7FINAL<- predict(rfFitCFnoPCAFINAL, testPredFINAL, type = "prob")
pred8FINAL<- predict(gbmFitFINAL1, testPredPCAFINAL, type = "prob")
predDFTESTFINAL<- data.frame(pred5FINAL, pred6FINAL, pred7FINAL, pred8FINAL)
```


```r
predDFTESTFINAL$problem_id<- testing$problem_id
predDFTESTFINAL$FinalPrediction<- predict(rfFitFINALAll, predDFTESTFINAL)
predDFTESTFINAL
```

```
##        A     B     C     D     E   A.1   B.1   C.1   D.1   E.1   A.2   B.2
## 1  0.120 0.736 0.078 0.020 0.046 0.106 0.726 0.112 0.020 0.036 0.094 0.700
## 2  0.818 0.064 0.098 0.004 0.016 0.784 0.072 0.120 0.014 0.010 0.936 0.030
## 3  0.350 0.216 0.362 0.012 0.060 0.360 0.238 0.322 0.026 0.054 0.146 0.594
## 4  0.960 0.016 0.014 0.008 0.002 0.948 0.016 0.022 0.006 0.008 0.916 0.008
## 5  0.954 0.010 0.028 0.000 0.008 0.962 0.010 0.012 0.000 0.016 0.952 0.018
## 6  0.050 0.080 0.132 0.070 0.668 0.046 0.088 0.146 0.042 0.678 0.022 0.132
## 7  0.080 0.026 0.110 0.730 0.054 0.104 0.036 0.118 0.696 0.046 0.032 0.002
## 8  0.084 0.602 0.054 0.148 0.112 0.070 0.654 0.044 0.124 0.108 0.066 0.774
## 9  1.000 0.000 0.000 0.000 0.000 1.000 0.000 0.000 0.000 0.000 0.998 0.000
## 10 0.978 0.006 0.014 0.002 0.000 0.988 0.004 0.004 0.004 0.000 0.976 0.014
## 11 0.390 0.436 0.048 0.092 0.034 0.392 0.428 0.046 0.102 0.032 0.104 0.684
## 12 0.074 0.080 0.762 0.022 0.062 0.072 0.066 0.762 0.016 0.084 0.038 0.028
## 13 0.032 0.896 0.032 0.020 0.020 0.020 0.928 0.010 0.014 0.028 0.022 0.932
## 14 1.000 0.000 0.000 0.000 0.000 1.000 0.000 0.000 0.000 0.000 1.000 0.000
## 15 0.036 0.062 0.042 0.068 0.792 0.032 0.064 0.024 0.072 0.808 0.024 0.064
## 16 0.020 0.026 0.004 0.008 0.942 0.028 0.016 0.006 0.010 0.940 0.030 0.044
## 17 0.958 0.006 0.016 0.004 0.016 0.960 0.010 0.010 0.002 0.018 0.992 0.002
## 18 0.026 0.874 0.016 0.028 0.056 0.016 0.884 0.018 0.022 0.060 0.040 0.876
## 19 0.028 0.872 0.018 0.024 0.058 0.054 0.826 0.024 0.034 0.062 0.078 0.814
## 20 0.000 0.984 0.002 0.006 0.008 0.006 0.974 0.000 0.006 0.014 0.006 0.988
##      C.2   D.2   E.2     A.3     B.3      C.3      D.3      E.3
## 1  0.128 0.050 0.028 0.28494 0.19843 0.087043 0.360051 0.069530
## 2  0.030 0.004 0.000 0.46979 0.13732 0.294224 0.038871 0.059791
## 3  0.170 0.018 0.072 0.36975 0.15815 0.359906 0.033352 0.078836
## 4  0.040 0.020 0.016 0.73258 0.04881 0.144365 0.042304 0.031946
## 5  0.020 0.002 0.008 0.74794 0.03258 0.182750 0.019701 0.017026
## 6  0.170 0.046 0.630 0.04409 0.18575 0.414931 0.097377 0.257850
## 7  0.112 0.824 0.030 0.15515 0.08194 0.092632 0.477032 0.193248
## 8  0.032 0.104 0.024 0.19036 0.11350 0.022122 0.324366 0.349649
## 9  0.000 0.000 0.002 0.97021 0.01276 0.002784 0.003242 0.011005
## 10 0.004 0.002 0.004 0.90819 0.01544 0.056414 0.006596 0.013361
## 11 0.074 0.086 0.052 0.63782 0.12574 0.126258 0.060329 0.049854
## 12 0.876 0.012 0.046 0.15851 0.08987 0.556924 0.055847 0.138855
## 13 0.008 0.010 0.028 0.05051 0.74599 0.108530 0.031748 0.063221
## 14 0.000 0.000 0.000 0.95119 0.01472 0.020862 0.006412 0.006817
## 15 0.034 0.042 0.836 0.08201 0.18560 0.031990 0.061822 0.638583
## 16 0.002 0.014 0.910 0.30034 0.14354 0.037701 0.039260 0.479158
## 17 0.000 0.000 0.006 0.79901 0.07187 0.062365 0.022009 0.044751
## 18 0.012 0.054 0.018 0.03823 0.85687 0.009294 0.012658 0.082949
## 19 0.030 0.062 0.016 0.16411 0.51747 0.038380 0.060175 0.219866
## 20 0.002 0.000 0.004 0.01331 0.85386 0.014850 0.029584 0.088393
##    FinalPrediction
## 1                B
## 2                A
## 3                B
## 4                A
## 5                A
## 6                E
## 7                D
## 8                B
## 9                A
## 10               A
## 11               B
## 12               C
## 13               B
## 14               A
## 15               E
## 16               E
## 17               A
## 18               B
## 19               B
## 20               B
```

## Final model accuracy

The Final Predictions are all correct when compared to submission feedback so we can get accuracy feedback of each individual model in addition to the final model.


```r
## RF default options on PCA processed data.

confusionMatrix(predDFTESTFINAL$FinalPrediction, predict(rfFitFINAL, testPredPCAFINAL))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction A B C D E
##          A 7 0 0 0 0
##          B 0 7 1 0 0
##          C 0 0 1 0 0
##          D 0 0 0 1 0
##          E 0 0 0 0 3
## 
## Overall Statistics
##                                         
##                Accuracy : 0.95          
##                  95% CI : (0.751, 0.999)
##     No Information Rate : 0.35          
##     P-Value [Acc > NIR] : 2.9e-08       
##                                         
##                   Kappa : 0.929         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity              1.00    1.000    0.500     1.00     1.00
## Specificity              1.00    0.923    1.000     1.00     1.00
## Pos Pred Value           1.00    0.875    1.000     1.00     1.00
## Neg Pred Value           1.00    1.000    0.947     1.00     1.00
## Prevalence               0.35    0.350    0.100     0.05     0.15
## Detection Rate           0.35    0.350    0.050     0.05     0.15
## Detection Prevalence     0.35    0.400    0.050     0.05     0.15
## Balanced Accuracy        1.00    0.962    0.750     1.00     1.00
```

```r
## RF repeated cross-validation 10 folds, 3 repeats on PCA processed data.

confusionMatrix(predDFTESTFINAL$FinalPrediction, predict(rfFitFINAL1, testPredPCAFINAL))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction A B C D E
##          A 7 0 0 0 0
##          B 1 7 0 0 0
##          C 0 0 1 0 0
##          D 0 0 0 1 0
##          E 0 0 0 0 3
## 
## Overall Statistics
##                                         
##                Accuracy : 0.95          
##                  95% CI : (0.751, 0.999)
##     No Information Rate : 0.4           
##     P-Value [Acc > NIR] : 3.41e-07      
##                                         
##                   Kappa : 0.928         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.875    1.000     1.00     1.00     1.00
## Specificity             1.000    0.923     1.00     1.00     1.00
## Pos Pred Value          1.000    0.875     1.00     1.00     1.00
## Neg Pred Value          0.923    1.000     1.00     1.00     1.00
## Prevalence              0.400    0.350     0.05     0.05     0.15
## Detection Rate          0.350    0.350     0.05     0.05     0.15
## Detection Prevalence    0.350    0.400     0.05     0.05     0.15
## Balanced Accuracy       0.938    0.962     1.00     1.00     1.00
```

```r
## RF default options on Box-Cox transformed, scaled, and centered data but no PCA transformation.

confusionMatrix(predDFTESTFINAL$FinalPrediction, predict(rfFitCFnoPCAFINAL, testPredFINAL))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction A B C D E
##          A 7 0 0 0 0
##          B 0 8 0 0 0
##          C 0 0 1 0 0
##          D 0 0 0 1 0
##          E 0 0 0 0 3
## 
## Overall Statistics
##                                     
##                Accuracy : 1         
##                  95% CI : (0.832, 1)
##     No Information Rate : 0.4       
##     P-Value [Acc > NIR] : 1.1e-08   
##                                     
##                   Kappa : 1         
##  Mcnemar's Test P-Value : NA        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity              1.00      1.0     1.00     1.00     1.00
## Specificity              1.00      1.0     1.00     1.00     1.00
## Pos Pred Value           1.00      1.0     1.00     1.00     1.00
## Neg Pred Value           1.00      1.0     1.00     1.00     1.00
## Prevalence               0.35      0.4     0.05     0.05     0.15
## Detection Rate           0.35      0.4     0.05     0.05     0.15
## Detection Prevalence     0.35      0.4     0.05     0.05     0.15
## Balanced Accuracy        1.00      1.0     1.00     1.00     1.00
```

```r
## GBM model on PCA processed data.

confusionMatrix(predDFTESTFINAL$FinalPrediction, predict(gbmFitFINAL1, testPredPCAFINAL))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction A B C D E
##          A 7 0 0 0 0
##          B 2 4 0 1 1
##          C 0 0 1 0 0
##          D 0 0 0 1 0
##          E 0 0 1 0 2
## 
## Overall Statistics
##                                         
##                Accuracy : 0.75          
##                  95% CI : (0.509, 0.913)
##     No Information Rate : 0.45          
##     P-Value [Acc > NIR] : 0.00643       
##                                         
##                   Kappa : 0.658         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.778    1.000    0.500    0.500    0.667
## Specificity             1.000    0.750    1.000    1.000    0.941
## Pos Pred Value          1.000    0.500    1.000    1.000    0.667
## Neg Pred Value          0.846    1.000    0.947    0.947    0.941
## Prevalence              0.450    0.200    0.100    0.100    0.150
## Detection Rate          0.350    0.200    0.050    0.050    0.100
## Detection Prevalence    0.350    0.400    0.050    0.050    0.150
## Balanced Accuracy       0.889    0.875    0.750    0.750    0.804
```

```r
## Final blended model accuracy.

confusionMatrix(predDFTESTFINAL$FinalPrediction, predict(rfFitFINALAll, predDFTESTFINAL))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction A B C D E
##          A 7 0 0 0 0
##          B 0 8 0 0 0
##          C 0 0 1 0 0
##          D 0 0 0 1 0
##          E 0 0 0 0 3
## 
## Overall Statistics
##                                     
##                Accuracy : 1         
##                  95% CI : (0.832, 1)
##     No Information Rate : 0.4       
##     P-Value [Acc > NIR] : 1.1e-08   
##                                     
##                   Kappa : 1         
##  Mcnemar's Test P-Value : NA        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity              1.00      1.0     1.00     1.00     1.00
## Specificity              1.00      1.0     1.00     1.00     1.00
## Pos Pred Value           1.00      1.0     1.00     1.00     1.00
## Neg Pred Value           1.00      1.0     1.00     1.00     1.00
## Prevalence               0.35      0.4     0.05     0.05     0.15
## Detection Rate           0.35      0.4     0.05     0.05     0.15
## Detection Prevalence     0.35      0.4     0.05     0.05     0.15
## Balanced Accuracy        1.00      1.0     1.00     1.00     1.00
```

```r
## Final Model Diagnostics

rfFitFINALAll$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 0%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 5580    0    0    0    0           0
## B    0 3797    0    0    0           0
## C    0    0 3422    0    0           0
## D    0    0    0 3216    0           0
## E    0    0    0    0 3607           0
```

