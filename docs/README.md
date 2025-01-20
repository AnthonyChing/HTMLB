# HTML Final Project Report \- IN GOD’S MIND

Authors: Aaron Lin (B12902116), Anthony Ching (B12902118), Izac Tsai (B13902066), Uranus Song (B12902066)  
Date: 2024/12/23

## Work Load

Each team member studied on 1 machine learning algorithm and worked on the corresponding section in this report. The other sections are done together evenly. All members attended weekly meetings ([meeting minutes](./IN%20GOD'S%20MEETING.html))

## Preprocessing

Data imputation:  
We use different imputation methods based on the characteristics of the feature:

* Season: Fill from date column  
* Pitcher/team rest: Count from previous relevant game  
* Rolling average and cumulative mean: Average of previous and next row, if there are two or more consecutive missing data, we use the front\_fill method which propagates the last valid value downwards. 

Different algorithms below may process the data further differently based on this.

## Validation Sets

We defined two sets of validation data that better suit the competition setting. All of our machine learning algorithms are compared with each other based on them.

1\. Late Game Validation:  
For stage 1, we build a validation set from all games after 7/15 in each year. For stage 2, we build the validation set with all games from 2023\. Both of these validation sets aim to match the scenario of each stage more accurately.

2\. Time Series Cross Validation:  
We sort the training data by date, then split it into 5 parts, then we build 4 training/validation data sets according to the following graph:

| Data1 | Data2 | Data3 | Data4 | Data5 |
| :---: | :---: | :---: | :---: | :---: |
| train 1 | validation 1 |  |  |  |
| train 2 |  | validation 2 |  |  |
| train 3 |  |  | validation 3 |  |
| train 4 |  |  |  | validation 4 |

We use the following heuristic to calculate the final cross validation errors: $Acc_{TSCV}=(1*Acc_1+2*Acc_2+3*Acc_3+4*Acc_4)/10$ where $Acc_i$ are calculated with validation $i$, $i = 1, 2, 3, 4$

## Logistic Regression

For logistic regression, we use stochastic gradient descent for efficiency, and train the model for fixed N=100000 epochs, selecting the best learning rate from {0.001, 0.002, 0.005, 0.01}. Moreover, since the algorithm can’t compute with missing values, filling the missing data must be done.  
Workflow:

1. Data imputation for missing features  
2. Drop categorical columns  
3. Train to find best learning rate and obtain the optimal w with it  
4. Predict

Since a too-small learning rate could be too slow to converge, and a too-big one could also fail to step down to the optimal solution, the validation accuracy is the best when learning rate \= 0.002, and this will have a better guarantee that the optimal solution can be reached.

Late game accuracy for stage 1

| Learning Rate | 0.001 | 0.002 | 0.005 | 0.01 |
| :---: | :---: | :---: | :---: | :---: |
| Accuracy | 52.63 | 53.17 | 53.17 | 51.09 |

Late game accuracy for stage 2

| Learning Rate | 0.001 | 0.002 | 0.005 | 0.01 |
| :---: | :---: | :---: | :---: | :---: |
| Accuracy | 52.04 | 54.64 | 50.28 | 50.47 |

TSCV accuracy

| Learning Rate | 0.001 | 0.002 | 0.005 | 0.01 |
| :---: | :---: | :---: | :---: | :---: |
| Accuracy | 53.77 | 54.48 | 53.11 | 53.22 |

## Support Vector Machine

For SVM, we choose the Gaussian kernel for our model. There are two parameters that we can tune, C and gamma, which correspond to the margin violation penalty coefficient, and the gamma in the Gaussian function. Unlike Decision trees, SVMs cannot naturally process missing values and non numerical data, thus we need two extra steps of preprocessing before we can deploy our model.    
Main workflow:

1. Data imputation on categorical columns  
2. Target encoding on categorical columns  
3. Data scaling  
4. Grid search parameters  
5. Retrain with entire dataset and predict

For data imputation, we fill missing data with the mode from the same team within the same season. Since the test data does not contain dates, we cannot use the techniques we used on the training data, thus we impute with the mean of the same team within the same season. (Note: The season can still be inferred from the ‘home\_team\_season’ and ‘away\_team\_season’ columns, and was processed separately)  
We encode boolean features with 0/1 encoding. For multiclass features, we use target encoding, which is a method of encoding categorical features with high cardinality. It replaces each class with a blend of posterior probability (probability of label given the class) and prior probability (probability of label over all the data). (Note: We must calculate the encoding on the training set, then fit our encoding mappings onto the validation set, to prevent the training data from being contaminated, not doing so would result in overly high validation accuracy.)  
For scaling, we simply scale the data to be between 0 and 1\.  
For parameter selection, we define a range for C and gamma, then select based on validation score. The values of parameters are spaced evenly on a logarithmic scale. An initial base10 grid search (10^-5 \~ 10^5 for C and gamma) showed that an optimal area with good validation accuracy was around C \= 0.1, gamma \= 0.1, later searches were conducted around this region with a base of 2 for a more fine grained search.

Accuracy spreads: (column and row headers denote the powers of 2\)  
[![][image5]](https://www.codecogs.com/eqnedit.php?latex=E_%7BTSCV%7D#0) 

| γ\\C | \-6 | \-5 | \-4  | \-3  | \-2  | \-1 | 0 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| \-7 | 47.0276 | 47.0276 | 47.0276 | 47.0095 | 46.1421 | 44.9407 | 45.7269 |
| \-6 | 47.0276 | 47.0276 | 47.0276 | 46.3815 | 44.8368 | 45.4513 | 46.1831 |
| \-5 | 47.0276 | 47.0276 | 46.5669 | 44.882 | 45.3519 | 46.0431 | 46.5038 |
| \-4  | 47.0276 | 46.856 | 45.0039 | 45.3429 | 45.8803 | 46.4361 | 46.8292 |
| \-3 | 47.0276 | 45.939 | 44.9994 | 45.8624 | 46.3278 | 47.0369 | 47.326 |
| \-2  | 47.0276 | 46.0924 | 45.9208 | 46.1288 | 46.6799 | 46.829 | 46.9871 |
| \-1 | 47.0276 | 46.8739 | 46.3906 | 46.4674 | 46.2868 | 46.5851 | 46.9916 |

[![][image6]](https://www.codecogs.com/eqnedit.php?latex=E_%7Blate%7D%5E%7Bstage%201%7D#0)/[![][image7]](https://www.codecogs.com/eqnedit.php?latex=E_%7Blate%7D%5E%7Bstage%202%7D#0)

| γ\\C | \-6 | \-5 | \-4  | \-3  | \-2  | \-1 | 0 | γ\\C | \-6 | \-5 | \-4  | \-3  | \-2  | \-1 | 0 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| \-7 | 45.4801 | 45.4801 | 45.4801 | 44.5691 | 43.9384 | 43.8683 | 45.41 | \-7 | 47.7771 | 47.7771 | 47.7771 | 47.7771 | 46.7126 | 46.1491 | 45.8986 |
| \-6 | 45.4801 | 45.4801 | 45.1297 | 43.6581 | 43.8683 | 45.1297 | 45.6903 | \-6 | 47.7771 | 47.7771 | 47.7145 | 46.7753 | 46.2117 | 45.9612 | 46.5248 |
| \-5 | 45.4801 | 45.4801 | 44.2888 | 43.5179 | 44.6392 | 46.1108 | 46.6714 | \-5 | 47.7771 | 47.7771 | 47.6519 | 46.2117 | 45.836 | 45.836 | 46.2743 |
| \-4  | 45.4801 | 43.3778 | 43.5179 | 44.5691 | 45.6903 | 46.1808 | 46.4612 | \-4  | 47.7771 | 47.2762 | 46.0238 | 46.2743 | 46.0865 | 46.5248 | 46.5874 |
| \-3 | 45.41 | 44.1486 | 44.0085 | 44.4289 | 45.1998 | 45.41 | 46.5312 | \-3 | 47.8397 | 46.3995 | 46.2117 | 45.9612 | 46.1491 | 46.3369 | 47.5893 |
| \-2  | 45.4801 | 43.7282 | 44.0085 | 43.588 | 43.6581 | 44.2888 | 45.9005 | \-2  | 47.7145 | 47.7771 | 47.2136 | 47.3388 | 47.0883 | 47.464 | 46.2117 |
| \-1 | 45.4801 | 45.4801 | 45.4801 | 45.3399 | 45.41 | 45.3399 | 44.3588 | \-1 | 47.7771 | 47.8397 | 48.1528 | 47.7771 | 47.6519 | 47.0257 | 47.9024 |

The cyan boxes have the best performance, so they were chosen for later comparisons

## Random Forest

For the random forest algorithm, we chose a few parameters to tune from:

1. Trees: The number of trees in the random forest.  
2. Max features: The number of features to consider when looking for the best split. (sqrt means max\_features=sqrt(n\_features), and similarly for log2.)  
3. Max depth: The maximum depth of the tree.

This algorithm handles missing features by choosing sub-branching criteria, so only indexing of pitchers and teams were done.  
Workflow:

1. Indexing on categorical columns  
2. Grid search with model parameters  
3. Retrain with entire dataset and predict

Because of the special design of the algorithm, random forest has a self-validation property called out-of-bag error, but for comparison against other models, we did not select our model using it. RandomForestClassifier from the Scikit-learn package was used to implement it. (The random\_state parameter was chosen to be 1126\)  
Grid search with error measures:  
[![][image5]](https://www.codecogs.com/eqnedit.php?latex=E_%7BTSCV%7D#0) / [![][image8]](https://www.codecogs.com/eqnedit.php?latex=E_%7Boob%7D#0)

| Trees |  | 100 |  | 300 |  | 500 |  | 700 |  | 900 |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| max features | max depth |  |  |  |  |  |  |  |  |  |  |
| sqrt | None | 0.4536 | 0.4665 | 0.4461 | 0.4612 | 0.4473 | 0.4545 | 0.4435 | 0.4578 | 0.4459 | 0.458 |
|  | 10 | 0.4422 | 0.4557 | 0.4411 | 0.4472 | 0.4385 | 0.4493 | 0.4346 | 0.4463 | 0.4339 | 0.4427 |
|  | 15 | 0.4504 | 0.4588 | 0.4423 | 0.4546 | 0.4406 | 0.4562 | 0.4417 | 0.4514 | 0.4414 | 0.4511 |
| log2 | None | 0.4591 | 0.468 | 0.444 | 0.468 | 0.4472 | 0.464 | 0.4456 | 0.4621 | 0.4434 | 0.4625 |
|  | 10 | 0.4415 | 0.4483 | 0.4353 | 0.4455 | 0.435 | 0.4465 | 0.4367 | 0.4451 | 0.4327 | 0.4436 |
|  | 15 | 0.4477 | 0.4675 | 0.4442 | 0.4552 | 0.4439 | 0.4563 | 0.4429 | 0.4552 | 0.4392 | 0.4527 |
| None | None | 0.4549 | 0.467 | 0.4478 | 0.4608 | 0.4482 | 0.4648 | 0.4476 | 0.4615 | 0.4479 | 0.4594 |
|  | 10 | 0.4424 | 0.4608 | 0.4407 | 0.4527 | 0.438 | 0.4481 | 0.4386 | 0.4484 | 0.4396 | 0.4473 |
|  | 15 | 0.448 | 0.4647 | 0.4416 | 0.4545 | 0.441 | 0.4573 | 0.4445 | 0.454 | 0.4423 | 0.454 |

[![][image6]](https://www.codecogs.com/eqnedit.php?latex=E_%7Blate%7D%5E%7Bstage%201%7D#0) / [![][image7]](https://www.codecogs.com/eqnedit.php?latex=E_%7Blate%7D%5E%7Bstage%202%7D#0)

| Trees |  | 100 |  | 300 |  | 500 |  | 700 |  | 900 |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| max features | max depth |  |  |  |  |  |  |  |  |  |  |
| sqrt | None | 0.4605 | 0.4778 | 0.457 | 0.4772 | 0.4541 | 0.464 | 0.4527 | 0.4678 | 0.4577 | 0.4584 |
|  | 10 | 0.4464 | 0.4509 | 0.4464 | 0.4559 | 0.452 | 0.4465 | 0.4436 | 0.4434 | 0.4464 | 0.4453 |
|  | 15 | 0.4534 | 0.4597 | 0.4485 | 0.4428 | 0.4436 | 0.4584 | 0.4394 | 0.4559 | 0.445 | 0.4496 |
| log2 | None | 0.4506 | 0.4753 | 0.4499 | 0.4584 | 0.4513 | 0.4665 | 0.4556 | 0.459 | 0.4598 | 0.4496 |
|  | 10 | 0.4506 | 0.4503 | 0.4429 | 0.4403 | 0.4485 | 0.444 | 0.4499 | 0.444 | 0.4485 | 0.4446 |
|  | 15 | 0.457 | 0.4672 | 0.4471 | 0.4509 | 0.452 | 0.4478 | 0.4492 | 0.4434 | 0.4478 | 0.4478 |
| None | None | 0.4429 | 0.4853 | 0.4478 | 0.4691 | 0.4513 | 0.4572 | 0.4534 | 0.454 | 0.4499 | 0.4559 |
|  | 10 | 0.4429 | 0.4547 | 0.4485 | 0.4559 | 0.4499 | 0.4459 | 0.457 | 0.4453 | 0.4527 | 0.4434 |
|  | 15 | 0.452 | 0.4753 | 0.4549 | 0.4728 | 0.4513 | 0.4547 | 0.4464 | 0.4509 | 0.4506 | 0.444 |

The cyan boxes have the best performance, so they were chosen for later comparisons (except for the one corresponding to [![][image8]](https://www.codecogs.com/eqnedit.php?latex=E_%7Boob%7D#0)).

## LightGBM

The full name of LightGBM is Light Gradient Boosting Machine, it is a learning framework based on gradient boosting method, which is constructed by multiple weak learners in a gradient descent way. In this competition, we use the boosting type to be a gradient boosting decision tree, that is we aggregate numerous trees based on how well one tree performs. The model provides some parameters for users to tune and add variety to the model like the fraction of used features to branch the data, and the fraction of used data to construct a tree, which theoretically add more randomness to different trees and hence achieve better generalization.  
Workflow:  
1\. Do data imputation to fill in the missing data  
2\. Drop all the categorical columns (Since we think categorical features is not that important)  
3\. Use grid search based on the validation accuracy  
4\. Retrain the data with best parameter combination and whole data 

Accuracy late 1\\ Accuracy late 2

| Learning Rate |  | 0.001 |  | 0.01 |  | 0.1 |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| max leaves | trees |  |  |  |  |  |  |
| 7 | 100 | 0.4961 | 0.5373 | 0.5494 | 0.5454 | 0.5431 | 0.5379 |
|  | 300 | 0.5053 | 0.5348 | 0.5438 | 0.5454 | 0.5473 | 0.5366 |
|  | 500 | 0.5025 | 0.5460 | 0.5438 | 0.5542 | 0.5396 | 0.5097 |
| 15 | 100 | 0.5389 | 0.5448 | 0.5529 | 0.5510 | 0.5431 | 0.5091 |
|  | 300 | 0.5326 | 0.5498 | 0.5438 | 0.5498 | 0.5452 | 0.5166 |
|  | 500 | 0.5340 | 0.5454 | 0.5375 | 0.5479 | 0.5445 | 0.5072 |
| 31 | 100 | 0.5480 | 0.5448 | 0.5515 | 0.5479 | 0.5382 | 0.5166 |
|  | 300 | 0.5438 | 0.5479 | 0.5424 | 0.5517 | 0.5431 | 0.5141 |
|  | 500 | 0.5375 | 0.5454 | 0.5424 | 0.5385 | 0.5298 | 0.5059 |

Accuracy tscv

| Learning Rate |  | 0.001 | 0.01 | 0.1 |
| :---: | :---: | :---: | :---: | :---: |
| max leaves | trees |  |  |  |
| 7 | 100 | 0.5297 | 0.5586 | 0.5568 |
|  | 300 | 0.5398 | 0.5618 | 0.5513 |
|  | 500 | 0.5498 | 0.5636 | 0.5436 |
| 15 | 100 | 0.5297 | 0.563 | 0.5564 |
|  | 300 | 0.5422 | 0.5668 | 0.5428 |
|  | 500 | 0.5516 | 0.5641 | 0.527 |
| 31 | 100 | 0.5297 | 0.5655 | 0.5414 |
|  | 300 | 0.5454 | 0.558 | 0.5323 |
|  | 500 | 0.5537 | 0.554 | 0.5235 |

## 

## Comparisons

Stage 1\\ Stage 2:

| Model | Logistic Regression |  | SVM |  | Random Forest |  | LightGBM |  | Uniform Blending |  |
| :---- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [![][image9]](https://www.codecogs.com/eqnedit.php?latex=E_%7Blate%7D#0) | 0.4266 | 0.4796 | 0.4338 | 0.4584 | 0.4394 | 0.4403 | 0.4415 | 0.4319 | N/A | N/A |
| [![][image10]](https://www.codecogs.com/eqnedit.php?latex=E_%7BTSCV%7D#0) | 0.4695 |  | 0.4484 |  | 0.4327 |  | 0.4322 |  | N/A |  |
| [![][image11]](https://www.codecogs.com/eqnedit.php?latex=E_%7Bout%7D#0) | 0.4943 | 0.4888 | 0.4223 | 0.4555 | 0.4191 | 0.4452 | 0.4248 | 0.4341 | 0.4233 | 0.4481 |
| Training Time (s) | 33.086 | 32.017 | 28.867 | 26.613 | 64.071 | 64.071 | 1.68 | 1.67 | 127.704 | 124.371 |
| [![][image12]](https://www.codecogs.com/eqnedit.php?latex=E_%7Bout%7D#0) of [![][image13]](https://www.codecogs.com/eqnedit.php?latex=%5Cmathcal%20D_%7Bsmall%7D#0) | 0.4467 | 0.4597 | 0.4205 | 0.4625 | 0.4199 | 0.4522 | 0.442 | 0.461 | N/A | N/A |
| [![][image14]](https://www.codecogs.com/eqnedit.php?latex=E_%7Bout%7D#0) of [![][image15]](https://www.codecogs.com/eqnedit.php?latex=%5Cmathcal%20D_%7Bmedium%7D#0) | 0.4513 | 0.4612 | 0.4223 | 0.4605 | 0.4175 | 0.439 | 0.4527 | 0.4596 | N/A | N/A |
| [![][image16]](https://www.codecogs.com/eqnedit.php?latex=E_%7Bout%7D#0) of [![][image17]](https://www.codecogs.com/eqnedit.php?latex=%5Cmathcal%20D'_%7Bsmall%7D#0) | 0.4723 | 0.4239 | 0.4184 | 0.46 | 0.4183 | 0.4428 | 0.4548 | 0.4691 | N/A | N/A |
| [![][image18]](https://www.codecogs.com/eqnedit.php?latex=E_%7Bout%7D#0) of [![][image19]](https://www.codecogs.com/eqnedit.php?latex=%5Cmathcal%20D'_%7Bmedium%7D#0) | 0.4362 | 0.4192 | 0.419 | 0.4638 | 0.4184 | 0.4419 | 0.4525 | 0.4699 | N/A | N/A |
| [![][image20]](https://www.codecogs.com/eqnedit.php?latex=E_%7BisNightGame%7D#0) | 0.3000 | 0.3041 | 0.2963 | 0.3732 | 0.2963 | 0.3728 | 0.2963 | 0.3546 | 0.29612  | 0.6294 |

### Findings

1. Accuracy  
   [![][image11]](https://www.codecogs.com/eqnedit.php?latex=E_%7Bout%7D#0) is calculated from the whole test set, which is a better measure than public score and private score. We can see that for stage 1: SVM \> RF \> LightGBM \> Logistic Regression, for stage 2: LightGBM \> RF \> SVM \> Logistic Regression.   
2. Stability across the two stages  
   We take all measure except for [![][image10]](https://www.codecogs.com/eqnedit.php?latex=E_%7BTSCV%7D#0) into account. For [![][image11]](https://www.codecogs.com/eqnedit.php?latex=E_%7Bout%7D#0), if we calculate [![][image21]](https://www.codecogs.com/eqnedit.php?latex=%7CE_%7Bout%7D%5E%7Bstage1%7D%20-%20E_%7Bout%7D%5E%7Bstage2%7D%7C#0) for each model, we get:

| Model | Logistic Regression | SVM | Random Forest | LightGBM |
| ----- | :---: | :---: | :---: | :---: |
| [![][image22]](https://www.codecogs.com/eqnedit.php?latex=%7CE_%7Bout%7D%5E%7Bstage1%7D%20-%20E_%7Bout%7D%5E%7Bstage2%7D%7C#0)   | 0.0055  | 0.0332 | 0.0261 | 0.0093 |

   Coupled with other measures, roughly: Logistic Regression \> LightGBM \> RF \> SVM

3. Efficiency  
   We take both the training time and the implementation difficulty into account. Training time is calculated with ‘time’ command on CSIE workstation 1, and we only train the model with the parameters corresponding to the best performance. We can see that in terms of pure training time: LightGBM \< SVM \< Logistic Regression \< RF  
   In terms of implementation, due to the nature of decision trees, we can simply index categorical features, whereas SVM and LR require extra encoding schemes if we wish to fully use the categorical features.  
4. Scalability   
   Data Size:  
   We defined [![][image13]](https://www.codecogs.com/eqnedit.php?latex=%5Cmathcal%20D_%7Bsmall%7D#0) and [![][image15]](https://www.codecogs.com/eqnedit.php?latex=%5Cmathcal%20D_%7Bmedium%7D#0) being 33% and 66% of randomly picked data from the original data set, respectively.  We sum the difference between the original [![][image11]](https://www.codecogs.com/eqnedit.php?latex=E_%7Bout%7D#0), and the [![][image11]](https://www.codecogs.com/eqnedit.php?latex=E_%7Bout%7D#0) of [![][image13]](https://www.codecogs.com/eqnedit.php?latex=%5Cmathcal%20D_%7Bsmall%7D#0) and [![][image15]](https://www.codecogs.com/eqnedit.php?latex=%5Cmathcal%20D_%7Bmedium%7D#0), across both stages to evaluate the scalability:

| Model | Logistic Regression | SVM | Random Forest | LightGBM |
| :---: | :---: | :---: | :---: | :---: |
| [![][image23]](https://www.codecogs.com/eqnedit.php?latex=%5Csum%7CE_%7Bout%7D%20-%20E_%7Bout%7D%5E%7B%5Cmathcal%7BD%7D_i%7D%7C#0) | 0.1473  | 0.0138 | 0.0156 | 0.2478 |

   We see that in terms of scalability with respect to data size: SVM \> Logistic RF \> Logistic Regression \> LightGBM

   Features:  
   We defined [![][image17]](https://www.codecogs.com/eqnedit.php?latex=%5Cmathcal%20D'_%7Bsmall%7D#0) and [![][image19]](https://www.codecogs.com/eqnedit.php?latex=%5Cmathcal%20D'_%7Bmedium%7D#0) being 64 and 114 most important columns  from the original data set calculated by random forest and lightGBM. We sum the difference between the original [![][image11]](https://www.codecogs.com/eqnedit.php?latex=E_%7Bout%7D#0), and the [![][image11]](https://www.codecogs.com/eqnedit.php?latex=E_%7Bout%7D#0) of [![][image17]](https://www.codecogs.com/eqnedit.php?latex=%5Cmathcal%20D'_%7Bsmall%7D#0) and [![][image19]](https://www.codecogs.com/eqnedit.php?latex=%5Cmathcal%20D'_%7Bmedium%7D#0), across both stages  to evaluate the scalability:

| Model | Logistic Regression | SVM | Random Forest | LightGBM |
| :---: | :---: | :---: | :---: | :---: |
| [![][image24]](https://www.codecogs.com/eqnedit.php?latex=%5Csum%7CE_%7Bout%7D%20-%20E_%7Bout%7D%5E%7B%5Cmathcal%7BD%7D_i'%7D%7C#0) | 0.2146  | 0.02 | 0.0072 | 0.1285 |

   We see that in terms of scalability with respect to feature count: RF \> SVM \> LightGMB \> Logistic Regression.

   Tasks:  
   We defined [![][image20]](https://www.codecogs.com/eqnedit.php?latex=E_%7BisNightGame%7D#0) to be the error rate calculated with the 0/1 error on a new task where we predict the column ‘is\_night\_game’ instead. We sum the difference between the original [![][image11]](https://www.codecogs.com/eqnedit.php?latex=E_%7Bout%7D#0), and [![][image25]](https://www.codecogs.com/eqnedit.php?latex=E_%7BisNightGame%7D#0) across both stages to evaluate the scalability: 

| Model | Logistic Regression | SVM | Random Forest | LightGBM |
| :---: | :---: | :---: | :---: | :---: |
| [![][image26]](https://www.codecogs.com/eqnedit.php?latex=%5Csum%7CE_%7Bout%7D%20-%20E_%7BisNightGame%7D%7C#0) | 0.379 | 0.2083 | 0.1952 | 0.208 |

   We see that in terms of scalability with respect to feature count: RF \> SVM \> LightGBM \> Logistic Regression.

5. Interpretability

	For random forest, we can record the branching criteria, which can then be interpreted by humans. For LightGBM, we can also find the importance of features. For SVM, we can infer data characteristics from SVs. For logistic regression, we can compare the weights of each feature.  
RF ≒ LightGBM \> SVM \> Logistic Regression

6. Validation Methods  
   The errors in the late game validation and TSCV are close to Eout. We believe this is due to the data being too difficult to fit, or the nature of these methods being similar.

7. ### Blending    We used uniform voting of the 4 models, and predicted 'True' to break ties. It doesn’t improve the accuracy significantly.

Overall, ranked from highest to lowest (1\~5):

| Model | Logistic Regression | SVM | Random Forest | LightGBM | Uniform Blending |
| :---- | :---: | :---: | :---: | :---: | :---: |
| Accuracy | 5 | 2 | 3 | 1 | 4 |
| Stability | 1 | 4 | 3 | 2 | NA |
| Efficiency | 3 | 2 | 4 | 1 | 5 |
| Scalability | 4 | 2 | 1 | 3 | NA |
| Interpretability | 3 | 2 | 1 | 1 | NA |
| Pros | Stable | Fast, Accurate | Good scalability | Fast, Accurate | Simple |
| Cons | Less accurate | Unstable | Inefficient | Bad scalability | Highly Dependent |
| Ranking | 4 | 3 | 2 | 1 | 5 |

Final Recommendation  
We recommend using LightGBM, for its good accuracy and efficiency.   
Pros: Human explainable, Good interpretability, Fast training time, Fast prediction time  
Cons: Too Complicated, Hard to tune

## References

1\. Source code: [https://github.com/AnthonyChing/HTMLB](https://github.com/AnthonyChing/HTMLB)  
2\. [Scikit-learn: Machine Learning in Python](https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html), Pedregosa *et al.*, JMLR 12, pp. 2825-2830, 2011\.  
3\. Chih-Chung Chang and Chih-Jen Lin, LIBSVM : a library for support vector machines. ACM Transactions on Intelligent Systems and Technology, 2:27:1--27:27, 2011\. Software available at [http://www.csie.ntu.edu.tw/\~cjlin/libsvm](http://www.csie.ntu.edu.tw/~cjlin/libsvm)   
4.LightGBM: [https://github.com/microsoft/LightGBM/tree/master](https://github.com/microsoft/LightGBM/tree/master) 

[image1]: <data:image/png;base64,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>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAANBAMAAABIqROZAAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMAdu/dIomrMhBEZpnNu1TMZGYvAAAAEklEQVR4XmP8z4AMmFB4Q4MLAMvoARmqZ9s7AAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAQAAAALBAMAAACqiTGYAAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMAid3NZkSrVBDvuzKZdiJR1kIBAAAAD0lEQVR4XmP8z8DEgA8BACFsARUFeiXrAAAAAElFTkSuQmCC>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGAAAAANBAMAAACk3HaPAAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMAid3NZkSrVBDvuzKZdiJR1kIBAAAAGElEQVR4XmP8z0AaYEIXIARGNRADRqQGAJx4ARlK+kdjAAAAAElFTkSuQmCC>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAANBAMAAAAgWpGhAAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMAVJmrZrt2RBCJze/dIjJISVHlAAAAFElEQVR4XmP8z4ANMKELQMCIFgYAOT8BGRT1y2kAAAAASUVORK5CYII=>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAUBAMAAAAEg8LyAAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMAVJmrZrt2RBCJze/dIjJISVHlAAAAF0lEQVR4XmP8z4ANMKELQMCoMCYYPMIA7wABJ22mtEMAAAAASUVORK5CYII=>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAUBAMAAAAEg8LyAAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMAVJmrZrt2RBCJze/dIjJISVHlAAAAF0lEQVR4XmP8z4ANMKELQMCoMCYYPMIA7wABJ22mtEMAAAAASUVORK5CYII=>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABsAAAANBAMAAACqdQjgAAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMAVJmrZrt2RBCJze/dIjJISVHlAAAAFUlEQVR4XmP8z4AEPjIh8xgYhgQXAHZcAgqBbA0SAAAAAElFTkSuQmCC>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAB0AAAANBAMAAACna3inAAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMAVJmrZrt2RBCJze/dIjJISVHlAAAAFUlEQVR4XmP8z4AMPjKhcBkYhhofAI9RAgoH5zlVAAAAAElFTkSuQmCC>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAANBAMAAAAgWpGhAAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMAVJmrZrt2RBCJze/dIjJISVHlAAAAFElEQVR4XmP8z4ANMKELQMCIFgYAOT8BGRT1y2kAAAAASUVORK5CYII=>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABsAAAANBAMAAACqdQjgAAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMAVJmrZrt2RBCJze/dIjJISVHlAAAAFUlEQVR4XmP8z4AEPjIh8xgYhgQXAHZcAgqBbA0SAAAAAElFTkSuQmCC>

[image12]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABsAAAANBAMAAACqdQjgAAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMAVJmrZrt2RBCJze/dIjJISVHlAAAAFUlEQVR4XmP8z4AEPjIh8xgYhgQXAHZcAgqBbA0SAAAAAElFTkSuQmCC>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACgAAAANBAMAAAApsTHbAAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMAEERUdomZq7vN3e8yImbKO59XAAAAFElEQVR4XmP8z4AJmNAFQGCECAIAHe0BGRfx9HEAAAAASUVORK5CYII=>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABsAAAANBAMAAACqdQjgAAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMAVJmrZrt2RBCJze/dIjJISVHlAAAAFUlEQVR4XmP8z4AEPjIh8xgYhgQXAHZcAgqBbA0SAAAAAElFTkSuQmCC>

[image15]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADcAAAANBAMAAAD/FOu+AAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMAEERUdomZq7vN3e8yImbKO59XAAAAGElEQVR4XmP8z4ATfGRCF0EGo5IMlEgCANPRAgrtzEexAAAAAElFTkSuQmCC>

[image16]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABsAAAANBAMAAACqdQjgAAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMAVJmrZrt2RBCJze/dIjJISVHlAAAAFUlEQVR4XmP8z4AEPjIh8xgYhgQXAHZcAgqBbA0SAAAAAElFTkSuQmCC>

[image17]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACgAAAAQBAMAAACW+SCeAAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMAEERUdomZq7vN3e8yImbKO59XAAAAF0lEQVR4XmP8z4AJmNAFQGBUEA1gFQQAZBABHwrIsYQAAAAASUVORK5CYII=>

[image18]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABsAAAANBAMAAACqdQjgAAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMAVJmrZrt2RBCJze/dIjJISVHlAAAAFUlEQVR4XmP8z4AEPjIh8xgYhgQXAHZcAgqBbA0SAAAAAElFTkSuQmCC>

[image19]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADcAAAAQBAMAAABAXPr7AAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMAEERUdomZq7vN3e8yImbKO59XAAAAGElEQVR4XmP8z4ATfGRCF0EGo5IMg1ASAIaiAhCbU39SAAAAAElFTkSuQmCC>

[image20]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFEAAAAQBAMAAACcpY7MAAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMAVJmrZrt2RBCJze/dIjJISVHlAAAAG0lEQVR4XmP8z0Ac+MiELoITjKokBoyqJAYAABtTAhAx7v52AAAAAElFTkSuQmCC>

[image21]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAHUAAAATBAMAAABch1/IAAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMAELvvZjKJIqvNRFSZdt3sD9Z/AAAAIUlEQVR4XmP8z0Au+MiELkICGNVLPBjVSzwY1Us8GCi9AMG+AhaAv2W9AAAAAElFTkSuQmCC>

[image22]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAG4AAAARBAMAAADOAYTcAAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMAELvvZjKJIqvNRFSZdt3sD9Z/AAAAHUlEQVR4XmP8z0AWYEIXIBKM6sMORvVhB6P6sAMA9mYBIWesTfsAAAAASUVORK5CYII=>

[image23]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFAAAAAQBAMAAABzZ+XyAAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMAIs0yiauZEO9URGZ2u92QCztbAAAAGUlEQVR4XmP8z0AcYEIXwAVGFeIFowrxAgC4HwEfln6kgAAAAABJRU5ErkJggg==>

[image24]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFAAAAARBAMAAAC4OzZXAAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMAIs0yiauZEO9URGZ2u92QCztbAAAAHElEQVR4XmP8z0AcYEIXwAVGFeIFowrxAqIVAgDmaAEhx0Zv0gAAAABJRU5ErkJggg==>

[image25]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFEAAAAQBAMAAACcpY7MAAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMAVJmrZrt2RBCJze/dIjJISVHlAAAAG0lEQVR4XmP8z0Ac+MiELoITjKokBoyqJAYAABtTAhAx7v52AAAAAElFTkSuQmCC>

[image26]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAHgAAAAQBAMAAAAv7abWAAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMAIs0yiauZEO9URGZ2u92QCztbAAAAHElEQVR4XmP8z0A+YEIXIAWMaiYRjGomEYxEzQAMPQEf2tCviAAAAABJRU5ErkJggg==>
