---
title: IN GOD'S MEETING

---

# IN GOD'S MEETING
## Course Review
### Models
:::spoiler Expand/Hide
- PLA
- Linear Regression
- Logistic Regression
- Feature Transform
- Linear Hard Margin SVM
- Kernel SVM
- Linear Soft Margin SVM
- Random Forest
- AdaBoost-Decision Tree
- Gradient Boosted Decision Tree
- Neural Network
:::
### Concepts
:::spoiler Expand/Hide
- Machine Learning:
    Improving some <font color = 'orange'> performance measure </font> with experience <font color='red'> computed</font> from <font color='cyan'>data</font>
- Essence of Machine Learning:
    1. Exists some 'underlying' pattern to be learned
    2. No programmable (easy) definition
    3. Somehow there is data about the pattern
- Target Function $f$:
    The function $f$ IN GOD'S MIND
- Learning Algorithm $\mathcal A$:
    The algorithm that chooses the final hypothesis $g$ from the hypothesis set $\mathcal H$
- Learning Model:
    Learning Algorithm + Hypothesis Set
- Linear Separable:
    Exists perfect perceptron $w$ that correctly classifies all examples ($x_n,\ y_n$)
- No Free Lunch Theorem:
    Without any assumptions on the learning problem on hand, all learning algorithms perform the same
- In-sample error $E_{in}(h)$ :
    $\frac{1}{N}\sum^{N}_{n=1}\mathbb{[}h(x_n)\neq y_n\mathbb{]}$
- Out-of-Sample Error $E_{out}(h)$ : 
    $\mathbb{E}_{\mathbf{x}\sim P} \mathbb{[}h(\mathbf{x})\neq f(\mathbf{x})\mathbb{]}$
- $\mathcal Z$ space
    The space where data after transformation lives
- Generalization:
    Moving from $E_{in}$ to $E_{out}$
- Generaliztion Error:
    $E_{out} - E_{in}$
- Bad Generalization:
    Low $E_{in}$, high $E_{out}$
- Model Complexity:
    How complicated the model is. $\approx$ VC dimension/ the number of parameters/ hypothesis set size/ power /boundary complexity
- Overfitting:
    low<font color='cyan'>**er**</font> $E_{in}$, high<font color='cyan'>**er**</font> $E_{out}$
- Underfitting:
    high<font color='cyan'>**er**</font> $E_{in}$, high<font color='cyan'>**er**</font> $E_{out}$
- Stochastic Noise:
    Noise caused by random
- Deterministic Noise:
    Noise caused by the difference between the best hypothesis and the target function
- Data Cleaning:
    Correct the label of data
- Data Pruning:
    Remove the example
- Data Hinting (Data Filling) (Data Imputation):
    Add virtual examples
- Regularization:
    Restricting the learning algorithm to find hypotheses in a smaller hypothesis set
- Validation Error $E_{val}$:
    Using samples not used for training to evaluate the error
- Augmented Error:
    In-sample error + regularizer
- Model Selection:
    Selecting which Learning Model (= Learning Algorithm + Hypothesis Set) to use. Hypothesis Set $\approx$ parameters
- Leave-One-Out Cross Validation:
    Average validation error of all validation sets containing only one example
- V-fold Cross Validation:
    Average validation error of all validation sets containing $\frac{N}{V}$ examples
- Three Learning Principles:
    - Occam's Razor (Less is More):
        The simplest model tha that fits the data is also the most plausible
    - Sampling Bias:
        If the data is sampled in a biased way, learning will produce a similarly biased outcome
    - Data Snooping:
        If a data set has affected any step in the learning process, its ability to assess the outcome has been compromised
- Margin:
    Distance from closest example to separating hyperplane
- Margin violation:
    The total amount of distances of examples that fall on the other side of the boundary
- Aggregation (Blending):
    Voting of different hypotheses
- Bootstrapping:
    re-sample $N'$ examples from $\mathcal D$ uniformly with replacement
:::
## Meeting Time
### Every Friday <font color="violet">$8:00 \sim 9:00$</font> p.m.
## Links
[GitHub](https://github.com/AnthonyChing/HTMLB)
[Kaggle stage 1](https://www.kaggle.com/competitions/html-2024-fall-final-project-stage-1)
[Kaggle stage 2](https://www.kaggle.com/competitions/html-2024-fall-final-project-stage-2)

## Dates
1. **First & Second stage submission deadline:**  
    <font color="pink">2024/12/15 23:59 UTC+8</font>
3. **Report deadline:**  
    <font color="pink">2024/12/23 13:00 UTC+8</font>

## Models  

1. Logistic regression -- ==Izac==
2. SVM -- ==Aaron==
3. Gradient Boosted Decision Tree -- ==Uranustrong==
4. Random Forest -- ==Anthony==



### Requirements
1. 4 Machine Learing algos
2. At Most 7 pages of Reports
    - accuracy 
        - kaggle two stages scores (public/private) (Assuming that we can see the private scores after)
        - late game validation score $E_{late}$
        - time series validation score $E_{tscv}$
            - 12 uploads in each stage (logi-regre-, SVM, RF, LightGBM, AdaBoost-stump, blend)
            - ~~(Unless we cheat -- last resort)~~
    - stability across the two stages
        - This is just accuracy
    - efficiency
        - Training
        - Predicting
        - Implementation
        - `time python3 main.py`
        - `Measure-Command <cmd>`
    - scalability
        - Data size (Evaluate with validation scores)
        - Features (Evaluate with validation scores), (One-hot encoding on teams and pitchers)
        - Tasks (Scenario) (We can ~~唬爛~~)
    - interpretability
        - (How well can we ~~唬爛~~)
        - Heuristic 
        - "How well can we learn about the data(Problem) from the model"
            - "In random forest, we can record the branching criteria, which can then interpret by humans."
            - "In SVM, we can infer data characteristics from SVs"
            - "In Gradient Boosting Machine we can also find the importance of features"
            - "In logistic regression we can compare the impacts that every feature has on the result."
    - pros and cons 
        - Summary and Recommendation and Suggestion 
    - data cleaning
        - same data cleaning method?
            1. dropping 
            2. encoding? 
---
## 2024/12/21
### Progress
#### Aaron
#### Anthony
- Course Reivew
#### Izac
#### Uranustrong
- For the efficiency, do we need to measure it on the same machine?
- Pushed the consensus log file
### To discuss
* What do we want to write our report with? (Word (8)/ Google Docs(9)/ HackMD(9)/ Overleaf(17)/ Goodnotes(17)?)
* Word (9)/ <font color='pink'>Google Docs(7)</font>/ HackMD(8)?
    - 1. Word 2. Google Docs 3. HackMD
    - 1. HackMD 2. Google Docs 3. Word. 
    - 1. HackMD 2. Google Docs 3. Word
    - 1.Google Docs 2.Word 3.HackMD
* Course Review
* A section for explanation of **Late Game Validation** and **Time Series Cross Validation** in our report
* Page allocation
    * Individual section (1*4): <font color='cyan'>**Reproducibility**</font>, Approaches, 
    * Common section: <font color='cyan'>**Comparison**</font>, Validation, Work load, References, Preprocessing
* Add documentations for our codes in `README.md`. [Matplotlib](https://github.com/matplotlib/matplotlib) [libsvm](https://github.com/cjlin1/libsvm)
* Include `requirements.txt` to tell other people what packages to install to run our models.
### Comments

Stage 1
|Model|$E_{late}$|$E_{TSCV}$ |$E_{out}$|Training Time|Implement Difficulty|Data size small|Data size medium|Columns small|Columns Medium|Task ($E_{isNightGame}$)|
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |

Stage 2
|Model|$E_{late}$|$E_{TSCV}$|$E_{out}$|
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

late

|Model|$E_{late}^1$|$E_{late}^2$|$E_{out}^1$|$E_{out}^2$|
|---|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

tscv

|Model|$E_{TSCV}^1$|$E_{TSCV}^2$|$E_{out}^1$|$E_{out}^2$|
|---|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

Recommendation

|Model|Implement Efficiency|Interpretability|Pro|Con|Rank|
|---|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |


---
## 2024/12/13
### Progress
#### Aaron
* Gaussian kernel SVM 
    * $C = 0.1,\ \gamma = 0.1:\ 58.457\%$ (78 col); $58.166\%$ (All col)
* BLENDING
#### Anthony
* Random Forest
    * Stage 1: Random forest (100000, 1126), 58.618%
    * Stage 2: Random forest (100000, 1126), 58.388%
#### Izac
* Trained with Uranus's data
* Abandoned KLR and other transformations due to poor math and python skills
* Choosing best learning rate(0.002) and epochs(100000)
#### Uranustrong
* Upload final preprocessing code (hopefully)
* Manually selecting the combinations of parameters.(aggregated number of trees, num_leaves, bagging frequency, bagging fraction)
* 
### To discuss
* Time series validation
*  [<font color="pink">**Final report**</font>](#Requirements)
* Quota allocation
* Data cleaning (Drop first games?)
* Blending
* Ada-Boost stump
### Comments
---
## 2024/12/6
### Progress
#### Aaron
* Preprocessor to encode test data according to training data mappings
#### Anthony
* Finished data preprocessing
* Trained LSTM
* Predicted
* <font color="cyan">Abandon LSTM, switch to Random Forest</font>
#### Izac
* Still trying among different kernels and parameters 
#### Uranustrong
### To discuss
* Other preprocessing methods
    * NaN filling
        * Directly calc:
            * season (fill with date)
            * team_rest
            * pitcher_rest
            * 
        * 
### Comments
* Don't encode the labels of the validation set into the training set lol
---
## 2024/11/29

### Progress
#### Aaron
* Data cleaning, replace categorical NULL with median, numerical NULL with mean
* Ran svm with linear kernal with 10-fold validation, accuracy: 64%(validation) (with scaled data)
    * Try different kernel
    * Try better validation to match scenario 
#### Anthony
* Some data cleaning
* :::spoiler Workflow
    ![IMG_1213](https://hackmd.io/_uploads/Bk5HyND7yx.png)

#### Izac
* Made the pre-process cleaner lol
* Conducted logistic regression on the training set
    * Chose SGD for better efficiency
    * Accuracy only around 60% on the original set -> data not even close to linearly separable?
    * Try poly-transform next week?
#### Uranustrong
- would like to try GBDT rather than Adaboost combined with decision stump
- could try to sample the later matches more to achieve more robust prediction?
- the effort to fill missing data seems vanishing(?)
### To discuss
 * Random forest?
 * Gradient Boosted Decision Tree?
### Comments
* There isn't a "date" column in the test set
---
## 2024/11/22
### Progress
#### Aaron
* Some preprocessing
    * Split date string
    * Target encoding on team & pitcher names
    * Remove some columns
#### Anthony
* "Managers"
* Uploaded coinflip, True, and False
#### Izac
* Still getting familiar with python and github lol
* Extracted floating numbers from the file to form the matrix for training
#### Uranustrong
- Stucked on handling training data

|||
|---|---|
|![Screenshot 2024-11-22 at 19.22.24](https://hackmd.io/_uploads/B1TOtk0fJg.png)|![Screenshot 2024-11-22 at 19.22.38](https://hackmd.io/_uploads/S19KFkAGJg.png)|


### To discuss
* Change the profile pictures
* Deal with missing features
### Comments
* There are 11065 rows that contains missing data lol 
* No missing team names, dates or labels 
* Stage 1
    * Coinflip: 0.52291
    * True: <font color="pink">0.53066 :crown: </font>
    * False: 0.46933
* Stage 2
    * Coinflip: 0.50830
    * True: <font color="pink">0.52990 :crown: </font>
    * False: 0.47009
---
## 2024/11/17

### To do

### Comments