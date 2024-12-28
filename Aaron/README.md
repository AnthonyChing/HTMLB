# File structure

```
├── blend                   // Script for uniform blending
├── data                    // Scripts for SVM specific data preprocessing 
│   ├── 33%
│   ├── 66%
│   ├── Late_Game_Dataset1
│   ├── Late_Game_Dataset2
│   ├── Test
│   └── TSCV
├── data_dropping           // Scripts for data size scalability train/test data generation
│   ├── 33%
│   └── 66%
├── is_night_game           // Scripts for Scripts for task scalability train/test data generation
├── preprocessing           // Old scripts for preprocessing
└── SVM                     // SVM
    ├── accuracy_spreads    // Grid search accuracy spreads
    ├── models              // Trained models
    └── outputs             // LIBSVM outputs
```

## SVM

* **eval.py/eval_ing.py:**
    Scripts for evaluating $E_{out}$.  
    Usage:  
    `python3 eval.py <prediction.csv> <answer.csv>`  

    Note: `eval_ing.py` is for evaluating $E_{isnightgame}$

* **SVM_LGval.py/SVM_TSCV.py:**
    Code for training SVM with Late game validation/time series cross validation, all data paths reference to a `data` directory, change first code cell paths to reference correct path.  
    Second code cell is for selecting stage.  

    Usage:  
    `python3 SVM_LGval.py` or
    `python3 SVM_TSCV.py`

    Note: These script will open/write to files. (or create new files)  

* **SVM_TSCV_drop.py/SCM_TSCV_ing.py**  
    Code for training SVM with time series cross validation, for data size/task scalability test.
    Second code cell is for selecting stage.  
    all data paths reference to a `data` directory, change first code cell paths to reference correct path.  
    For `SVM_TSCV_drop.py`, `drop_count = _` should be set to either 50 or 100, to test with 66% of columns or 33% of columns respectively.  

    Usage:  
    `python3 SVM_TSCV_drop.py` or
    `python3 SVM_TSCV_ing.py`

    Note: These script will open/write to files. (or create new files)