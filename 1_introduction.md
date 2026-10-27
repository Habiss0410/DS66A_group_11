# Marketing Campaign Response Prediction

### Predicting customer response to a Portuguese bank's telemarketing campaign under temporal distribution shift

---

## 1. Introduction

A bank in Portugal runs telemarketing campaigns to persuade customers to subscribe to a **term deposit**, giving the bank a stable source of funds.

Because calls are costly and the conversion rate is low (about 11.7%), correctly identifying likely subscribers can improve campaign efficiency.

> **Research question:** Can a machine learning model that uses pre-call information and past interaction data predict subscription, prioritize customers, and maintain performance when the data distribution shifts over time?

The project aims at a time-aware prediction workflow and targeted marketing instead of mass calling.

---

## 2. Problem Definition and Temporal Characteristics of the Data

This is a **binary classification** problem: predicting whether a customer subscribes to a term deposit (`yes`/`no`).

### Target variable and class imbalance

* **Target variable:** `y` (`1` = subscribed, `0` = did not subscribe).
* **Base distribution:** About 11.7% of customers subscribe and 88.3% do not.
* **Change over time:** The subscription rate rises from **8.3% in Block 1 to 53.0% in Block 4** across the temporal test blocks. In addition, roughly the first 20,000 rows contain no customers previously contacted (`pdays = -1`). This is genuine *prior shift* and *covariate shift*, so a random split does not reflect the real scenario of predicting the future.

### Dataset overview

The dataset `bank-full.csv` contains **45,211 observations and 17 variables**. After removing 3 records with zero call duration, **45,208 observations** remain, with no missing values. The file is ordered by call time but has no date/year column.

| Group                    | Variables                                         | Role                          |
| ------------------------ | ------------------------------------------------- | ----------------------------- |
| **Customer profile**     | `age`, `job`, `marital`, `education`              | Demographics                  |
| **Financial status**     | `balance`, `housing`, `loan`, `default`           | Balance and financial burden  |
| **Current campaign**     | `contact`, `day`, `month`, `duration`, `campaign` | Contact information           |
| **Previous campaign**    | `pdays`, `previous`, `poutcome`                   | Interaction history           |

---

## 3. Methodology and Modeling Decisions

### A. Leakage prevention and feature engineering

* **Removing unsuitable features:**

  * Drop `duration` because it is known only after the call.
  * Drop `day` because it has little business meaning.
  * Drop `campaign_log` (`DROP_CAMPAIGN = True`) because `campaign` is fully known only after the campaign with that customer ends. `contact_intensity` (`campaign + previous`) is also disabled (`ADD_CONTACT_INTENSITY = False`) because it does not improve results and contains `campaign`.
  * `month` is not used for splitting; it is used only to check the temporal structure, while its encoding in the model (`none`, `cyc`, `onehot`) and whether to keep `contact` are chosen by temporal CV on the training set.

* **Data transformations:**

  * **`balance`:** Signed logarithm to reduce skewness and the influence of outliers. `previous` uses `log1p`.
  * **`pdays`:** Split into `was_contacted_before` (previously contacted or not) and `pdays_log` (log-transformed, `NaN` when never contacted; XGBoost uses the native `NaN`, Logistic Regression and Random Forest fill 0).
  * **`age_group`:** Age bands `18-30`, `31-45`, `46-60`, `60+` added alongside `age`.
  * **Preprocessing:** Imputation, scaling, and one-hot encoding live inside a `Pipeline` and are fit only on each fold's training part to limit data leakage.

### B. Temporal validation and evaluation strategy

* **Data split:** Train on the first 80% of the data (the past) and test on the last 20%; the test set is never used for tuning, feature selection, threshold selection, or model selection. `TimeSeriesSplit` with 5 expanding-window folds is used for Optuna hyperparameter tuning, optimizing fold-weighted ROC-AUC (more recent folds get larger weights).
* **Model selection:** A default-parameter Logistic Regression baseline is compared with tuned Logistic Regression, Random Forest (standard or Balanced RF), and XGBoost. A complex model is chosen only if its mean ΔAUC over the baseline exceeds the standard error and it beats the baseline in at least 4 of 5 folds; otherwise the baseline is kept.
* **Block-wise evaluation:** The test set is split into 4 temporal blocks and evaluated with **Block-Averaged ROC-AUC**, **Minimum AUC**, and **Lift@20%**; pooled test ROC-AUC is only secondary. A frozen model (*Static/Frozen*) is also compared with a model retrained as new labels arrive (*Retrained*).
* **Threshold optimization:** Out-of-fold (**OOF**) predictions from the 2 most recent folds determine the F1-optimal threshold instead of defaulting to `0.5`; the threshold is updated after each retraining.

---

## 4. Project Objectives

1. **Explore influencing factors:** Identify customer characteristics and interaction history related to subscription through Logistic Regression coefficients and Random Forest / XGBoost feature importance; diagnose concept drift with per-block univariate ROC-AUC.
2. **Build models:** Train, tune, and compare Logistic Regression, Random Forest, and XGBoost against a Logistic Regression baseline using temporal cross-validation.
3. **Create business value:** Rank customers by subscription probability using **Precision@K%**, **Recall@K%**, and **Lift@K%** (with Lift@20% as the focus), and recommend periodic retraining and budget-driven calling of the top-K% customers to reduce call costs.
