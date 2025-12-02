# Analysis and Modeling of Road Accident Severity in France (2005–2021)

## 1. Introduction

Road accidents represent a major public health, safety, and economic issue. Every year, thousands of people are killed or seriously injured on French roads. Understanding the factors that influence the severity of an accident is crucial for improving prevention, guiding public policies, and optimizing claims handling or pricing for insurers.

This project relies on a set of open data covering bodily injury traffic accidents in France between 2005 and 2021. The objective is to develop a predictive model capable of estimating the severity of an accident based on known conditions: characteristics of the road, the vehicle, the driver, the circumstances, etc.

## 2. Problem Understanding and Data

### Business Objective

The goal is to predict the **severity of an accident** among three classes:
- **1**: Unharmed
- **2**: Killed
- **3**: Hospitalized Wounded

This task is formulated as a **multi-class classification problem**. A good model could:
- Help identify high-risk situations
- Support the implementation of targeted safety policies
- Provide automatic alerts in embedded tools
- Better understand profiles of serious accidents

### 🗃 Data Source

The data comes from the [Open Data](https://www.data.gouv.fr/fr/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2023/) portal. It aggregates detailed information on accidents, vehicles, and users involved.

The datasets were:
- Cleaned
- Merged
- Encoded
- And enriched with cyclical variables (`TimeOfDay`, `DayOfWeek`, `Month`) via their `sin` and `cos` components.

### Specific Preprocessing

The initial target variable `grav` contained 4 classes:
- **1: Unharmed**
- **2: Killed**
- **3: Hospitalized Wounded**
- **4: Lightly Injured**

Class 4 (lightly injured) representing only **2.7% of cases**, was **removed** to:
- Reduce imbalance
- Stabilize learning
- Simplify analysis

The final target classification therefore includes only **3 significant classes**.

### Available Variables

The final dataset includes variables of the following types:
- **Normalized Numeric** (`age`, `nbv`)
- **Encoded Categorical** (`catu`, `catv`, `manv`, `prof`, etc.)
- **Cyclical** (sine and cosine of the hour, day of the week, month)
- **Context Variables** (lighting, agglomeration, obstacle, etc.)

All spanning **2.3 million rows** covering the whole of France between 2005 and 2021.

## 3. Exploratory Data Analysis (EDA)

### Objective

The objective of this exploratory analysis is to study the distribution of available variables, understand the data structure, and identify potential relationships between explanatory variables and the target variable `grav`, representing accident severity. Since the data had already been cleaned, class 4 (lightly injured) was removed due to its very low representativeness (2.7%), in order to improve the stability and efficiency of future classification models.

### Target Variable Distribution

After processing, the `grav` variable contains three classes:

- **1: Unharmed** — 42.05%
- **2: Killed** — 20.51%
- **3: Hospitalized Wounded** — 37.44%

This relatively balanced distribution allows working in a multi-class classification context without major imbalance. Removing class 4 helped reduce a significant initial imbalance.

### Univariate and Bivariate Analysis

#### Age (`age`)
Age is normalized in the dataset. Boxplot analysis indicates that killed or hospitalized injured people are, on average, slightly older than those unharmed. This variable seems to have a link with accident severity.

#### User Category (`catu`)
The majority of users involved belong to category 1 (likely drivers). However, other categories (passengers, pedestrians, cyclists, etc.) are proportionally more represented in severe classes, suggesting greater vulnerability.

#### Lighting (`lum`)
Although most accidents occur in broad daylight, a significant portion of serious accidents takes place in low visibility conditions (night without lighting, night with insufficient lighting), indicating the influence of lighting on severity.

#### Agglomeration (`agg`)
Accidents are more frequent in built-up areas (agglomerations). However, accident severity is significantly higher outside built-up areas, likely due to higher speeds and a less controlled road environment.

#### Month, Day of the Week, Time of Day
- **Month**: A slight increase in serious accidents is observed during summer and autumn months, linked to increased traffic during holidays.
- **Day of the Week**: Accidents are more frequent during the week, with a concentration on working days (Monday to Friday).
- **Time**: Two daily peaks appear: in the morning (7 am–9 am) and at the end of the day (4 pm–8 pm), corresponding to rush hours. These peaks are also present for serious accidents.

---

### Road and Contextual Variable Analysis

#### Impact Type (`choc`)
Frontal or side impacts are more often associated with serious accidents, while other types (rear, parked, etc.) appear more in cases without injury.

#### Maneuver (`manv`)
A predominant maneuver (code 9) appears massively in all classes, but certain specific maneuvers (e.g., overtaking, U-turn) are over-represented in fatal accidents, suggesting the importance of driver behavior.

#### Vehicle Category (`catv`)
One vehicle category is ultra-dominant (likely private cars). Other categories like motorized two-wheelers or heavy goods vehicles are, proportionally, more often involved in serious accidents.

#### Road Profile and Plan (`prof`, `plan`)
Flat and straight roads are the most frequent. However, roads presenting gradients or curves are more associated with serious accidents, indicating an impact of relief and road geometry on severity.

#### Road Category (`catr`)
Certain road categories (main roads outside built-up areas, highways, etc.) are more likely to cause serious accidents, confirming observations made regarding agglomerations.

#### Obstacle (`obs`) and Special Situation (`situ`)
The presence of fixed obstacles (trees, walls, etc.) or special situations (roadworks, parking, unusual traffic) are aggravating factors in the occurrence of a serious accident.

This exploratory analysis identified several variables strongly correlated with accident severity, notably:
- User profile (age, gender, category)
- Environmental conditions (lighting, agglomeration, time)
- Vehicle type and maneuver
- Road characteristics (profile, plan, category)
- Specific circumstances (impact type, obstacles)

These variables will be prioritized in the modeling phase. Furthermore, temporal analysis highlighted high-risk periods (rush hours, summer periods), which can guide targeted prevention actions.
This EDA constitutes a solid basis for formulating business hypotheses and developing a reliable and interpretable predictive model.

## 4. Data Preparation for Modeling

### Dataset Splitting

The final dataset contains over 2.3 million observations. To ensure robust model evaluation, the data was divided into three subsets:

- **Training Set (train)**: 70%
- **Validation Set (val)**: 15%
- **Test Set (test)**: 15%

The split was performed in a stratified manner to preserve the class distribution of the target variable `grav`. This approach allows training the model on the `train` set, tuning hyperparameters on `val`, and then evaluating final performance on `test`, which remains completely independent.

### Target Variable Processing

The `grav` variable was recoded into three classes for modeling:
- 0: Unharmed
- 1: Killed
- 2: Hospitalized Wounded

This transformation ensures compatibility with classification algorithms, such as XGBoost, which expects integer classes starting at 0.

### Encoding and Normalization

All categorical variables had already been encoded. Numerical variables (`age`, `nbv`) had been normalized. Additionally, cyclical variables (month, hour, day of the week) had been transformed into `sin` and `cos` to respect their periodic nature.
No further processing was necessary at this stage.

### Target Variable Reduction

As mentioned earlier, class 4 (lightly injured) was removed for representativeness reasons (less than 3% of cases). The model was therefore trained only on the three remaining classes to ensure better learning stability and better precision on serious cases.

### Class Rebalancing

Even after removing class 4, the "Killed" class remained under-represented. To compensate for this imbalance, the SMOTE (Synthetic Minority Over-sampling Technique) method was used. This technique creates synthetic examples for the minority class based on nearest neighbors, thus improving learning without loss of information.
SMOTE was applied only to the training set, within a modeling pipeline.

### Variable Reduction

After an initial modeling phase with all variables, feature importance was evaluated using XGBoost. The 10 most important variables were selected for the final version of the model:
- `agg`: zone type (agglomeration or not)
- `catv`: vehicle category
- `catr`: road category
- `obs`: obstacle type
- `catu`: user type
- `plan`: road plan
- `sexe`: user gender
- `nbv`: number of lanes
- `circ`: traffic regime
- `choc`: impact type

This selection simplifies the model, reduces training times, and maintains comparable performance while improving sensitivity on critical cases.

## 5. Modeling

### Model Choice

The main model used in this project is **XGBoost (Extreme Gradient Boosting)**, an ensemble algorithm based on decision trees. This choice is based on several criteria:
- Excellent performance on tabular data
- Native handling of imbalance via weighting and regularization
- Ability to integrate a complete pipeline (preprocessing + model)
- Interpretability via feature importance scores
- Native multi-class support (`objective = "multi:softmax"`)

Other models (such as Random Forest or Logistic Regression) were not tested, as the project goal was not model comparison, but the full optimization of a single model.

### Modeling Pipeline Implementation

A **complete pipeline** was built using `imblearn.pipeline.Pipeline`, integrating:
- **SMOTE** (for automatic class rebalancing)
- **XGBoost** (model training)

This pipeline ensures that resampling (SMOTE) is done **only on the training set**, thus avoiding data leakage between learning and evaluation phases.

### Hyperparameter Optimization

Hyperparameter optimization was performed with `RandomizedSearchCV`, testing 20 random combinations of parameters on a 3-fold cross-validation set. Parameters explored included:
- `n_estimators`: number of trees
- `max_depth`: maximum tree depth
- `learning_rate`: learning rate
- `subsample`: fraction of samples used for each tree
- `colsample_bytree`: fraction of variables used at each split
- `gamma`: regularization (minimum loss reduction required to make a further partition)

Training was performed with **early stopping**, allowing optimization to stop automatically if performance on the validation set did not improve after 10 iterations.

### Custom Evaluation Metric

Since the business objective was to **better detect fatal accidents**, the metric used to guide optimization was not global accuracy, but **recall on the "Killed" class** (encoded 1).
A custom function was defined to extract recall for this class and used as `scoring` in `RandomizedSearchCV`.
This oriented the optimization towards better sensitivity to the most critical cases, even at the cost of a slight sacrifice on other classes.

### Two Approaches Tested

Two versions of the model were tested:
- **Full Model**: trained with all variables in the dataset
- **Reduced Model**: trained only with the 10 most important variables

Comparing the two models allowed evaluating the impact of dimensionality reduction on model performance, particularly on detecting serious cases.

### Model Training

Training was performed on the `X_train` set (70% of data), with 3-fold cross-validation on this same subset. The validation set (`X_val`, 15%) was used for intermediate evaluation, and the test set (`X_test`, 15%) was kept until the final evaluation.

The final model was trained on both:
- A **full set with all available variables**
- A **reduced set** containing only the **10 most important variables**, identified via the `feature_importances_` attribute of XGBoost.

This second model aimed to test if reducing complexity allowed for better performance on serious cases.

## 6. Results

### Full Model Results

On the validation set, the full model achieved:

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Unharmed | 0.70 | 0.80 | 0.75 |
| Killed | 0.47 | 0.49 | 0.48 |
| Hospitalized | 0.63 | 0.52 | 0.57 |

- Global Accuracy: **63%**
- Recall for "Killed" class: **49%**

This model correctly detected about **1 in 2 fatal accidents**, which already represented a clear improvement over the initial baseline (42%).

### Reduced Model Results (10 variables)

With only the 10 most important variables, the model achieved on the validation set:

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Unharmed | 0.75 | 0.74 | 0.75 |
| Killed | 0.46 | 0.58 | 0.51 |
| Hospitalized | 0.63 | 0.53 | 0.58 |

- Global Accuracy: **63%**
- Recall for "Killed" class: **58%**

This model, although simpler, allowed **significantly improving the detection of the most serious cases**, which corresponded to the project's business objective.

### Most Important Variables



The importance graph shows that the most determining variables in the prediction are:

1. `agg`: zone type (agglomeration)
2. `catv`: vehicle type
3. `catr`: road type
4. `obs`: obstacle encountered
5. `catu`: user type
6. `plan`: road profile
7. `sexe`: user gender
8. `nbv`: number of lanes
9. `circ`: traffic regime
10. `choc`: impact type

## 7. Business Interpretation

Analysis of the final model highlights several critical factors influencing accident severity:

- **Zones outside agglomerations** are more dangerous, likely due to higher speeds.
- **Two-wheeled vehicles**, **heavy goods vehicles**, or **pedestrians** are more exposed to serious cases.
- The presence of a fixed or mobile **obstacle** (wall, tree, other vehicle) strongly aggravates the situation.
- **Dangerous maneuvers** (overtaking, U-turns, etc.) are linked to fatal accidents.
- Accidents on **winding or steep roads** are more often associated with serious outcomes.
- Certain personal characteristics like **gender** or **user type** (pedestrian, passenger...) also influence severity.

These observations can be used to:
- Prioritize **road intervention zones** or safety areas
- Develop **targeted prevention policies**
- Guide **risk pricing** for insurance
- Integrate **predictive alerts in embedded systems**

## 8. Conclusion and Perspectives

This project built a multi-class classification model aiming to predict the severity of a road accident in France, based on data from the 2005–2021 period. Through a rigorous approach (data exploration, cleaning, rebalancing, optimization), we obtained a high-performing model, both globally (63% accuracy) and, importantly, on detecting the most serious cases (58% recall for fatal accidents).

Using XGBoost, combined with SMOTE and optimization guided by a business metric (recall on the "Killed" class), allowed building a model that is robust, targeted, and relatively simple thanks to a final selection of 10 variables.

### Model Limitations

Despite these good results, several limitations must be highlighted:
- **Lack of behavioral data**: the data does not account for the driver's state (alcohol, phone, fatigue...), which is essential in reality.
- **Static approach**: the model only takes into account a snapshot of the accident, without dynamic or sequential data.
- **Partial recall on fatal cases**: despite a clear improvement, 42% of fatal accidents remain undetected by the model.

### Perspectives

Several avenues can be explored to improve this model or adapt it to other contexts:
- **Adding new variables** from other sources (precise weather conditions, driver state, traffic density, speed limits).
- **Hierarchical or sequential approach**: first predict global severity, then refine between serious cases (hospitalization vs death).
- **Real-time framework evaluation** for embedded applications (connected cars, GPS alerts...).
- **Deployment of an interactive tool** allowing simulation of accident severity according to chosen parameters (dashboard, web application...).

This project demonstrates that a machine learning algorithm can effectively contribute to better understanding and anticipating road accident severity, provided one remains aware of its limitations and continues to enrich it with finer and more relevant data.
