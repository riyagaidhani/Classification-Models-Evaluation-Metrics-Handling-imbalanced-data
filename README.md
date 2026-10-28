# Classification-Models-Evaluation-Metrics-Handling-imbalanced-data
# AI/ML Task 4: Classification Models, Evaluation Metrics & Handling Imbalanced Data

Internship task (Maincrafts Technology). A binary classification system built on the Breast Cancer dataset and evaluated with metrics beyond accuracy

## Dataset
Breast Cancer Wisconsin dataset (scikit-learn built-in)
- 569 samples, 30 features
- Target: `0 = Malignant` (212 samples), `1 = Benign` (357 samples)

## Tools Used
Python, Jupyter Notebook, pandas, NumPy, scikit-learn, matplotlib, seaborn

## Approach
1. Loaded the dataset and checked the class distribution
2. Stratified 80/20 train-test split (`random_state=42`)
3. Scaled features using `StandardScaler` (fit on train data only, to avoid data leakage)
4. Trained a baseline Logistic Regression model
5. Evaluated using confusion matrix, classification report, ROC curve and AUC
6. Handled class imbalance using `class_weight="balanced"`
7. Compared against a Decision Tree Classifier

## Results
Malignant is treated as the positive class for Precision, Recall and F1.

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.982 | 0.976 | 0.976 | 0.976 | 0.995 |
| Logistic Regression (balanced) | 0.956 | 0.911 | 0.976 | 0.943 | 0.995 |
| Decision Tree | 0.912 | 0.848 | 0.929 | 0.886 | 0.916 |

Baseline Logistic Regression confusion matrix: 41 TP, 1 FN, 1 FP, 71 TN.

## Key Learnings
- **Metric choice:** In medical diagnosis, Recall matters most because a False Negative means a missed cancer case. F1-score and AUC give a fuller picture than accuracy alone.
- **Why accuracy is not enough:** On imbalanced data, a model can score high accuracy by favouring the majority class while missing the minority class.
- **Imbalance handling:** This dataset is only mildly imbalanced (about 63/37). `class_weight="balanced"` kept recall the same but lowered precision, so it did not improve overall results here.
- **Model comparison:** Logistic Regression was more stable and generalized better. The Decision Tree overfit (100% train accuracy vs about 91% test accuracy), though it is easier to interpret.

## Final Model Decision
**Logistic Regression (baseline)** is selected: it has the highest F1-score (0.976) and AUC (0.995), strong recall, and stable performance on unseen data
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook AI_ML_Task4_Classification_Evaluation.ipynb
#Maincrafts Technology
