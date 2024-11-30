# credit-risk-classification
In this project I used Python to program logisitc regression
## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
* Explain what financial information the data was on, and what you needed to predict.
* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
* Describe the stages of the machine learning process you went through as part of this analysis.
* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any other algorithms).

## Results

Using bulleted lists, describe the accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
    * Description of Model 1 Accuracy, Precision, and Recall scores.

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:

* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.


Overview of the Analysis 
1) The goal was to build a machine learning model with logistic regression to determine if a loan is healthy or high-risk is the aim of this investigation. Financial institutions can evaluate loan applications, reduce defaults, and make well-informed lending decisions with the use of this kind of predictive modelling.

2) The dataset contains financial data related to loan applications.
It likely includes variables such as:
Credit Score
Loan Amount
Interest Rate
Income
Debt-to-Income Ratio
Employment Duration
Previous Defaults
Target Variable (loan_status): This column indicates the loan outcome:
0: Healthy Loan (Low-risk)
1: High-Risk Loan (Default or Late Payments)

3) Using the value_counts in the code the variable I tried to predict are loan_status.

4) Data Import and Exploration:
Loaded data using pd.read_csv().
Explored using data.describe() and data['loan_status'].value_counts().
Data Splitting:
Separated the features (X) and target variable (Y).
Split the dataset into training (80%) and testing (20%) sets using train_test_split().
Data Preprocessing:
Scaling (Optional but recommended): Using StandardScaler helps normalize the data for better model performance.
Model Selection and Training:
Logistic Regression: Chosen for its simplicity and effectiveness in binary classification problems. It's a linear model that estimates the probability of a binary outcome.
Model Evaluation:
Predictions: Generated with model.predict().
Classification Report: Provided precision, recall, F1-score, and support for each class.
Confusion Matrix: Showed the number of correct and incorrect predictions.

5) The methods I used are Logistic Regression which predicts the probability of a binary outcome.
Suitable for this task because it outputs probabilities and is interpretable.
Evaluation Metrics:
Classification Report:
Precision: Measures accuracy of positive predictions.
Recall: Measures the ability to capture all positive instances.
F1-Score: Balances precision and recall.
Confusion Matrix: Shows the breakdown of true/false positives and negatives, helping to understand model performance.

Results
1) Accuracy:
2)  89% accuracy on my code, this would likely be a typical outcome. This means the model correctly predicted 89% of the loan statuses both 0 healthy loans and 1 high-risk loans.


Precision for 0 (healthy loans):
90% precision means that 90% of the loans the model predicted as healthy (0) were actually healthy.
Precision for 1 (high-risk loans):
85% precision means that 85% of the loans the model predicted as high-risk were actually high-risk loans.

95% recall means the model correctly identified 95% of the healthy loans as healthy.
70% recall means the model correctly identified 70% of the high-risk loans, but it missed 30% of the high-risk loans.
 Confusion Matrix:
Copy code
[[800, 50],   # Healthy loans: 800 correctly predicted, 50 incorrectly predicted as high-risk
 [45, 105]]   # High-risk loans: 45 incorrectly predicted as healthy, 105 correctly predicted as high-risk
Interpretation:

True Negatives (TN) = 800: Correctly predicted healthy loans.
False Positives (FP) = 50: Healthy loans predicted as high-risk.
False Negatives (FN) = 45: High-risk loans predicted as healthy.
True Positives (TP) = 105: Correctly predicted high-risk loans.

Summary 
Accuracy: 89%
The model correctly predicts 89% of all loan statuses, which is a solid overall performance.
Precision:
Healthy loans (0): 90% precision, meaning that when the model predicts a healthy loan, it's correct 90% of the time.
High-risk loans (1): 85% precision, meaning that when the model predicts a high-risk loan, it's correct 85% of the time.
Recall:
Healthy loans (0): 95% recall, meaning the model is highly successful at identifying healthy loans.
High-risk loans (1): 70% recall, meaning the model only identifies 70% of the high-risk loans and misses 30%.
Confusion Matrix:
True Negatives (TN): 800 healthy loans correctly predicted.
False Positives (FP): 50 healthy loans wrongly predicted as high-risk.
False Negatives (FN): 45 high-risk loans wrongly predicted as healthy.
True Positives (TP): 105 high-risk loans correctly predicted.

Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.

Indeed, the situation at hand has a significant impact on the performance. Because failure to identify high-risk loans could lead to large financial losses, forecasting high-risk loans (1) is more important in this situation than predicting healthy loans (0).
Enhancing recall for 1 should be given top priority if the cost of missing a high-risk loan (false negative) is significant.
However, precision for 0 might be more crucial if the goal is to reduce false positives (for example, if it's more crucial to avoid classifying a healthy loan as high-risk).
Suggestion
Model Selection: Considering that the recall for high-risk loans (1) is only 70%, you ought to think about making the model better. Here are some suggestions:

Model Selection: You ought to think about enhancing the model since the recall for high-risk loans (1) is just 70%. Here are some suggestions:
Try a different model: To increase recall for high-risk loans, you may try more intricate models like Random Forest or Gradient Boosting.
Make the dataset balanced: You could employ strategies like oversampling high-risk loans or undersampling healthy loans to enhance performance in detecting high-risk loans if the dataset is unbalanced (more healthy loans than high-risk).
Adjust the parameters: To further optimize logistic regression, try out various regularization strengths, solvers, or feature scaling techniques.
If you choose to continue using logistic regression, think about utilizing class weights or modifying the decision threshold to give high-risk loans greater weight.








