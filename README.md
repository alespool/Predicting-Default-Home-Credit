## Project Outline:

This project focuses on the analysis of lending club  data.

### Objectives

In this project, we aim to accomplish the following objectives:

- Perform exploratory data analysis (EDA) on the dataset and its tables to understand the relationships between various features.
- Conduct statistical inference to determine if borrowers with high levels of debt to income ratio are more associated with a default.
- Apply an ensemble of machine learning models to predict whether a borrower has a default with the highest recall.

### Dataset Overview

The data is provided by [Home Credit](http://www.homecredit.net/about-us.aspx), a service dedicated to provided lines of credit (loans) to the unbanked population. Predicting whether or not a client will repay a loan or have difficulty is a critical business need, and Home Credit is hosting this competition on Kaggle to see what sort of models the machine learning community can develop to help them in this task.

There are 7 different sources of data:

- **application_train/application_test:** the main training and testing data with information about each loan application at Home Credit. Every loan has its own row and is identified by the feature SK_ID_CURR. The training application data comes with the TARGET indicating 0: the loan was repaid or 1: the loan was not repaid.
- **bureau:** data concerning client's previous credits from other financial institutions. Each previous credit has its own row in bureau, but one loan in the application data can have multiple previous credits.
- **bureau_balance:** monthly data about the previous credits in bureau. Each row is one month of a previous credit, and a single previous credit can have multiple rows, one for each month of the credit length.
- **previous_application:** previous applications for loans at Home Credit of clients who have loans in the application data. Each current loan in the application data can have multiple previous loans. Each previous application has one row and is identified by the feature SK_ID_PREV.
- **POS_CASH_BALANCE:** monthly data about previous point of sale or cash loans clients have had with Home Credit. Each row is one month of a previous point of sale or cash loan, and a single previous loan can have many rows.
- **credit_card_balance:** monthly data about previous credit cards clients have had with Home Credit. Each row is one month of a credit card balance, and a single credit card can have many rows.
- **installments_payment:** payment history for previous loans at Home Credit. There is one row for every made payment and one row for every missed payment.

This diagram shows how all of the data is related:


![image](https://storage.googleapis.com/kaggle-media/competitions/home-credit/home_credit.png)

Moreover, we are provided with the definitions of all the columns (in `HomeCredit_columns_description.csv`) and an example of the expected submission file. 


## Table of Contents

- [Exploratory Data Analysis](#Exploratory-Data-Analysis)
- [Feature Encoding](#Featuer-Encoding)
- [Statistical Inference](#Statistical-Inference)
- [Feature Engineering](#Featuer-Engineering)
- [Training model](#Training-Model)
- [Conclusions](#conclusions)

## Exploratory Data Analysis

In this section, we explore the dataset by performing the following:

- Data summary and statistics
- Correlation analysis between variables
- Data visualization to identify key trends

## Statistical Inference

We conduct statistical inference to answer a specific question: Are younger people more associated with the onset of default? We explore this question using non-parametric tests.

- Hypothesis Testing: We use Mann-Whitney U test to determine the difference.

## Feature Engineering

Before fitting an ensemble of machine learning models, we prepare the dataset by:

- Creating several new features, both polynomials and based on domain knowledge.
- Preprocessing ordinal and nominal categorical fetures.

## Predictions:

- We used LightGBM to predict the default outcome, achieving a 78% ROC AUC score.

![ROC_AUC](images\roc_auc.png)

- The model was then given different thresholds to lessen the number of false negatives, resulting in 70% total defaulters catched.

![Model_Probabilities](images\model_threshold.png)

- Another model built on top of the default predictions, which also uses LightGBM, was used to predict the credit score of a customer, with a 3.6 RMSE.

![True_Values](images\regression_credit_score.png)

## Model Deployment:

- The model we created gets exported to Google Cloud for easier retrieval and remote usage.

# Conclusions:

We found during the EDA that there is an association between age of the person applying and the probability of default, as well as a negative correlation between EXT_SOURCE 3, 2, 1 with the probability of default.

We further confirmed the first finding by running a non-parametric distribution test between younger and older people, which confirmed the first have a significantly higher probability of default. This was corroborated by a bootstrap and confidence intervals.

In the end, we were able to fit a model with a 78% ROC AUC score 

We used a LightGBM, which has been fine-tuned using Optuna. 

After checking the feature importances exported from our model, we can conclude that our model is good at predicting the default status for people. If the stakeholders wanted to improve the targetting of loan borrowers, they could focus on understanding the relationships of their clients with metrics such:

- The external source code;
- How many credit terms (length of payment in months);
- The age of the applicant;

Our model would be an acceptable starting point for various tasks:

- **Initial Screening:** The model can serve as an initial filter in the loan application process to quickly identify low-risk applicants. Since the precision for no default is high, it can reduce the workload for human analysts by flagging applications that are very likely to be safe.
- **Risk Assessment Tool:** It can be part of a broader risk assessment toolkit, providing one data point among many that analysts consider when evaluating loan applications.
- **Identifying Patterns:** The model might reveal important patterns or features associated with defaults, which can inform policy or strategy even if the model itself isn't used for automated decision-making.
- **Targeted Marketing:** The model can help in identifying characteristics of customers who are likely to not default, which can be useful for marketing campaigns aimed at the most promising prospects.
- **Customer Support:** For existing customers, the model can be used to flag accounts that may need proactive outreach or support to prevent default.

We also created a second model, which builds on the first one and can be used to assess the risk score of a certain credit. This is also implemented using a LightGBM model, and is evaluated using RMSE. We obtained a RMSE value of 3.62, which, for the range associated to the credit risk score, is low and appropriate, showing a good predictive power.

This prediction was driven mainly by 3 categories of variables:

- The expected loss from the credit;
- The amount of credit and that of the goods to purchase;
- The external sources (just like in the earlier classification);

This model could be used for similar points as the previous one, especially as it builds on top of it, therefore the recommendations remain the same.

## Improvements:

- **Feature Engineering** Revisit the features used in the model to see if additional relevant information can be added or irrelevant features can be removed.
- **Cost-sensitive Learning:** Incorporate the financial impact of false positives and false negatives into the model training process.

## Getting Started

To reproduce this analysis, follow these steps:

**Dependencies**:
- Python 3.11+  
- Required libraries listed in the requirements.txt file

**Usage**:
- Clone this repository.
- Install the required dependencies using `pip install -r requirements.txt`.
- Download the dataset from Kaggle place it in the 'data/raw_data' folder.
- Run the provided Python script.

## License

This project is licensed under the [MIT License](LICENSE).

For any inquiries, please feel free to contact us via email at [alessionespoli.97@gmail.com](mailto:alessionespoli.97@gmail.com).
