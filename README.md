### Class Disbalance

Select metrics that take into account the imbalance of classes for example Cohen kappa, Matthew Correlation Coefficent
Choose a class balancing algorithm - oversampling: SMOTE and others, also we can try data generation approaches (GAN)

### Features

#### device_name

Requires a separate analysis, division into at least 2 features (name, device). Name and device statistics are unknown, according to
names, the number of categories can be quite large, this can be a problem, we may need a feature reduction scheme, or their  transformation for example PCA, autencoders or something similar. 

This feature have many missing values, we need to think about handling this issue.

#### app_category

This feature have NANs, we need to think about processing this problem

This features is distributed unevenly, it is possible to exclude features with low frequency, grouping into one or more groups
We can think about data generation (GANs or something similar).

#### ad_category

This feature have NANs, we need to think about processing this problem

This features is distributed unevenly, it is possible to exclude features with low frequency, grouping into one or more groups
We can think about data generation (GANs or something similar).

#### click

Click feature has low variability, better to exclude it from model building.

#### interaction_with_app

This feature have non-normal distribution, we need to consider this when working with classifiers that are sensitive to this requirement.
Think about the method of normalizing this feature

### Categorical variables

We need to choose a scheme for converting categories to a number (OneHot, Embeddings etc), there are many approaches.
There are open source libraries in python (CategoryEncoders).

### Feature Correlation 

Check correlations between features

### Feature Importance

The importance of features can be calculated in various ways. Chi-Squared, Mutual Information, Catboost feature importance, shap values, etc

### Model Selection

The first model to try is CatBoost - it can work with categories out of the box, fast, many different settings, works with the GPU. 
Problems: overfit, so it is desirable to compare with logistic regression. If the difference in accuracy is small, then we can think about tweaking LogReg. In any case, it is better to compare two models, linear and non-linear.
