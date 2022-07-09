# Feature Engineering Course by Kaggle

I'll take some notes I find interesting and resourceful from this [course](https://www.kaggle.com/code/ryanholbrook/what-is-feature-engineering) offered by Kaggle.

This course teach about one of the most important steps on the way to build a great machine learning model: *feature engineering*.

I'll learn how to

  - determine which features are the most important with **MUTUAL INFORMATION**
  - invent new features in severeal real-world problem domains
  - encode high-cardinality categoricals with a **TARGET ENCODING**
  - create segmentation features with **K-MEANS CLUSTERING**
  - decompose a dataset's valriation into features with **PRINCIPAL COMPONENT ANALYSIS**

along with lots of exercises.

## Goal of Feature Engineering

Simply make your data better suited to the problem at hand.
Consider "apparent temperature" measures like the heat index and the wind chill. You might perform feature engineering to:
  
  - improve a model's predictive performance
  - reduce computation or data needs
  - improve interpretability of the results

## Guiding Principle of Feature Engineering

For a feature to be useful, it must have a relationship to the target that your model is able to learn. 

***

# 1.0 Mutual Information

## **Where** do we even begin when encounter a new dataset?

A great first step is to construct a ranking with **feature utility metric**, a function measuring associations between a feature and the target. Then, you can choose a smaller set of the most useful features to develop initially and have more confidence that your time will be well spent.

The metric here we'll use is called "mutual information". It is a lot like correlation in that it measures a relationship between two quantities. The advantage of mutual information is that it can detect *any kind of relationship*, while *correlation only detects linear relationships*.

MI is especially useful at the start of feature development when you might not know what model you'd like to use yet. It is:
  - easy to use and interpret,
  - computationally efficient,
  - theoretically well-founded,
  - resistant to overfitting and
  - able to detect any kind of relationship.
  
## **What** it measures?

Describes relationships in terms of *uncertainty*. The mutual information between two quantities is a msure of the extent to which knowledge of one quantity reduces uncertainty about the other. 
Example from the Ames Housing Data. The figure shows the relationship between the exterior qualiuty of a house and the price it sold for. Each point represents a house.

  ![image](https://user-images.githubusercontent.com/67332395/178091757-00234b52-080b-4d51-97a5-a2952b5d7f0a.png)
  
  *Knowing the exterior quality of a house reduces uncertainty about its sale price.*
  
  From the figure, we can see that knowing the value of *ExterQual* should make you more certain about the corresponding *SalePrice* -- each category of *ExterQual* tends to concentrate SalePrice to within a certain range. The mutual information that ExterQual has with SalePrice is the average reduction of uncertainty in SalePrice taken over the four values of ExterQual. Since *Fair* occurs less often than *Typical*, for instance, Fair gets less weight in the MI score.
  
  - MI can help you to understand the relative potential of a feature as a predictor of the target, considered by itself.
  - It's possible for a feature to be very informative when interacting with other features, but not so informative all alone. MI can't detect interactions between features. It is a univariate metric.
  - The actual usefulness of a feature depends on the model you use it with. A feature is only useful to the extent that its relationship with the target is one your model can learn. Just because a feature has a high MI score doesn't mean your model will be able to do anything with that information. You may need to transform the feature first to expose the association.
      
      
## Scikit-learn Library for MI

MI treat discrete features differently from continuous features. We need to tell which are which. So, float types are *ot discrete*. Categoricals can be treated as discrete by giving them label encoding.
Sklearn has two mutual information metrics in its feature_selection module.
  
  - for real valued targets: **mutual_info_regression**
  - for categorical targets: **mutual_info_classif**
    
```
from sklearn.feature_selection import mutual_info_regression

def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

mi_scores = make_mi_scores(X, y, discrete_features)
mi_scores[::3]  # show a few features with their MI scores
```

Add a bar plot to make comparisons easier:

```
def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores)
```
![mutual information scores](https://user-images.githubusercontent.com/67332395/178092068-45b22c7e-379c-42f7-b491-00a6dae17c5a.png)
