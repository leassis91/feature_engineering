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

Let's take a closer look at a couple of these. 
As we might expect, the high-scoring ```curb_weight``` feature exhibits a strong relationship with ```price```, the target.


```
sns.relplot(x="curb_weight", y="price", data=df);
```

![image](https://user-images.githubusercontent.com/67332395/178113701-a8518420-4bab-476a-a5b8-1c42345e6a69.png)

The `fuel type` feature has a fairly low MI score, but as we can see from the figure, it clearly separates two `price` populations with different trends within the horsepower feature. This indicates that `fuel_type` contributes an interaction effect and might not be unimportant after all. Before deciding a feature is unimportant from its MI score, it's good to investigate any possible interaction effects -- domain knowledge can offer a lot of guidance here.

`sns.lmplot(x="horsepower", y="price", hue="fuel_type", data=df);`

![image](https://user-images.githubusercontent.com/67332395/178113894-24061050-5382-45dd-98ef-210ea8b4d240.png)

Time to exercise!
