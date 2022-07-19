## Intro

Unsupervised algorithms don't make use of a target; instead, their purpose is to learn some property of the data, to represent the strcuture of the features in a certain way.
It's like a *feature discovery* technique.

> ***Clustering** simply means the assigning of data points to **groups**, based upon how similar the points are to each other.*

A clustering algorithm makes "birds of a feather flock together".

When used for FEATURE ENGINEERING, we could attempt to discover groups of customers representing a makret segment, for instance, or geographiic areas that share similar weather patterns.
Adding a feature of cluster labels can help machine learning models untangle complicated relationsihps of space or proximity.

***

## Cluster Labels as a Feature

Clustering acts like traditional "binning" or "discretization" transform. Multiple features >> *vector quantization*, or just "multi-dimensional binning".

![image](https://user-images.githubusercontent.com/67332395/179777923-9e793804-1efc-494b-8b1c-2d5031dd2d6c.png)
***Left**: Clustering a single feature. **Right**: Clustering across two features.*

Added to a dataframe, a feature of cluster labels might look like this:

| Longitude | Latitude | Cluster |
|-----------| -------- | ------- |
| -93.619   | 42.054   | 3 |
| -93.619   | 42.053   | 3 | 
| -93.638   | 42.060   | 1 |
| -93.602   | 41.988   | 0 |



Remember: `Cluster` feature is **categorical.** Here, it's shown with a label encoding as a typical clusstering algorithms would produce; depending on your model, a one-hot encoding may be more appropriate.

The motivating idea for adding cluster labels is that the clusters will break up complicated relationships across features into simpler chunks. 
Our model can then just learn the simpler chunks one-by-one instead of having to learn the complicated whole all at once. It's a "divide and conquer" strategy.

![image](https://user-images.githubusercontent.com/67332395/179779409-7c70e24f-3561-44b7-9abf-0f846443a526.png)
_Clustering the YearBuilt feature helps this linear model learn its relationship to SalePrice._


The figure shows how clustering can improve a simple linear model. The curved relationship between the YearBuilt and SalePrice is too complicated for this kind of model -- it underfits. On smaller chunks however the relationship is almost linear, and that the model can learn easily.

***

## k-Means Clustering

Measures similarity using ordinary straight-line distance (Euclidean distance). It creates clusters by placing a number of points, called **centroids**, inside the feature-space. Each point in the datraset is assigned to the cluster of whichever centroid it's closedst to. The "k" in "K-means" is how many centroids (that is, clusters), it creates. You define the k yourself.

We will foucus on three parameters from scikit-learn's implementation: `n_clusters`, `max_iter` and `n_init`.
It's a simple two-step process. The algorithm starts by randomly initializing some predefined number (`n_clusters`) of centroids. It then iterates ver these two operations:

1. assign points to the nearest cluster centroid
2. move each centroi to minimize the distance to its points.

It iterates over these two steps until the centroids aren't moving anymore, or until some maximum number of iterations has passed (max_iter).

It often happens that the initial random position of the centroids ends in a poor clustering. For this reason the algorithm repeats a number of times (n_init) and returns the clustering that has the least total distance between each point and its centroid, the optimal clustering.

The animation below shows the algorithm in action. It illustrates the dependence of the result on the initial centroids and the importance of iterating until convergence.

![image](https://i.imgur.com/tBkCqXJ.gif)

You may need to increase the max_iter for a large number of clusters or n_init for a complex dataset. Ordinarily though the only parameter you'll need to choose yourself is n_clusters (k, that is). The best partitioning for a set of features depends on the model you're using and what you're trying to predict, so it's best to tune it like any hyperparameter (through cross-validation, say).
