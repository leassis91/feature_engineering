## Tips on Discovering New Features

* Understand the features. Refer to your dataset's data documentation, if available.
* Research the problem domain to acquire **domain knowledge**. If your problem is predicting house prices, do some research on 
real-estate for instance. Wikipedia can be a good starting point, but books and journal articles will often have the best 
information.
* Study previous work. Solution write-ups from past Kaggle competitions are a great resource.
* Use data visualization. Visualization can reveal pathologies in the distribution of a feature or complicated relationships
that could be simplified. Be sure to visualize your dataset as you work through the feature engineering process.

## Mathematical Transforms

```autos['stroke_ratio'] = autos.stroke / autos.bore```


The more complicated a combination is, the more difficult it will be for a model to learn, like this formula for an 
engine's "displacement", a measure of its power:

``` autos["displacement"] = (
    np.pi * ((0.5 * autos.bore) ** 2) * autos.stroke * autos.num_of_cylinders)
```

Data visualization can suggest transformations, often a "reshaping" of a feature through powers or logarithms. The distribution of WindSpeed in US Accidents is highly skewed, for instance. 
In this case the logarithm is effective at normalizing it:

If the feature has 0.0 values, use np.log1p (log(1+x)) instead of np.log

``` 
accidents["LogWindSpeed"] = accidents.WindSpeed.apply(np.log1p)
```


Plot a comparison

```
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
sns.kdeplot(accidents.WindSpeed, shade=True, ax=axs[0])
sns.kdeplot(accidents.LogWindSpeed, shade=True, ax=axs[1]);
```

![image](https://user-images.githubusercontent.com/67332395/179639144-0df7fce5-bdd0-4f71-903a-5f0b5b6adb56.png)

## Counts

Features describing the presence or absence of something often come in sets, the set of risk factors for a disease, say. You can aggregate such features by creating a count.

These features will be binary (1 for Present, 0 for Absent) or boolean (True or False). In Python, booleans can be added up just as if they were integers.


```
roadway_features = ["Amenity", "Bump", "Crossing", "GiveWay", "Junction", "NoExit", "Railway", "Roundabout", "Station", "Stop", "TrafficCalming", "TrafficSignal"]

accidents["RoadwayFeatures"] = accidents[roadway_features].sum(axis=1)
accidents[roadway_features + ["RoadwayFeatures"]].head(10)
```



## More tips on creating Features

It's good to keep in mind your model's own strengths and weaknesses when creating features. Here are some guidelines:

- Linear models learn sums and differences naturally, but can't learn anything more complex.
- Ratios seem to be difficult for most models to learn. Ratio combinations often lead to some easy performance gains.
- Linear models and neural nets generally do better with normalized features. Neural nets especially need features scaled to values not too far from 0. Tree-based models (like random forests and XGBoost) can sometimes benefit from normalization, but usually much less so.
- Tree models can learn to approximate almost any combination of features, but when a combination is especially important they can still benefit from having it explicitly created, especially when data is limited.
- Counts are especially helpful for tree models, since these models don't have a natural way of aggregating information across many features at once.
