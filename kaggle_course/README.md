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
