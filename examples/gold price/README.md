# Predicting Gold Prices: A Journey with Machine Learning

## Introduction:
Gold has always been a focal point of interest for investors and market analysts. In this article, we will explore how to predict gold prices using machine learning. Our goal is to develop a powerful model that can guide us through the complex world of financial data.

## Data Set
The data set used in our analysis includes time series data of GLD, as well as SPX, USO, SLV, and EUR/USD. The data spans a broad time range starting from 2008.

## Data Set and Preprocessing:
Our analysis was conducted on a data set that includes GLD (gold) prices along with different financial assets such as SPX, USO, SLV, and EUR/USD. The data set spans a broad time range starting from 2008. We began by calculating the correlation matrix of our data set, which helped us understand the relationship of GLD with other financial assets.

<img width="912" alt="Ekran Resmi 2024-01-30 22 22 28" src="https://github.com/lacteavian/ML/assets/69520906/070d6189-b125-4bb1-8354-2160c6ac1a5a">

+ **GLD and SPX (S&P 500 Index):** The correlation analysis reveals a very weak positive relationship between GLD and SPX.
  ![gld_spx_plot](https://github.com/lacteavian/ML/assets/69520906/c0aeb75a-c8da-4805-b185-2c321dfcf39e)

+ **GLD and USO (Oil):**  A weak negative correlation has been identified between GLD and USO.
  ![gld_uso_plot](https://github.com/lacteavian/ML/assets/69520906/86ba378b-338e-4c4a-b5a1-03b36fd16a0e)

+ **GLD and SLV (Silver):** The analysis shows a strong positive correlation between GLD and SLV. This suggests that movements in silver prices generally coincide with similar directions in gold prices.
  ![gld_slv_plot](https://github.com/lacteavian/ML/assets/69520906/3daa0927-edc2-4d61-9fdd-d7dcba2b62e0)

+ **GLD and EUR/USD:** There is a very weak negative correlation between GLD and EUR/USD.
![gld_eurusd_plot](https://github.com/lacteavian/ML/assets/69520906/13968369-f814-45e5-86a0-44b749fadf25)

## Model Selection and Training:
For our model, we chose the Random Forest Regressor, a tool that is both powerful and flexible for regression. This model combines multiple decision trees to provide more accurate and robust predictions. We divided our data set into training and testing sets and trained the Random Forest Regressor model on this data.

## Model Evaluation:
To assess the performance of our model, we used various metrics. Among these, the most important were the Mean Squared Error (MSE) and Mean Absolute Error (MAE). These metrics helped us understand how well our model predicted the prices in the data set.

## Conclusion:
Machine learning is a powerful tool for unraveling the complexities of financial markets and predicting future price movements. The model we developed using the Random Forest Regressor showed promising results in predicting gold prices. However, further work is needed to improve the model.


