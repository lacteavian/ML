import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load the data
file_path = 'datasets/gld_price.csv'
gld_data = pd.read_csv(file_path)

# Calculating the correlation matrix
correlation_matrix = gld_data[['SPX', 'GLD', 'USO', 'SLV', 'EUR/USD']].corr()

# Plotting the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

print(correlation_matrix['GLD'])


def plot_relation(df, date_name, target, feature_list):
    for i in range(len(feature_list)):
        # Creating subplots for GLD and SLV
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Making a plot for GLD
        ax1.plot(df[date_name].values, df[target].values, color='gold', label=target)
        ax1.set_xlabel('Date')
        ax1.set_ylabel(target, color='gold')
        ax1.tick_params('y', colors='gold')

        # Creating a second y-axis for SLV
        ax2 = ax1.twinx()
        ax2.plot(df[date_name].values, df[feature_list[i]].values, color='silver', label=feature_list[i])
        ax2.set_ylabel(feature_list[i], color='silver')
        ax2.tick_params('y', colors='silver')

        # Adding title and showing the plot
        plt.title(f"{target} and {feature_list[i]} over time")
        fig.tight_layout()
        plt.show()


feature_list = ["SPX", "USO", "SLV", "EUR/USD"]
plot_relation(gld_data, "Date", "GLD", feature_list)

X = gld_data.drop(['Date', 'GLD'], axis=1)
y = gld_data['GLD']

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=1)

regressor = RandomForestRegressor(n_estimators=100)
regressor.fit(X_train, Y_train)
test_data_prediction = regressor.predict(X_test)
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared error : ", error_score)
