import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv(r'C:\Users\RAJESWARI\Downloads\coffee_shop_revenue.csv')
print(data.head())

# Get information about the DataFrame
data.info()

# Prepare the data
x = data.iloc[:, 0:5].values
y = data.iloc[:, 6].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Create and train the model
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Make predictions
y_pred_train = regressor.predict(x_train)
y_pred_test = regressor.predict(x_test)


# Evaluate the model
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
print("Mean Squared Error (Train):", mse_train)
print("Mean Squared Error (Test):", mse_test)

r2 = r2_score(y_test, y_pred_test)
print("R² Score:", r2)

# Visualize the training set results
plt.scatter(x_train[:, 0], y_train, color='red', label='Actual')  # Only first feature
plt.plot(x_train[:, 0], regressor.predict(x_train), color='blue', label='Predicted')
plt.title('Revenue vs no_of_customers_per_day (Training set)')
plt.xlabel('no_of_customers_per_day')
plt.ylabel('Revenue')
plt.legend()
plt.show()

# Visualize the test set results
plt.scatter(x_test[:, 0], y_test, color='red', label='Actual')  # Only first feature
plt.plot(x_train[:, 0], regressor.predict(x_train), color='blue', label='Predicted')
plt.title('Revenue vs no_of_customers_per_day (Test set)')
plt.xlabel('no_of_customers_per_day')
plt.ylabel('Revenue')
plt.legend()
plt.show()

# Plot Mean Squared Error
mse_values = [mse_train, mse_test]
labels = ['Train', 'Test']

plt.bar(labels, mse_values, color=['blue', 'orange'])
plt.title('Mean Squared Error')
plt.ylabel('MSE')
plt.show()



# Comparison Plot
plt.scatter(y_test, y_pred_test, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Actual vs Predicted')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()