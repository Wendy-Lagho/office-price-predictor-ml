import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data/nairobi_office_prices.csv')
X = df['SIZE'].values
y = df['PRICE'].values

def mean_squared_error(y_true, y_pred):
    """
    Compute Mean Squared Error between true and predicted values
    """
    return np.mean((y_true - y_pred) ** 2)

def gradient_descent(X, y, m, c, learning_rate=0.0001, epochs=10):
    """
    Perform gradient descent to find optimal parameters
    Args:
        X: input feature (office size)
        y: target values (price)
        m: initial slope
        c: initial intercept
        learning_rate: learning rate for gradient descent
        epochs: number of training iterations
    Returns:
        m, c: optimized parameters
        errors: list of errors per epoch
    """
    n = len(X)
    errors = []

    for epoch in range(epochs):
        # Make predictions
        y_pred = m * X + c

        # Calculate error
        error = mean_squared_error(y, y_pred)
        errors.append(error)
        print(f"Epoch {epoch + 1}, Error: {error:.2f}")

        dm = (-2 / n) * np.sum(X * (y - y_pred))
        dc = (-2 / n) * np.sum(y - y_pred)

        m = m - learning_rate * dm
        c = c - learning_rate * dc

    return m, c, errors

# Set random initial values
np.random.seed(42)
initial_m = np.random.randn()
initial_c = np.random.randn()

# Train the model
final_m, final_c, errors = gradient_descent(X, y, initial_m, initial_c)

# Plot the results
plt.figure(figsize=(12, 6))

# Plot the data points
plt.scatter(X, y, color='blue', label='Data points')

# Plot the line of best fit
X_line = np.array([min(X), max(X)])
y_line = final_m * X_line + final_c
plt.plot(X_line, y_line, color='red', label='Line of best fit')

plt.xlabel('Office Size (sq. ft.)')
plt.ylabel('Price')
plt.title('Nairobi Office Prices: Linear Regression with Gradient Descent')
plt.legend()
plt.grid(True)
plt.show()

# Predict price for 100 sq. ft.
prediction_size = 100
predicted_price = final_m * prediction_size + final_c
print(f"\nPredicted price for {prediction_size} sq. ft.: {predicted_price:.2f}")
print(f"\nFinal model parameters:")
print(f"Slope (m): {final_m:.4f}")
print(f"Intercept (c): {final_c:.4f}")