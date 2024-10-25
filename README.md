## Nairobi Office Price Prediction

### Overview
This project implements a machine learning model to predict office prices in Nairobi based on office size. 
The model uses simple linear regression with gradient descent to establish the relationship between office 
size and price, providing insights into the Nairobi commercial real estate market.

### Project Description
The project focuses on developing a predictive model for office prices in various locations across Nairobi. 
Using features such as office size, the model learns patterns in the real estate market to make price predictions. 
This implementation serves as a practical example of machine learning applications in real estate valuation.

### Dataset
The dataset is provided as a Microsoft Excel Comma Separated Values File (.csv) containing information about office 
properties in Nairobi.
- File type: CSV (Comma Separated Values)
- Filename: ``nairobi_office_prices.csv``

### Machine Learning Approach
The implementation uses:

- **Algorithm**: Linear Regression
- **Learning Method**: Gradient Descent
- **Performance Metric**: Mean Squared Error (MSE)
- **Training Process**: 10 epochs with random parameter initialization
- **Features Used**: Office Size (primary predictor)
- **Target Variable**: Office Price

### Project Structure
```
nairobi-office-price-prediction/
│
├── data/
│   └── nairobi_office_prices.csv
│
├── src/
│   └── linear_regression.py
│
└── README.md
```
### Requirements
- Python 3.x
- NumPy
- Pandas
- Matplotlib

### Usage
1. Clone the repository:
```angular2html
git clone https://github.com/Wendy-Lagho/office-price-predictor-ml.git
```
2. Install required packages:
```angular2html
pip install numpy pandas matplotlib
```
3. Run the model:
```angular2html
python src/linear_regression.py
```

### Author
Wendy Lagho
