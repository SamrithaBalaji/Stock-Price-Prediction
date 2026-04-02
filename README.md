📈 Stock Price Prediction using Machine Learning (Time Series Analysis)
📌 Project Overview

This project focuses on predicting stock prices using historical time series data and machine learning techniques. It demonstrates how past stock trends can be analyzed to estimate future price movements.

🎯 Objective

To build a machine learning model that learns patterns from historical stock data and predicts future stock prices.

🧠 Methodology
Collected stock data using Yahoo Finance (yfinance)
Performed data preprocessing and visualization
Applied MinMax scaling to normalize data
Created time-based sequences using sliding window technique
Trained a Linear Regression model
Evaluated model performance using error metrics
Visualized actual vs predicted stock prices
🛠️ Tech Stack
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
yfinance
📊 Features
Real-time stock data collection
Data visualization (trend & moving average)
Time series preprocessing
Machine learning model training
Prediction of future stock prices
Graph comparison of actual vs predicted values
📁 Project Structure

stock-price-prediction/
│
├── data/
│ └── stock_data.csv
│
├── outputs/
│ ├── stock_trend.png
│ ├── moving_average.png
│ ├── actual_vs_predicted.png
│ └── predictions.csv
│
├── main.py
├── requirements.txt
└── README.md

📈 Results

The model predicts stock prices based on past trends and visualizes the comparison between actual and predicted values. The closeness of the curves indicates the effectiveness of the model.

⚠️ Limitations

Stock prices are influenced by external factors such as market news, economic conditions, and global events. Hence, predictions may not always be accurate.

🚀 Future Improvements
Use advanced models like LSTM for better accuracy
Incorporate news sentiment analysis
Include additional financial indicators

👩‍💻 Author
Samritha
