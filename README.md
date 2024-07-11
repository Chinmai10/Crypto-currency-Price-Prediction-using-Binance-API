# CryptoCurrency Price Prediction and Analysis

## Introduction:
This project aims to analyze historical Cryptocurrency price data ( here Bitcoin is taken as example ) and build predictive models for forecasting future price movements. It covers various aspects of data preparation, visualization, time series forecasting, and machine learning modeling using Python libraries such as pandas, numpy, matplotlib, seaborn, Plotly, Facebook Prophet, and XGBoost.

### Data Preparation:
- The initial step involves obtaining data from binance.com using the **Binance Python API**.
- The retrieved data includes historical price information for Bitcoin (BTC) ( taken as example ) against USDT (Tether) with a daily interval.
- The data is formatted into a pandas DataFrame for further processing.
- Timestamps are converted into human-readable dates for better interpretation.
- Finally, the processed data is exported to a CSV file for future use.

### Data Visualization:
- After data preparation, the project proceeds to visualize the Bitcoin closing prices over time.
- Matplotlib and Seaborn libraries are utilized to create a line plot showing the distribution of Bitcoin closing prices.
- Additionally, Plotly is used to generate a candlestick chart, providing a graphical representation of price movements (open, high, low, close) over time.

### Time Series Forecasting with Prophet:
- The project involves forecasting future Bitcoin prices using the **Facebook Prophet** library.
- The historical Bitcoin price data is loaded from the previously prepared CSV file.
- Prophet is initialized and fitted to the historical data to learn patterns and trends.
- Future dates are generated for forecasting, and predictions are made using the trained model.
- The forecasted results, including predicted prices and uncertainty intervals, are visualized using Prophet's built-in plotting functionalities.

### Regression Modeling with XGBoost:
- The project explores regression modeling to predict future Bitcoin prices.
- **XGBoost** (Extreme Gradient Boosting) is chosen as the regression model due to its effectiveness in handling tabular data.
- Feature engineering is performed by creating lagged features (past price values) as input features and using the next day's price as the target variable.
- The dataset is split into training and validation sets using time series cross-validation.
- XGBoost regressor is trained on the training data and evaluated on the validation set using **Root Mean Square Error (RMSE)**.
- Predictions are made on the validation set, and the model's performance is assessed based on RMSE.

### Binary Classification Modeling with XGBoost:
- The project extends the regression modeling to **binary classification** to predict the direction of price movement (up or down).
- A binary target variable is created based on whether the next day's price is higher or lower than the current day's price.
- XGBoost classifier is trained on the binary classification task using the same lagged features as input.
- The model's performance is evaluated based on **Area Under the Receiver Operating Characteristic Curve (ROC AUC)**.
- Confusion matrix and ROC curve are visualized to assess the classifier's performance in predicting price movements.

Each part of the project contributes to a comprehensive analysis of Bitcoin price data, covering data retrieval, visualization, time series forecasting, and predictive modeling using machine learning techniques.

## References:

- [Binance Python API Documentation](https://python-binance.readthedocs.io/en/latest/)
- [Facebook Prophet Documentation](https://facebook.github.io/prophet/docs/quick_start.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/)
- Additional references for time series analysis, machine learning modeling, and cryptocurrency market analysis can be found in the code comments.
