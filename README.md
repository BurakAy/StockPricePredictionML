## About
Technologies Used: Python, Flask, yfinance, pandas, seaborn, matplotlib, scikit-learn, JavaScript

Flask web app for S&P 500 stock analysis, utilizing yfinance for data retrieval and linear regression for price prediction.
Model accuracy is assessed using mean squared error (MSE) and mean absolute error (MAE). 
Visualizations such as line charts, heatmaps, and scatter plots are presented for data analysis.
CSV data file creation processes is automated.

### Line chart generated
![image](https://github.com/BurakAy/StockPricePredictionML/assets/14030652/677b6cfd-2be7-4694-8d5c-b8e5c910a38c)

### Scatter plot generated
![image](https://github.com/BurakAy/StockPricePredictionML/assets/14030652/d97a5b4a-47f4-42d2-9154-e58ee5a5ff83)

### Heat map generated
![image](https://github.com/BurakAy/StockPricePredictionML/assets/14030652/099ebbec-8873-49c6-93af-d2399966f24e)

## User Guide
The application has been deployed as a Flask web application. 
1. To access the application, visit the following URL: https://stockpricepredictionml.onrender.com

*Note: The application is hosted on a free plan which spins down the instance with inactivity. This may result in slow or delayed loading of the app and charts. On the first visit to the URL, please allow up to one minute for the application to spin up on the hosting provider.

2. When the application has loaded, you should see the following user interface:
![image](https://github.com/BurakAy/StockPricePredictionML/assets/14030652/f1cdc408-a492-434d-b1d2-1ffbb031f6ff)

3. Select a stock ticker from the dropdown menu under ‘Tickers’ to generate the charts for the selected ticker.
![image](https://github.com/BurakAy/StockPricePredictionML/assets/14030652/e37e62ba-d194-4ce9-92b6-8935c56e9fa1)

4. You can view each generated chart by clicking on its associated button found under ‘Charts’.
![image](https://github.com/BurakAy/StockPricePredictionML/assets/14030652/799ee3f3-f095-4016-8ba5-fda78e311709)

5. You can find the mean squared error and mean absolute error values under the ‘Price Prediction’ chart selection.
![image](https://github.com/BurakAy/StockPricePredictionML/assets/14030652/f382ee0d-c804-4b22-8037-5b2dfc3c3b22)

