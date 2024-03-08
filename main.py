from flask import Flask, render_template, request
import os
import base64
import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from io import BytesIO
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

matplotlib.use('agg')

# getting the tickers of all companies listed in the S&P 500 and storing in a list
tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
tickers = list(tickers.Symbol)
# these two tickers have no data so are removed
tickers.remove('BF.B')
tickers.remove('BRK.B')
tickers.sort()
line_img_data = ''
heat_img_data = ''
predicted_img_data = ''
mse = 0.0
mae = 0.0

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def index():
    selected = tickers[0]
    company = ''
    industry = ''
    employees = ''
    country = ''

    get_ticker_data(selected)
    company, industry, employees, country = get_company_details(selected)

    if request.method == 'POST':
        selected = request.form.get('ticker')
        company, industry, employees, country = get_company_details(selected)
        get_ticker_data(selected)

    return render_template(
        'index.html',
        data = tickers,
        selected_ticker = selected,
        company_name = company,
        industry = industry,
        employees = employees,
        country = country,
        price_history = line_img_data,
        correlation = heat_img_data,
        predicted = predicted_img_data,
        accuracy_metric_mse = format(mse, ".3f"),
        accuracy_metric_mae = format(mae, ".3f")
    )

# getting details about the selected ticker
def get_company_details(ticker):
    company_name = yf.Ticker(ticker).info['longName']
    industry = yf.Ticker(ticker).info['industry']
    employees = yf.Ticker(ticker).info['fullTimeEmployees']
    country = yf.Ticker(ticker).info['country']
    return company_name, industry, employees, country

# read the CSV file to get share price data for the selected ticker
def get_ticker_data(ticker):
    ticker_file = f"{ticker}.csv"
    file_path = os.path.join("./data_files", ticker_file)
    df = pd.read_csv(file_path)
    prepare_data(df)

# prepare the data by dropping unnecessary columns and removing missing values
def prepare_data(ticker_data):
    df = ticker_data.drop(columns=['ticker', 'Adj Close', 'Date'], axis=1)
    df.dropna(inplace = True)
    train_model(df)

# training the model with the selected ticker's data and implementing linear regression
def train_model(ticker_data):
    global mse
    global mae

    # independent variables
    X = ticker_data[['Open', 'High', 'Low']]

    # dependant variables
    y = ticker_data['Close']

    # testing the model with 20% of data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # using the linear regression algorithm to model the relationship between the variables
    model = LinearRegression()
    model.fit(X_train, y_train)

    # make predictions
    prediction = model.predict(X_test)

    # measuring the accuracy of the prediction
    mse = metrics.mean_squared_error(y_test, prediction)
    mae = metrics.mean_absolute_error(y_test, prediction)

    # calling a method for creating the charts
    plot_data(ticker_data, y_test, prediction)

# creating a line chart which displays one year of share price history
def plot_line_chart(ticker_data):
    fig = Figure()
    data = fig.subplots()
    data.plot(ticker_data['Close'])
    data.set_title('One Year Closing Price History')
    data.set_xlabel('Trading Days')
    data.set_ylabel('Closing Price ($ USD)')
    img_buf = BytesIO()
    fig.savefig(img_buf, format='png')
    img_buf.seek(0)
    return base64.b64encode(img_buf.getbuffer()).decode('utf-8')

# creating a heatmap which displays the correlation value between data
def plot_heatmap(ticker_data):
    plt.figure()
    sns.heatmap(ticker_data.corr(), annot=True)
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    return base64.b64encode(img_buf.getbuffer()).decode('utf-8')

# creating a scatterplot which displays the actual share price vs the predicted share price
def plot_scatter_plot(actual, prediction):
    plt.figure()
    plt.scatter(actual, actual, label='Actual', color='blue', alpha=0.7)
    plt.scatter(actual, prediction, label='Predicted', color='green', alpha=0.7)
    plt.plot([min(actual), max(actual)], [min(actual), max(actual)], color='red', label='Regression Line')
    plt.xlabel("Actual Prices ($ USD)")
    plt.ylabel("Predicted Prices ($ USD)")
    plt.title("Actual vs Predicted Share Prices")
    plt.legend()
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    return base64.b64encode(img_buf.getbuffer()).decode('utf-8')

# method for creating the three charts
def plot_data(ticker_data, actual, prediction):
    global line_img_data
    global heat_img_data
    global predicted_img_data

    line_img_data = plot_line_chart(ticker_data)
    heat_img_data = plot_heatmap(ticker_data)
    predicted_img_data = plot_scatter_plot(actual, prediction)

    plt.close('all')
    
if __name__ == '__main__':
    app.run(debug=True)