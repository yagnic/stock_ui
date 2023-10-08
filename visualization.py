import streamlit as st
import pandas as pd
import plotly.express as px
import yfinance as yf
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

# Function to fetch historical data for a ticker
def get_data(ticker):
    tickerData = yf.Ticker(ticker)
    # Get the data for the last two years. Change the dates if you need a different time period
    tickerDf = tickerData.history(period='1d', start='1980-01-01', end='2023-10-04')
    return tickerDf, ticker

# Function to fetch and cache stock data
@st.cache(allow_output_mutation=True)
def get_cached_data():
    # Fetch the NIFTY 50 table from Wikipedia
    table = pd.read_html('https://en.wikipedia.org/wiki/NIFTY_500', header=0)
    # Depending on the structure of the page, this index might need adjustment

    # Extract the ticker symbols.
    ticker_symbols = table[2]['Symbol'].values.tolist()
    ticker_symbols = [col + ".NS" for col in ticker_symbols]

    # Fetch data for all tickers using multiprocessing
    num_processes = cpu_count()
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap_unordered(get_data, ticker_symbols), total=len(ticker_symbols)))

    # Process results and create a dictionary with the data
    ticker_data = {}
    for result in results:
        if result:
            tickerDf, ticker = result
            ticker_data[ticker] = tickerDf

    predictions_data = pd.read_csv("bogada_1.csv")
    eval_data = pd.read_csv("eval_2.csv")

    return ticker_data, predictions_data,eval_data

# Streamlit App
st.title('Stock Data Dashboard')

# Fetch and cache stock data
ticker_data,predictions_data,eval_data = get_cached_data()

# Sidebar for selecting a stock
selected_stock = st.sidebar.selectbox('Select a stock', list(ticker_data.keys()))




# Create variables for accuracy range
min_accuracy, max_accuracy = eval_data['accuracy'].min(), eval_data['accuracy'].max()

# Create a range slider in the sidebar to select accuracy range
min_accuracy, max_accuracy = st.sidebar.slider("Select Accuracy Range", eval_data['accuracy'].min(), eval_data['accuracy'].max(), (min_accuracy, max_accuracy))

# Option to refresh data with custom date range
refresh_data = st.sidebar.checkbox('Refresh Data')
if refresh_data:
    start_date = st.sidebar.date_input('Start Date', pd.to_datetime('1980-01-01'))
    end_date = st.sidebar.date_input('End Date', pd.to_datetime('2023-10-04'))
    if start_date < end_date:
        tickerData = yf.Ticker(selected_stock)
        ticker_data[selected_stock] = tickerData.history(period='1d', start=start_date, end=end_date)
    else:
        st.sidebar.error('End Date must be after Start Date')

# Display stock data
if selected_stock in ticker_data and ticker_data[selected_stock].shape[0]>0:

    
    # Define your score
    score = round(eval_data.loc[eval_data.ticker==selected_stock].accuracy.values[0] * 100)
    
    # Display the score using a styled box with Markdown and HTML
    st.title("Scorecard")
    
    # Define your scores
    score1 = round(eval_data.loc[eval_data.ticker==selected_stock].accuracy.values[0] * 100)
    score2 = round(eval_data.loc[eval_data.ticker==selected_stock].total_pl.values[0] * 100)
    
    # Display the scores using a styled box with Markdown and HTML
    st.subheader("Metrics")
    
    scores_box = f"""
<div style="display:flex;">
    <div style="flex-grow: 1; text-align:center; padding:10px; background-color:#f0f0f0; border-radius:5px; border: 1px solid #000;">
        <h2>Accuracy</h2>
        <p>{score1}</p>
    </div>
    <div style="flex-grow: 1; text-align:center; padding:10px; background-color:#f0f0f0; border-radius:5px; border: 1px solid #000;">
        <h2>Total Profit Loss</h2>
        <p>{score2}</p>
    </div>
</div>
"""

    
    st.markdown(scores_box, unsafe_allow_html=True)
    

    st.subheader(f"Data for {selected_stock}")
    data = ticker_data[selected_stock].reset_index()
    predictions_data_ticker = predictions_data.loc[predictions_data.ticker==selected_stock]
    data["Date"] = pd.to_datetime(data.Date)
    data["day"] = data.Date.dt.day
    data["year"] = data.Date.dt.year
    data["%change"] = (data.Close - data.Open) / data.Open
    data["%hl"] = (data.High - data.Low) / data.Low
    data["Close_1"] = data.Open.shift(1)
    data["Gap"] = (data.Open - data.Close_1) / data.Close_1

    st.write(data)

    # Plot interactive line chart for all data
    # Subsection for Interactive Line Plot with Date Filter

    
    # Date Filter
    st.subheader('Interactive Line Plot for All Data')
    
    # Date Filter
    date_range = st.date_input('Select Date Range', [data['Date'].min(), data['Date'].max()])

    data['Date'] = data['Date'].dt.tz_localize(None) 
    
    # Convert the selected date range to NumPy datetime64 objects
    
    
    filtered_data = data[(data['Date'] >= pd.to_datetime(date_range[0])) & (data['Date'] <= pd.to_datetime(date_range[1]))]
    
    # Create an interactive line plot for the selected date range
    fig = px.line(filtered_data, x='Date', y=['%change', '%hl', 'Gap'], labels={'Date': 'Date', 'value': 'Percentage'},
                  title='Percentage Change, Percentage High-Low, and Gap')
    st.plotly_chart(fig)
    # Filter data for the last 30 days
    # last_30_days = datetime.now() - timedelta(days=30)
    # data_last_30_days = data[data['Date'] >= last_30_days]

    # Filter data for the last 30 days
    last_30_days = datetime.now() - timedelta(days=30)
    data['Date'] = data['Date'].dt.tz_localize(None)  # Remove timezone information for comparison
    data_last_30_days = data[data['Date'] >= last_30_days]

    predictions_data_ticker.Matching_Date = pd.to_datetime(predictions_data_ticker.Matching_Date)

    predictions_data_temp = predictions_data_ticker.loc[predictions_data_ticker.Matching_Date.isin(data_last_30_days.Date)]

    predictions_data_temp = predictions_data_temp[["Matching_Date","predictions"]]
    predictions_data_temp.columns = ["Date","predictions"]

    data_last_30_days = pd.merge(data_last_30_days,predictions_data_temp,on=["Date"],how="left")
    
    # Plot interactive line chart for the last 30 days
    st.subheader('Interactive Line Plot for the Last 30 Days')
    
    fig_last_30_days = px.line(data_last_30_days, x='Date', y=['%change', '%hl', 'Gap',"predictions"], labels={'Date': 'Date', 'value': 'Percentage'},
                      title='Percentage Change, Percentage High-Low, and Gap for the Last 30 Days')
    st.plotly_chart(fig_last_30_days)



    st.subheader('Comparison of performance for previous 30 days')

    # Create an interactive line plot for the selected date range
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Additional data for the last 30 days (sample data)
    additional_data = data_last_30_days[["Date", "%change", "predictions"]]
    
    # Add a red line at zero
    fig.add_shape(
        type='line',
        x0=additional_data['Date'].min(),
        x1=additional_data['Date'].max(),
        y0=0,
        y1=0,
        line=dict(color='red', width=2),
    )
    
    # Add bubbles for predictions and true values
    fig.add_trace(
        go.Scatter(
            x=additional_data['Date'],
            y=additional_data['predictions'],
            mode='markers',
            name='Predictions',
            marker=dict(color='blue', size=10),  # Change color and size as needed
        ),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(
            x=additional_data['Date'],
            y=additional_data['%change'],
            mode='markers',
            name='True',
            marker=dict(color='green', size=10),  # Change color and size as needed
        ),
        secondary_y=True,
    )
    
    fig.update_layout(
        title_text="Close Price Trend for the Last 30 Days with Predictions",
        xaxis_title="Date",
        yaxis_title="Percentage Change / Predictions",
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Predictions", secondary_y=False)
    fig.update_yaxes(title_text="True", secondary_y=True)
    
    st.plotly_chart(fig)



    



    # Plot Close price trend for the last 30 days
    st.subheader('Close Price Trend for the Last 30 Days')
    fig_close_price = px.line(data_last_30_days, x='Date', y=['Close','Volume'], labels={'Date': 'Date', 'Close': 'Close Price'},
                             title='Close Price Trend for the Last 30 Days')

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Scatter(x=data_last_30_days.Date, y= data_last_30_days.Close, name="Close"),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=data_last_30_days.Date, y= data_last_30_days.Open, name="Open"),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=data_last_30_days.Date, y= data_last_30_days.High, name="High"),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=data_last_30_days.Date, y= data_last_30_days.Low, name="Low"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=data_last_30_days.Date, y=data_last_30_days.Volume, name="Volume"),
        secondary_y=True,
    )
    
    # Add figure title
    fig.update_layout(
        title_text="lose Price Trend for the Last 30 Days"
    )
    
    # Set x-axis title
    fig.update_xaxes(title_text="Date")
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Close", secondary_y=False)
    fig.update_yaxes(title_text="Volume", secondary_y=True)
    st.plotly_chart(fig)
else:
    st.write(f"Data not available for {selected_stock}")











# You can now use min_accuracy and max_accuracy to filter stocks based on accuracy range
filtered_stocks = eval_data[(eval_data['accuracy'] >= min_accuracy) & (eval_data['accuracy'] <= max_accuracy)]['ticker'].unique()

# Display filtered stocks in the sidebar
st.sidebar.subheader("Filtered Stocks by Accuracy Range")
if len(filtered_stocks) > 0:
    st.sidebar.write("Stocks with selected accuracy range:")
    st.sidebar.write(filtered_stocks)
else:
    st.sidebar.write("No stocks found with selected accuracy range.")


    

    

#     # Plot interactive line chart for the last 30 days
#     st.subheader('Interactive Line Plot for the Last 30 Days')
#     fig_last_30_days = px.line(data_last_30_days, x='Date', y=['%change', '%hl', 'Gap'], labels={'Date': 'Date', 'value': 'Percentage'},
#                   title='Percentage Change, Percentage High-Low, and Gap for the Last 30 Days')
#     st.plotly_chart(fig_last_30_days)
# else:
#     st.write(f"Data not available for {selected_stock}")

# To run the Streamlit app, save this script to a .py file and run it using: streamlit run your_script.py
