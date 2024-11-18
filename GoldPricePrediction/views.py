from django.shortcuts import render
from django.http import HttpResponse
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import login as auth_login
from django.http import JsonResponse
import plotly.offline as opy
from plotly.graph_objs import Scatter
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from .models import GoldPrice,NewsArticle
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import joblib
from dateutil.relativedelta import relativedelta
from .models import GoldPrice
from django.db.models import Max
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import decimal
import requests



def input_form(request):
    


    # static data
    # articles = [{
    #         "title": "Gold Hits Record High as Analysts Predict Continued Strength Amid Economic Uncertainty",
    #         "urlToImage": "https://media.zenfs.com/en/us.finance.gurufocus/0af84412a4c60cfbffd7a59f080bb18e",
    #         "publishedAt": "2024-10-22",
    #     },            
        
    # ]
   
    # articles=[]

    #     url = "https://newsapi.org/v2/everything?q=gold%20market&apiKey=f19192fe014a408581ba7dae464b335e"
    # response = requests.get(url)
    # news_data = response.json()  # Get JSON response from the API
    # if news_data['status'] == 'ok':
    #     for article in news_data['articles'][:4]:  # Extract first 4 articles
    #         # Parse the 'publishedAt' field to a Python datetime object
    #         published_at_str = article['publishedAt']  # Format: '2024-10-17T14:03:45Z'
    #         published_at = datetime.strptime(published_at_str, '%Y-%m-%dT%H:%M:%SZ').date()  # Convert to date object

    #         # Create and save the article in the database
    #         news_article = NewsArticle(
    #             title=article['title'],
    #             url=article['url'],
    #             url_to_image=article.get('urlToImage', ''),
    #             published_at=published_at  # Store only the date part
    #         )
    #         news_article.save()  # Save to the database

    #         # Optionally, you can append the article to the list to display
    #         articles.append({
    #             'title': article['title'],
    #             'url': article['url'],
    #             'urlToImage': article.get('urlToImage', ''),
    #             'publishedAt': published_at.strftime('%Y-%m-%d')  # Format for display
    #         })

    articles = NewsArticle.objects.all()
    articles_data = list(
    articles.values('title', 'url', 'url_to_image', 'published_at')
    )

    for article in articles_data:
        article['published_at'] = article['published_at'].strftime('%Y-%m-%d')  # Format as 'YYYY-MM-DD'

    # Save the serialized data to the session
    request.session['articles'] = articles_data
    
    return render(request, 'GoldPricePrediction/news.html',{'articles':articles_data})

def process_input(request):
    if request.method == 'POST':
        # Get the input value from the form
        user_input = request.POST.get('user_input')
        model_name = "yiyanghkust/finbert-tone"  # Pretrained FinBERT model for financial sentiment
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # Tokenize the input text
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=512)

        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits.squeeze().tolist()
        
        score = logits[2] - logits[0]

        print(score)

        score=round(score,2)

        
        # Pass the relevant article to the template
    



        return render(request, 'GoldPricePrediction/news.html', {'user_input':user_input,'sentiment_score': score,'articles': request.session['articles']})

    return render(request, 'GoldPricePrediction/news.html')





def three_day_prediction(df,last_date_features,linear):
    days_to_predict=3
    current_date = df.index[-1]
    current_s3 = last_date_features['S_3'].iloc[0]
    current_s9 = last_date_features['S_9'].iloc[0]

    predictions = []

    for day in range(1, days_to_predict + 1):

        input_features = pd.DataFrame([[current_s3, current_s9]], columns=['S_3', 'S_9'])
        predicted_price = linear.predict(input_features)[0]
        predicted_price = round(predicted_price, 2)

        # Append the prediction
        predictions.append({
            'date': (current_date + timedelta(days=1)).strftime('%Y-%m-%d'),
            'predicted_price': predicted_price
        })

        # Use this predicted price to compute rolling averages for the next iteration
        current_date += timedelta(days=1)
        new_row = pd.DataFrame({
            'price': [predicted_price],
            'S_3': [current_s3],  # Will be updated after adding the new row
            'S_9': [current_s9],  # Will be updated after adding the new row
            'next_day_price': [None]  # Not used, placeholder
        }, index=[current_date])

        # Add new row to the DataFrame
        df = pd.concat([df, new_row])

        # Update rolling averages
        df['S_3'] = df['price'].rolling(window=3).mean()
        df['S_9'] = df['price'].rolling(window=9).mean()

        # Update current rolling averages for the next prediction
        current_s3 = df.loc[current_date, 'S_3']
        current_s9 = df.loc[current_date, 'S_9']

    print("Predictions for the next few days:")
    for prediction in predictions:
        print(prediction)
    
    return predictions


    






def predict():
    linear=joblib.load('lr_model.pkl')
    
    today = datetime.now(tz=pytz.timezone('US/Eastern')).date()
    last_year = today - relativedelta(years=1)

    price_data = GoldPrice.objects.filter(date__gte=last_year, date__lte=today)

    price_data_list=list(price_data.values('date','price'))

    df=pd.DataFrame(price_data_list)
    # print(df.head())

    # df['date']=df['date']

    last_date_db=df['date'].iloc[-1]+ pd.Timedelta(days=1)
    print("to add value from this date to db: ",last_date_db) 
    print("and today's date is: ",today)

        # checking the db what date it contains
    if last_date_db<today:
        df_recent = yf.download('GLD',start=last_date_db, end=today,auto_adjust=True)
        df_recent = df_recent.reset_index()
        df_recent.drop(columns=['Open','High','Low','Volume'],inplace=True)
        df_recent.rename(columns={'Date': 'date', 'Close': 'price'}, inplace=True)
        df_recent['date']=df_recent['date'].dt.date
        df_recent['price']= df_recent['price'].round(2)
        
        start_date=last_date_db
        end_date=today

        complete_date_range=pd.date_range(start=start_date,end=end_date,freq='D').date

        df_recent = df_recent.set_index('date').reindex(complete_date_range).reset_index()
        df_recent.rename(columns={'index': 'date'}, inplace=True)


        df_recent['price'] = df_recent['price'].fillna(method='ffill')


        print(df_recent.head())

     


        # print(df_recent.head())
        # print(df_recent.shape)
       

        # # final check on what was the last date present in the db and all dates whose data is fetched
        # df_recent_filtered = df_recent[df_recent['date'] > last_date_db.date()]

        # # concatenated both the dataframes
        df=pd.concat([df,df_recent],ignore_index=True)
        print(df.tail(10))


        # # updating the database also with recent fetched data
        gold_price_objects = [
          GoldPrice(date=row['date'], price=row['price'])
            for _, row in df_recent.iterrows()
        ]

        for obj in gold_price_objects:
            if not obj.date or not obj.price:
                print(f"Invalid data: {obj.date}, {obj.price}")
                continue

        GoldPrice.objects.bulk_create(gold_price_objects)
    
    
    df.set_index('date',inplace=True)


    df['S_3'] = df['price'].rolling(window=3).mean()
    df['S_9'] = df['price'].rolling(window=9).mean()

    df['next_day_price'] = df['price'].shift(-1)

    last_row_date=df.index[-1]
    closing_price_date=last_row_date.strftime("%d/%m/%y")
    closing_price=df["price"].iloc[-1]
    date_next_day= (df.index[-1] + timedelta(days=1)).strftime("%d/%m/%y")

    last_date_features = pd.DataFrame([df.loc[df.index[-1], ['S_3', 'S_9']].values], columns=['S_3', 'S_9'])


    df = df.dropna() #today's price will be dropped because it doesnt contains the next day price i.e y_test

    X = df[['S_3', 'S_9']]
    y=df['next_day_price']


    predicted_price = linear.predict(X)

    predicted_price = pd.DataFrame(predicted_price, index=y.index, columns=['price'])
    predicted_price['price'] = predicted_price['price'].round(2)
    predicted_price['close'] = y



    #price column contains predicted_price for the next day based on current day information
    #close contains the actual price of the next day

    x_data = predicted_price.index
    y_data_predicted = predicted_price['price']
    y_data_actual = predicted_price['close']



    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x_data, y=y_data_predicted,
                        mode='lines',
                        name='Predicted Price'))
    fig.add_trace(go.Scatter(x=x_data, y=y_data_actual,
                        mode='lines',
                        name='Actual Price'))
    PredictionPlot_div = opy.plot({
        'data': [Scatter(x=x_data, y=y_data_predicted, mode='lines', name='Predicted Price', opacity=0.8),
        Scatter(x=x_data, y=y_data_actual, mode='lines', name='Actual Price', opacity=0.8)],
        'layout': {'title': 'Predicted VS Actual Price', 'xaxis': {'title': 'Date'}, 'yaxis': {'title': 'Gold ETF price (in $)'}}
    }, auto_open=False, output_type='div')


    #========== CUMULATIVE RETURNS ==========
    # gold = pd.DataFrame()

   

    data = {'signal': []}  # Start with an empty list or any data you want to add
    df_p = pd.DataFrame(data)
    df_p['signal'] = np.where(predicted_price.price.shift(1) < predicted_price.price,"Buy","No Position")


    # gold['price'] = df[t:]['Close']
    # gold['predicted_price_next_day'] = predicted_price['price']
    data = {'signal': []}  
    df_p = pd.DataFrame(data)
    df_p['signal'] = np.where(predicted_price.price.shift(1) < predicted_price.price,"Buy","No Position")

    gold_df = pd.DataFrame()
    gold_df['gold_returns'] = predicted_price['price'].pct_change().shift(-1)
        
    gold_df['strategy_returns'] = gold_df['gold_returns']
    ratio = '%.2f' % (gold_df['strategy_returns'].mean() / (gold_df['strategy_returns'].std() * (252**0.5)))
    # x_data = gold.index
    # y_data = ((gold['strategy_returns']+1).cumprod()).values


    r2_score = linear.score(X, y)*100
    r2_score = float("{0:.2f}".format(r2_score))

    predicted_next_day_price=linear.predict(last_date_features)[0]
    predicted_next_day_price = round(predicted_next_day_price, 2)
    # s3_value = last_date_features['S_3'].iloc[0]
    # s9_value = last_date_features['S_9'].iloc[0]

    # new_row = pd.DataFrame({
    # 'price':[closing_price],
    # 'S_3': [s3_value],
    # 'S_9': [s9_value],
    # 'next_day_price': [predicted_next_day_price]
    # }, index=[last_row_date])

    # df = pd.concat([df, new_row])


    p_list=three_day_prediction(df,last_date_features,linear)
    p1=p_list[1]['predicted_price']
    p2=p_list[2]['predicted_price']

    day1=(last_row_date+ timedelta(days=2)).strftime("%d/%m/%y")
    day2=(last_row_date+ timedelta(days=3)).strftime("%d/%m/%y")

 


    # print(df.tail())
    

    
    return day2,p2,day1,p1,predicted_next_day_price,date_next_day,closing_price,closing_price_date,PredictionPlot_div,r2_score,df_p,ratio



# Plot the complete graph
def PlotClosingPrice():
    

    price_data = GoldPrice.objects.all()
    
    price_data_list=list(price_data.values('date','price'))
    df=pd.DataFrame(price_data_list)
    df['date']=pd.to_datetime(df['date'])
    df.set_index('date',inplace=True)

    df = df.dropna() 

    x_data=df.index
    y_data=df['price']
    ClosingPricePlot_div = opy.plot({
    'data': [Scatter(x=x_data, y=y_data, mode='lines', name='test', opacity=0.8, marker_color='green')],
    'layout': {'title': 'Gold ETF Price Series', 'xaxis': {'title': 'Date'}, 'yaxis': {'title': 'Gold ETF price (in $)'}}
    }, output_type='div')

    return ClosingPricePlot_div





def gold_price_tracker(p1, p2, p3):
    today = datetime.now(tz=pytz.timezone('US/Eastern')).date()
    last_3_month = today - relativedelta(months=3)

    # Fetch historical data
    price_data = GoldPrice.objects.filter(date__gte=last_3_month, date__lte=today)
    price_data_list = list(price_data.values('date', 'price'))

    df = pd.DataFrame(price_data_list)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Prepare historical data for plotting
    x_data = df.index
    y_data = df['price']

    # Predict future dates
    last_date = df.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, 4)]
    future_prices = [p1, p2, p3]

    # Create a DataFrame for future data
    future_data = pd.DataFrame({'date': future_dates, 'price': future_prices})
    future_data.set_index('date', inplace=True)

    # Combine historical and future data
    combined_df = pd.concat([df, future_data])

    x_combined = combined_df.index
    y_combined = combined_df['price']

    # Add a vertical line for today's date
    gold_price_tracker_div = opy.plot({
        'data': [
            Scatter(
                x=x_combined, 
                y=y_combined, 
                mode='lines', 
                name='Historical Prices', 
                opacity=0.8, 
                marker_color='green'
            ),
            Scatter(
                x=future_dates, 
                y=future_prices, 
                mode='markers',  # Only markers (dots)
                name='Predicted Prices', 
                marker=dict(size=8, color='red')
            ),
        ],
        'layout': {
            'title': 'Gold Price Tracker',
            'xaxis': {'title': 'Date'},
            'yaxis': {'title': 'Gold ETF price (in $)'},
            'shapes': [
                {
                    'type': 'line',
                    'x0': today,
                    'x1': today,
                    'y0': min(y_combined),  # Bottom of the graph
                    'y1': max(y_combined),  # Top of the graph
                    'line': {
                        'color': 'black',
                        'width': 2,
                        'dash': 'dot',  # Dashed vertical line
                    },
                }
            ],
            'plot_bgcolor': '#f0f0f0',  # Set plot background color (light gray)
        'paper_bgcolor': '#FFFED3'
        }
        
    }, output_type='div')

    return gold_price_tracker_div






def RegressionModel():
    linear=joblib.load('lr_model.pkl')
    RegressionModelFormula = "Gold ETF Price (y) = %.2f * 3 Days Moving Average (x1) + %.2f * 9 Days Moving Average (x2) + %.2f (constant)" % (linear.coef_[0], linear.coef_[1], linear.intercept_)

    return RegressionModelFormula

# def PredictionPlot():
#     return PredictionPlot_div

# # def r2_scoreCalculate():
#     # R square

#     r2_score = linear.score(X[t:], y[t:])*100
#     r2_score = float("{0:.2f}".format(r2_score))
#     return r2_score

# def CumulativeReturns():
#     return CumulativeReturns_div

# def SharpeRatioCalculate():
#     return '%.2f' % (gold['strategy_returns'].mean()/gold['strategy_returns'].std()*(252**0.5))



# def GetClosingPriceDate():
#     return Df.index[-1].strftime("%d/%m/%y")

# def GetNextDay():
#     return (Df.index[-1] + timedelta(days=1)).strftime("%d/%m/%y")


def home(request):
    day2,p2,day1,p1,predicted_next_price,next_date,closing_price,closing_price_date,predict_plt,r2_score,df_p,ratio=predict()
    context = {
        'ClosingPricePlot_div' : PlotClosingPrice(),
        'gold_price_tracker': gold_price_tracker(predicted_next_price,p1,p2),
        'PredictionPlot_div' : predict_plt,
        'SharpeRatio' : ratio,
        'Signal' : df_p['signal'].iloc[-1],
        'Day2':day2,
        'Predicted_Price_Day2':p2,
        'Day1':day1,
        'Predicted_Price_Day1':p1,
        'NextDate':next_date,
        'PredictedPrice' : predicted_next_price,
        'ClosingPrice' : closing_price,
        'ClosingDate' : closing_price_date,
        
    }

    return render(request, 'GoldPricePrediction/home.html', context)

# def base(request):
#     return render(request, 'GoldPricePrediction/base.html')

def information(request):
    day2,p2,day1,p1,predicted_next_price,next_date,closing_price,closing_price_date,predict_plt,r2_score,df_p,_=predict()
    context = {
        'RegressionModelFormula' : RegressionModel(),
        'r2_score' : r2_score,
    }
    return render(request, 'GoldPricePrediction/information.html', context)


def plots_view(request):
    day2,p2,day1,p1,predicted_next_price,next_date,closing_price,closing_price_date,predict_plt,r2_score,df_p,_=predict()

    context = {
        'ClosingPricePlot_div' : PlotClosingPrice(),
        'PredictionPlot_div' : predict_plt,
    }


    return render(request, 'GoldPricePrediction/plots.html',context )



# def gold_price(request):
#     # api_key = 'goldapi-11lmmjrlpvgste1-io'  
#     api_key = 'goldapi-47ksm3brlnle-iobakugo'  # Replace 'YOUR_API_KEY_HERE' with your actual API key
#     gold_api_url = 'https://www.goldapi.io/api/XAU/USD'
#     headers = {'x-access-token': api_key}

#     try:
#         response = requests.get(gold_api_url, headers=headers)
#         if response.status_code == 200:
#             gold_data = response.json()
#             gold_price = gold_data['price']
#             context = {'gold_price': gold_price}
#             return render(request, 'GoldPricePrediction/gold_price.html', context)

#     except requests.RequestException as e:
#         print(f"API Error: {e}")
    
#     return redirect('home')  













# # Function to set a cookie
# def set_cookie(request):
#     response = render(request, 'GoldPricePrediction/home.html')
#     # Set a cookie named 'gold_prediction' with a value
#     response.set_cookie('gold_prediction', 'predicted_value')
#     return response


# from django.shortcuts import render
# from django.http import HttpResponse

# # Function to get a cookie
# def get_cookie(request):
#     # Get the value of the 'gold_prediction' cookie
#     predicted_value = request.COOKIES.get('gold_prediction')
#     return HttpResponse(f'Predicted value from cookie: {predicted_value}')






def add_gold_historical_data(request):
    Df = yf.download('GLD', '2008-01-01', '2024-11-13', auto_adjust=True)
    Df = Df.reset_index()
    Df['Date']=Df['Date'].dt.date


    start_date = Df['Date'].min()
    end_date = Df['Date'].max()
    complete_date_range = pd.date_range(start=start_date, end=end_date, freq='D').date

    Df = Df.set_index('Date').reindex(complete_date_range).reset_index()
    Df.rename(columns={'index': 'Date'}, inplace=True)

    Df['Close'] = Df['Close'].fillna(method='ffill')

    valid_data = Df.dropna(subset=['Date', 'Close'])

    gold_price_objects = [
        GoldPrice(date=row['Date'], price=row['Close'])
        for _, row in valid_data.iterrows()
    ]

    GoldPrice.objects.bulk_create(gold_price_objects)

    return JsonResponse({'status': 'success', 'message': 'Gold historical data added successfully'})


def train_model(request):
    today = datetime.now(tz=pytz.timezone('Asia/Kolkata'))
    last_year = today - relativedelta(years=1)

    price_data = GoldPrice.objects.filter(date__lt=last_year)

    price_data_list=list(price_data.values('date','price'))
    df=pd.DataFrame(price_data_list)
    
    print(df.head())
    print(df.shape)
    print(df.tail())

    df['date']=pd.to_datetime(df['date'])
    df.set_index('date',inplace=True)


    df['S_3'] = df['price'].rolling(window=3).mean()
    df['S_9'] = df['price'].rolling(window=9).mean()

    df['next_day_price'] = df['price'].shift(-1)
    df = df.dropna() 
    X = df[['S_3', 'S_9']]
    y=df['next_day_price']

    linear = LinearRegression().fit(X, y)
    joblib.dump(linear,'lr_model.pkl')

    return JsonResponse({'status': 'success', 'message': 'Model trained successfully and saved'})










# from django.http import HttpResponse

# def set_user_timezone(request):
#     # Get user's preferred timezone (for example, let's assume 'US/Eastern')
#     user_timezone = 'US/Eastern'  # You might fetch this from user settings

#     response = render(request, 'GoldPricePrediction/home.html')

#     # Set 'user_timezone' cookie with the user's preferred timezone
#     response.set_cookie('user_timezone', user_timezone)

#     return response

# def get_user_timezone(request):
#     # Retrieve 'user_timezone' cookie value
#     user_timezone = request.COOKIES.get('user_timezone')

#     return HttpResponse(f"User's Timezone: {user_timezone}")


# def info_page(request):
#     # Add any context data or logic needed to render Info.html
#     return render(request, 'GoldPricePrediction/info.html')


# def info_page(request):
#     # Fetch gold price data from your database or an API
#     # Example data - modify this with your actual data retrieval
#     gold_prices = [
#         {'date': '2023-12-01', 'price': 1500},
#         # ... (Fetch data for other dates)
#     ]

#     # Send data to the template
#     context = {
#         'gold_prices': gold_prices,
#         # Add other context data needed for your page
#     }
#     return render(request, 'GoldPricePrediction/info.html', context)













# def login_page(request):
#     if request.method == 'POST':
#         form = AuthenticationForm(request, request.POST)
#         if form.is_valid():
#             user = form.get_user()
#             auth_login(request, user)
#             return redirect('home')  # Redirect to home page after successful login
#     else:
#         form = AuthenticationForm()
#     return render(request, 'login.html', {'form': form})




# def login_page(request):
#     if request.method == 'POST':
#         username = request.POST.get('username')
#         password = request.POST.get('password')
#         User.objects.create(username=username, password=password)
#         # Here, you might want to add authentication logic
#         # For simplicity, this example just saves the user directly to the database
#         return redirect('home')  # Redirect to the home page after login

#     return render(request, 'login.html')