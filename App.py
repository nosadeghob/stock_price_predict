# uncomment the pickle !!!!!!!!!!!!!!!

import datetime, pickle # for saving data, for example the classification
import streamlit         as st
import numpy             as np 
import yfinance          as yf
import matplotlib.pyplot as plt #for graphing
from sklearn                 import preprocessing, svm
from sklearn.linear_model    import LinearRegression
from sklearn.model_selection import train_test_split

st.markdown("<h1 style='text-align: center; color: white;'>Stock Price Prediction</h1>", unsafe_allow_html=True)
st.set_option('deprecation.showPyplotGlobalUse', False)
#st.title('Stock Price Prediction')
plt.figure(figsize=(12, 6))

with st.sidebar:
    user_input = st.text_input('Enter Stock Ticker', 'AAPL')
    #user_start_date = st.text_input('Enter Start Date', '2010-01-01')
    #user_end_date = st.text_input('Enter end Date', '2022-01-01')
    user_start_date = st.date_input('Enter Start Date',datetime.date(2010, 1, 1))
    user_end_date = st.date_input('Enter end Date', datetime.date(2022, 12, 31))
    #prediction_days = st.number_input('Enter The Number Of Prediction Days')
    prediction_days = st.slider('Enter The Number Of Prediction Days', 1, 30, 10)

if user_start_date>user_end_date:
    # st.sidebar.write(":red[Error: the start date can't be after the end date !!!]")
    st.sidebar.markdown(f'<h1 style="background-color:#FFFFFF;text-align: center;color:#C34104;font-size:24px;">Error: the start date cant be after the end date !!!</h1>', unsafe_allow_html=True)

else:
    df = yf.download(user_input, user_start_date, user_end_date)

# describing data
subheader_output = 'Raw Dataframe from ' + str(user_start_date.year) + '-' + str(user_end_date.year) +' ('+ str(df.shape[0]) + ' Days)'
st.subheader(subheader_output) # used a variable because it cant handle more than 3 inputs
st.write(df)
#st.table(df)
#print(df.shape)

#visualizations
st.subheader('Closing price')
plt.title(user_input) #***************
plt.xlabel('Days')
plt.ylabel('Close Price')
plt.plot(df['Adj Close'], 'b')
#plt.show()
st.pyplot()

##############################################################

df = df.drop(['Close'], axis=1)

df['HL_PCT'] = (df['High'] - df['Low']) / df['Adj Close'] * 100.0 #PCT means percent 
# the "HL_PCT" is supposted to be "high - low"

df['PCT_CHANGE'] = (df['Adj Close'] - df['Open']) / df['Open'] * 100.0 # we multipliy it just to make it smaller
df = df[['Adj Close', 'HL_PCT', 'PCT_CHANGE', 'Volume']]

# describing data
subheader_output = 'New Dataframe from ' + str(user_start_date.year) + '-' + str(user_end_date.year)
st.subheader(subheader_output) # used a variable because it cant handle more than 3 inputs
st.write(df)

#visualizations
st.subheader('PCT_CHANGE & HL_PCT')
title = user_input + '(PCT_CHANGE & HL_PCT)'
plt.title(title) #***************
plt.xlabel('Days')
plt.ylabel('Close Price')
plt.plot(df['PCT_CHANGE'])
plt.plot(df['HL_PCT'], 'r')
plt.legend(["PCT_CHANGE", "HL_PCT"], loc ="lower right") # show the guide on the corner
#plt.show()
st.pyplot()

###################################

st.subheader('Closing Price vs Time chart with 100MA & 200MA') # 'MA' : mean average
ma100 = df['Adj Close'].rolling(100).mean()
ma200 = df['Adj Close'].rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Adj Close'], 'b')
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.legend(['Closing price', 'ma100', 'ma200'], loc ="lower right")
st.pyplot()
#st.pyplot(fig)

###################################

df.fillna(-99999, inplace = True) # fill means fill and na means NULL, so this line fills the empty places with the number -99999 becaues it does not have an effec on the output
#forecast_out = int(math.ceil(0.005*len(df))) # math.ceil rounds the numder to the nearest whole number. also "forcast_out" shows the days of guessing in advance, for example  if it says 30, it means its guessing 30 days in advance
# "0.1" is used to predect 10 percent of the data
forecast_col = 'Adj Close'
forecast_out = prediction_days
#print(forecast_out, "days")

df['lable'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['lable'], axis = 1)) # we use the drop to say anything except lable

X = preprocessing.scale(X) # Center to the mean (normalizing), takes a bit more time
X_future = X[-forecast_out:] # the missing data (future)
X = X[:-forecast_out] # the data we have

df.dropna(inplace = True)

Y = np.array(df['lable']) # so the "X" is out futures and the "Y" is out lable (what we guess)

#X = preprocessing.scale(X) # Center to the mean (normalizing), takes a bit more time

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2) # shuffeling the data and splitting it to training and testing data

clf = LinearRegression() # the process of classification. we can switch this functiong with other Regression functions like "svm.SVR()"
clf.fit(x_train, y_train) # training (or fitting) based on the trainig data
Svm = svm.SVR()
Svm.fit(x_train, y_train)

clf_test_predict = clf.predict(x_test)
st.subheader('Testing Data vs the "linear Regression" prediction') # 'MA' : mean average
plt.plot(y_test, 'b')
plt.plot(clf_test_predict, 'r')
plt.legend(['Testing Data', 'Prediction'], loc ="lower right")
st.pyplot()

########## pickle #############

# with open('linearregression.pickle', 'wb') as f: # saving the classified (trained) data so we dont need to train it every time
#     pickle.dump(clf, f)

# pickle_in = open('linearregression.pickle', 'rb') # reading and loading the trained data into the clf again
# clf = pickle.load(pickle_in)

#######################

accuracy = clf.score(x_test, y_test) # testing the classification using the testing data. "score" returns a score indicating the accuracy (squerd error)
accuracy_svm = Svm.score(x_test, y_test)

# print('accuracy: ', accuracy)

# describing data 
st.subheader('The Accuracy Of The "linear regression" Model ') # used a variable because it cant handle more than 3 inputs
st.markdown(f'<h1 style="background-color:#FFFFFF;text-align: center;color:#FF0000;font-size:24px;">{accuracy}</h1>', unsafe_allow_html=True)
#st.write(accuracy)
#print(len(X), len(Y))

############################

forecast_set = clf.predict(X_future)
Title_forcast_set = '"linear regression" Prediction Of The Next ' + str(forecast_out) + ' Days'
st.subheader(Title_forcast_set) # used a variable because it cant handle more than 3 inputs

col1, col2, col3 = st.columns(3)
with col2:
    st.write(forecast_set)
    
#print(forecast_set) #************************$$$$$$$$$$$$$$$$$
df['Forecast'] = np.nan
#print(df['Forecast']) just NaN

############################

last_date = df.iloc[-1].name # finding the last date./ iloc is Purely integer-location based indexing for selection by position./ last position/ name is the data that is on the left of the dataframe
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day
#print(df['Forecast']) just NaN

#####################

for i in forecast_set: # to show the dates on the graph.
    next_date = datetime.datetime.fromtimestamp(next_unix) # The fromtimestamp() function is used to return the date corresponding to a specified timestamp
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) -1)] + [i] # "[i]" is the forecast_set value loc = Access a group of rows and columns by label(s) or a boolean array.
    #print(next_date)
#print(df['Forecast']) the last 16 indexes have values
#print(df.head())

#cach_for_end = df['Forecast']

### using side by side columns
col1, col2,  = st.columns(2)

with col1:
    st.subheader('Closing Price & Prediction')
    # df['Adj Close'].plot()
    # df['Forecast'].plot()
    plt.plot(df['Adj Close'], 'b')
    plt.plot(df['Forecast'], 'r')
    #plt.plot(forecast_set, 'g')
    plt.legend(['Closing price', 'Prediction'], loc ="lower right") # show the guide on the corner
    plt.xlabel('Date')
    plt.ylabel('Price')
    #plt.show()
    st.pyplot()

with col2:  
    st.subheader('Prediction')
    plt.plot(df['Forecast'], 'r')
    plt.xlabel('Date')
    plt.ylabel('Price')
    #plt.show()
    st.pyplot()
    
    ######################################### THIS IS THE SAME THING FOR SUPPORT VECTORE REGRESSION ###################################

Svm_test_predict = Svm.predict(x_test)
st.subheader('Testing Data vs the "Suppor Vector Regression" prediction') # 'MA' : mean average
plt.plot(y_test, 'b')
plt.plot(Svm_test_predict, 'g')
plt.legend(['Testing Data', 'Prediction'], loc ="lower right")
st.pyplot()

st.subheader('The Accuracy Of The"Support Vector Regression" Model') # used a variable because it cant handle more than 3 inputs
st.markdown(f'<h1 style="background-color:#FFFFFF;text-align: center;color:#008000;font-size:24px;">{accuracy_svm}</h1>', unsafe_allow_html=True)
#st.write(accuracy_svm)

forecast_set_svm = Svm.predict(X_future)
Title_forcast_set_svm = '"Support vector regression" Prediction Of The Next ' + str(forecast_out) + ' Days'
st.subheader(Title_forcast_set) # used a variable because it cant handle more than 3 inputs

col1, col2, col3 = st.columns(3)
with col2:
    st.write(forecast_set_svm)
    
#print(forecast_set_svm) #************************$$$$$$$$$$$$$$$$$
df['Forecast_svm'] = np.nan
#print(df['Forecast_svm']) just NaN

df['Forecast_svm'][-len(forecast_set_svm):] = forecast_set_svm

### using side by side columns
col1, col2 = st.columns(2)

with col1:
    st.subheader('Closing Price & Prediction')
    # df['Adj Close'].plot()
    # df['Forecast_svm'].plot()
    plt.plot(df['Adj Close'], 'b')
    plt.plot(df['Forecast_svm'], 'g')
    #plt.plot(forecast_set_svm, 'g')
    plt.legend(['Closing price', 'Prediction'], loc ="lower right") # show the guide on the corner
    plt.xlabel('Date')
    plt.ylabel('Price')
    #plt.show()
    st.pyplot()
    
with col2:
    st.subheader('Prediction')
    plt.plot(df['Forecast_svm'], 'g')
    plt.xlabel('Date')
    plt.ylabel('Price')
    #plt.show()
    st.pyplot()
    
#########################################
    
st.subheader('Support Vector Regression VS linear regression')
fig = plt.figure(figsize=(12, 6))
#plt.plot(cach_for_end, 'r')
plt.plot(df['Forecast'], 'r')
plt.plot(df['Forecast_svm'], 'g')
plt.legend(['linear regression', 'Support Vector Regression'], loc ="lower right")
st.pyplot()
#st.pyplot(fig)
    
#########################################    

footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Developed by <a style='display: block; text-align: center;' href="https://github.com/nosadeghob" target="_blank">Mohammad sadegh Eftekhar</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)