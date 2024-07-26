pipimport dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np

pd.options.mode.chained_assignment = None
app = dash.Dash()
server = app.server

scaler=MinMaxScaler(feature_range=(0,1))


#segment 1
df_aapl = pd.read_csv("AAPL.csv")
df_aapl['Date']=df_aapl['Date']

df_aapl["Date"]=pd.to_datetime(df_aapl.Date,format="%Y-%m-%d")
df_aapl.index=df_aapl['Date']

data=df_aapl.sort_index(ascending=True,axis=0)
new_data=pd.DataFrame(index=range(0,len(df_aapl)),columns=['Date','Close'])

for i in range(0,len(data)):
    new_data["Date"][i]=data['Date'][i]
    new_data["Close"][i]=data["Close"][i]

new_data.index=new_data.Date
new_data.drop("Date",axis=1,inplace=True)

dataset=new_data.values

train=dataset[0:1008,:]
valid=dataset[1008:,:]

scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)

x_train,y_train=[],[]

for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
    
x_train,y_train=np.array(x_train),np.array(y_train)

x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

model=load_model("saved_lstm_model_live_aapl.h5")

inputs=new_data[len(new_data)-len(valid)-60:].values
inputs=inputs.reshape(-1,1)
inputs=scaler.transform(inputs)

X_test=[]
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test=np.array(X_test)

X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
closing_price=model.predict(X_test)
closing_price=scaler.inverse_transform(closing_price)

train=new_data[:1008]
valid=new_data[1008:]
valid['Predictions']=closing_price

#segment 1 end

scaler_2=MinMaxScaler(feature_range=(0,1))

#segment 2
df_tsla = pd.read_csv("TSLA.csv")
df_tsla['Date']=df_tsla['Date']

df_tsla["Date"]=pd.to_datetime(df_tsla.Date,format="%Y-%m-%d")
df_tsla.index=df_tsla['Date']

data_2=df_tsla.sort_index(ascending=True,axis=0)
new_data_2=pd.DataFrame(index=range(0,len(df_tsla)),columns=['Date','Close'])

for i in range(0,len(data)):
    new_data_2["Date"][i]=data_2['Date'][i]
    new_data_2["Close"][i]=data_2["Close"][i]

new_data_2.index=new_data_2.Date
new_data_2.drop("Date",axis=1,inplace=True)

dataset_2=new_data_2.values

train_2=dataset_2[0:1008,:]
valid_2=dataset_2[1008:,:]

scaler_2=MinMaxScaler(feature_range=(0,1))
scaled_data_2=scaler_2.fit_transform(dataset_2)

x_train_2,y_train_2=[],[]

for i in range(60,len(train_2)):
    x_train_2.append(scaled_data_2[i-60:i,0])
    y_train_2.append(scaled_data_2[i,0])
    
x_train_2,y_train_2=np.array(x_train_2),np.array(y_train_2)

x_train_2=np.reshape(x_train_2,(x_train_2.shape[0],x_train_2.shape[1],1))

model_2=load_model("saved_lstm_model_live_tsla.h5")

inputs_2=new_data_2[len(new_data_2)-len(valid_2)-60:].values
inputs_2=inputs_2.reshape(-1,1)
inputs_2=scaler.transform(inputs_2)

X_test_2=[]
for i in range(60,inputs_2.shape[0]):
    X_test_2.append(inputs_2[i-60:i,0])
X_test_2=np.array(X_test_2)

X_test_2=np.reshape(X_test_2,(X_test_2.shape[0],X_test_2.shape[1],1))
closing_price_2=model_2.predict(X_test_2)
closing_price_2=scaler.inverse_transform(closing_price_2)

train_2=new_data_2[:1008]
valid_2=new_data_2[1008:]
valid_2['Predictions']=closing_price_2

#segment 2 end

scaler_3=MinMaxScaler(feature_range=(0,1))

#segment 3
df_msft = pd.read_csv("MSFT.csv")
df_msft['Date']=df_msft['Date']

df_msft["Date"]=pd.to_datetime(df_msft.Date,format="%Y-%m-%d")
df_msft.index=df_msft['Date']

data_3=df_aapl.sort_index(ascending=True,axis=0)
new_data_3=pd.DataFrame(index=range(0,len(df_aapl)),columns=['Date','Close'])

for i in range(0,len(data_3)):
    new_data_3["Date"][i]=data_3['Date'][i]
    new_data_3["Close"][i]=data_3["Close"][i]

new_data_3.index=new_data_3.Date
new_data_3.drop("Date",axis=1,inplace=True)

dataset_3=new_data_3.values

train_3=dataset_3[0:1008,:]
valid_3=dataset_3[1008:,:]

scaler_3=MinMaxScaler(feature_range=(0,1))
scaled_data_3=scaler_3.fit_transform(dataset_3)

x_train_3,y_train_3=[],[]

for i in range(60,len(train_3)):
    x_train_3.append(scaled_data_3[i-60:i,0])
    y_train_3.append(scaled_data_3[i,0])
    
x_train_3,y_train_3=np.array(x_train_3),np.array(y_train_3)

x_train_3=np.reshape(x_train_3,(x_train_3.shape[0],x_train_3.shape[1],1))

model_3=load_model("saved_lstm_model_live_tsla.h5")

inputs_3=new_data_3[len(new_data_3)-len(valid_3)-60:].values
inputs_3=inputs_3.reshape(-1,1)
inputs_3=scaler.transform(inputs_3)

X_test_3=[]
for i in range(60,inputs_3.shape[0]):
    X_test_3.append(inputs_3[i-60:i,0])
X_test_3=np.array(X_test_3)

X_test_3=np.reshape(X_test_3,(X_test_3.shape[0],X_test_3.shape[1],1))
closing_price_3=model_3.predict(X_test_3)
closing_price_3=scaler_3.inverse_transform(closing_price_3)

train_3=new_data_3[:1008]
valid_3=new_data_3[1008:]
valid_3['Predictions']=closing_price_3

#segment 3 end

scaler_4=MinMaxScaler(feature_range=(0,1))

#segment 4
df_fb = pd.read_csv("FB.csv")
df_fb['Date']=df_fb['Date']

df_fb["Date"]=pd.to_datetime(df_fb.Date,format="%Y-%m-%d")
df_fb.index=df_fb['Date']

data_4=df_fb.sort_index(ascending=True,axis=0)
new_data_4=pd.DataFrame(index=range(0,len(df_fb)),columns=['Date','Close'])

for i in range(0,len(data_4)):
    new_data_4["Date"][i]=data_4['Date'][i]
    new_data_4["Close"][i]=data_4["Close"][i]

new_data_4.index=new_data_4.Date
new_data_4.drop("Date",axis=1,inplace=True)

dataset_4=new_data_4.values

train_4=dataset_4[0:1008,:]
valid_4=dataset_4[1008:,:]

scaler_4=MinMaxScaler(feature_range=(0,1))
scaled_data_4=scaler_4.fit_transform(dataset_4)

x_train_4,y_train_4=[],[]

for i in range(60,len(train_4)):
    x_train_4.append(scaled_data_4[i-60:i,0])
    y_train_4.append(scaled_data_4[i,0])
    
x_train_4,y_train_4=np.array(x_train_4),np.array(y_train_4)

x_train_4=np.reshape(x_train_4,(x_train_4.shape[0],x_train_4.shape[1],1))

model_4=load_model("saved_lstm_model_live_fb.h5")

inputs_4=new_data_4[len(new_data_4)-len(valid_4)-60:].values
inputs_4=inputs_4.reshape(-1,1)
inputs_4=scaler_4.transform(inputs_4)

X_test_4=[]
for i in range(60,inputs_4.shape[0]):
    X_test_4.append(inputs_4[i-60:i,0])
X_test_4=np.array(X_test_4)

X_test_4=np.reshape(X_test_4,(X_test_4.shape[0],X_test_4.shape[1],1))
closing_price_4=model_4.predict(X_test_4)
closing_price_4=scaler_4.inverse_transform(closing_price_4)

train_4=new_data_4[:1008]
valid_4=new_data_4[1008:]
valid_4['Predictions']=closing_price_4

#segment 4 end

df= pd.read_csv("stock_data_live.csv")

app.layout = html.Div([
   
    html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),
   
    dcc.Tabs(id="tabs", children=[
       
        dcc.Tab(label='Apple Stock Data',children=[
			html.Div([
				html.H2("Actual closing price",style={"textAlign": "center"}),
				dcc.Graph(
					id="Actual Data Apple",
					figure={
						"data":[
							go.Scatter(
								x=train.index,
								y=valid["Close"],
								mode='markers'
							)

						],
						"layout":go.Layout(
							title='scatter plot',
							xaxis={'title':'Date'},
							yaxis={'title':'Closing Rate'}
						)
					}

				),
				html.H2("LSTM Predicted closing price",style={"textAlign": "center"}),
				dcc.Graph(
					id="Predicted Data Apple",
					figure={
						"data":[
							go.Scatter(
								x=valid.index,
								y=valid["Predictions"],
								mode='markers'
							)

						],
						"layout":go.Layout(
							title='scatter plot',
							xaxis={'title':'Date'},
							yaxis={'title':'Closing Rate'}
						)
					}

				)				
			])        		


        ]),
        dcc.Tab(label='Tesla Stock Data',children=[
			html.Div([
				html.H2("Actual closing price",style={"textAlign": "center"}),
				dcc.Graph(
					id="Actual Data Tesla",
					figure={
						"data":[
							go.Scatter(
								x=train_2.index,
								y=valid_2["Close"],
								mode='markers'
							)

						],
						"layout":go.Layout(
							title='scatter plot',
							xaxis={'title':'Date'},
							yaxis={'title':'Closing Rate'}
						)
					}

				),
				html.H2("LSTM Predicted closing price",style={"textAlign": "center"}),
				dcc.Graph(
					id="Predicted Data Tesla",
					figure={
						"data":[
							go.Scatter(
								x=valid_2.index,
								y=valid_2["Predictions"],
								mode='markers'
							)

						],
						"layout":go.Layout(
							title='scatter plot',
							xaxis={'title':'Date'},
							yaxis={'title':'Closing Rate'}
						)
					}

				)				
			])        		


        ]),
        dcc.Tab(label='Microsoft Stock Data',children=[
			html.Div([
				html.H2("Actual closing price",style={"textAlign": "center"}),
				dcc.Graph(
					id="Actual Data Microsoft",
					figure={
						"data":[
							go.Scatter(
								x=train_3.index,
								y=valid_3["Close"],
								mode='markers'
							)

						],
						"layout":go.Layout(
							title='scatter plot',
							xaxis={'title':'Date'},
							yaxis={'title':'Closing Rate'}
						)
					}

				),
				html.H2("LSTM Predicted closing price",style={"textAlign": "center"}),
				dcc.Graph(
					id="Predicted Data Microsoft",
					figure={
						"data":[
							go.Scatter(
								x=valid_3.index,
								y=valid_3["Predictions"],
								mode='markers'
							)

						],
						"layout":go.Layout(
							title='scatter plot',
							xaxis={'title':'Date'},
							yaxis={'title':'Closing Rate'}
						)
					}

				)				
			])        		


        ]),
        dcc.Tab(label='Facebook Stock Data',children=[
			html.Div([
				html.H2("Actual closing price",style={"textAlign": "center"}),
				dcc.Graph(
					id="Actual Data Facebook",
					figure={
						"data":[
							go.Scatter(
								x=train_4.index,
								y=valid_4["Close"],
								mode='markers'
							)

						],
						"layout":go.Layout(
							title='scatter plot',
							xaxis={'title':'Date'},
							yaxis={'title':'Closing Rate'}
						)
					}

				),
				html.H2("LSTM Predicted closing price",style={"textAlign": "center"}),
				dcc.Graph(
					id="Predicted Data Facebook",
					figure={
						"data":[
							go.Scatter(
								x=valid_4.index,
								y=valid_4["Predictions"],
								mode='markers'
							)

						],
						"layout":go.Layout(
							title='scatter plot',
							xaxis={'title':'Date'},
							yaxis={'title':'Closing Rate'}
						)
					}

				)				
			])        		


        ]),
        dcc.Tab(label='Comparison of Stock Data', children=[
            html.Div([
                html.H1("Stocks High vs Lows", 
                        style={'textAlign': 'center'}),
              
                dcc.Dropdown(id='my-dropdown',
                             options=[{'label': 'Tesla', 'value': 'TSLA'},
                                      {'label': 'Apple','value': 'AAPL'}, 
                                      {'label': 'Facebook', 'value': 'FB'}, 
                                      {'label': 'Microsoft','value': 'MSFT'}], 
                             multi=True,value=['FB'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='highlow'),
                html.H1("Stocks Market Volume", style={'textAlign': 'center'}),
         
                dcc.Dropdown(id='my-dropdown2',
                             options=[{'label': 'Tesla', 'value': 'TSLA'},
                                      {'label': 'Apple','value': 'AAPL'}, 
                                      {'label': 'Facebook', 'value': 'FB'},
                                      {'label': 'Microsoft','value': 'MSFT'}], 
                             multi=True,value=['FB'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='volume')
            ], className="container"),
        ])


    ])
])







@app.callback(Output('highlow', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
    dropdown = {"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","MSFT": "Microsoft",}
    trace1 = []
    trace2 = []
    for stock in selected_dropdown:
        trace1.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["High"],
                     mode='lines', opacity=0.7, 
                     name=f'High {dropdown[stock]}',textposition='bottom center'))
        trace2.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["Low"],
                     mode='lines', opacity=0.6,
                     name=f'Low {dropdown[stock]}',textposition='bottom center'))
    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"High and Low Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'Date'},
             yaxis={"title":"Price (USD)"})}
    return figure


@app.callback(Output('volume', 'figure'),
              [Input('my-dropdown2', 'value')])
def update_graph(selected_dropdown_value):
    dropdown = {"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","MSFT": "Microsoft",}
    trace1 = []
    for stock in selected_dropdown_value:
        trace1.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["Volume"],
                     mode='lines', opacity=0.7,
                     name=f'Volume {dropdown[stock]}', textposition='bottom center'))
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data, 
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"Market Volume for {', '.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M',
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'Date'},
             yaxis={"title":"Transactions Volume"})}
    return figure



if __name__=='__main__':
	app.run_server(debug=True)