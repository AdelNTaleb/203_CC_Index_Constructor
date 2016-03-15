                                #### Import function, data and hardcoded input #### 

import flask

from flask import render_template, redirect, url_for, request
import time
import matplotlib.finance as finance
import datetime
from pandas import read_csv


#Adel I modified your function so that we can get the correct format
def get_benchmark(tick,y1,m1,d1,y2,m2,d2):
    startdate = datetime.date(y1, m1, d1)
    enddate = datetime.date(y2, m2, d2)
    ticker= tick
    benchmark_df= read_csv(finance.fetch_historical_yahoo(ticker, startdate, enddate),index_col='Date')
    benchmark_Index=benchmark_df['Close']
    benchmark_Index=benchmark_Index.to_frame()
    benchmark_Index.columns=['values']
    benchmark_Index=benchmark_Index.reset_index()
    benchmark_Index["New Date"]=benchmark_Index["Date"].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').strftime('%d/%m/%y'))
    benchmark_Index["Date"]=benchmark_Index["New Date"].map(lambda x: datetime.datetime.strptime(x, '%d/%m/%y'))
    benchmark_Index=benchmark_Index.iloc[::-1]
    benchmark_Index=benchmark_Index.drop(["New Date"],1)
    benchmark_Index=benchmark_Index.set_index(["Date"])
    benchmark_Index=benchmark_Index.divide(benchmark_Index.ix[0])
    return benchmark_Index


# Get benchmark
#SP100=get_benchmark("^OEX",2015,8,14,2016,2,2)
#SP500=get_benchmark("^GSPC",2015,8,14,2016,2,2)
NKY225=get_benchmark("^N225",2013,5,28,2016,3,01)

from pandas import *

# Import Python Library
import json
from urllib2 import urlopen  # python 2 syntax

# Import users' define function
from Function_Library import *



# Import data
#Prices_df=read_excel('Data SP100 Values.xlsx',0)
#MktCap_df=read_excel('Data SP100 Values.xlsx',1)
# Test Nikkei 225
Prices_df=read_excel('NKY225 - Prices.xlsx')
MktCap_df=read_excel('NKY225 - MktCap.xlsx')

Prices_df_return=Returns_df(Prices_df)


# Libor : risk free rate - try to find a way to get it online?
ThreeM_USD_libor = 0.00619

# Number of month to compute the momentum
global Nb_Month_1
global Nb_Month_2
#following lines commented to test user input
#Nb_Month_1 = 3
#Nb_Month_2 = 6

# Correspond to the backtest period.
global t


# Constraint function : in our case, volatility is the constraint.


                                #### Server set-up #### 



# Create the application.
Server = flask.Flask(__name__)
    
global Max_Weight_Allowed
global Max_Vol
@Server.errorhandler(404)

def page_not_found(e):
    return flask.render_template('404.html'), 404

@Server.route('/',methods=['GET','POST'])
def intro():
    
    return flask.render_template('index.html')

@Server.route('/test/',methods=['GET','POST'])
def testing():
    
    return flask.render_template('test.html')

@Server.route('/main/',methods=['GET','POST'])
def index():

    if (flask.request.args.get('NbMonth1') is None):
        return flask.render_template('Index_Generator.html')
        
    Nb_Month_1 = int(flask.request.args.get('NbMonth1'))
    Nb_Month_2 = int(flask.request.args.get('NbMonth2'))
    #return the homepage if the method is blank
    if (flask.request.args.get('method') is None):
        return flask.render_template('Index_Generator.html')
    Method=flask.request.args.get('method')

    t=int(flask.request.args.get('backtest_len'))
    print t
    #the following lines prevent the code from bugging if the Ranking method is selected
    if Method=="Ranking":
        Max_Weight_Allowed=0
        Max_Vol=0
        name="Momentum"+" "+str(Method)+" "+str(Nb_Month_1)+"-"+str(Nb_Month_2)+" "+"Months"+" "+"Index"
        print Max_Vol
        print name
        
    else:
        Max_Vol=float(flask.request.args.get('vol_cap'))
        Max_Weight_Allowed=float(flask.request.args.get('max_weight'))
        name="Momentum"+" "+str(Method)+" "+"("+"Vol:"+" "+str(Max_Vol)+"%"+")"+" "+str(Nb_Month_1)+"-"+str(Nb_Month_2)+" "+"Months"+" "+"Index"
        print Max_Vol
        print name
        

    #compute the composition of the selected index as of now
    current_composition=optimal_weights(Prices_df,Method,Max_Vol,Max_Weight_Allowed,MktCap_df,Nb_Month_1,Nb_Month_2,ThreeM_USD_libor)
    #convert the weights into unit percentages
    current_composition=current_composition[current_composition>0]*100

    #Convert current composition data to json --> needed as the dataToJson doesn't output the right format for the graphs
    current_composition_df=current_composition.to_frame()
    current_composition_df.columns=["Weights"]
    current_composition_json=current_composition_df.to_json(date_format='iso',orient='split')
    current_composition_temp=json.loads(current_composition_json)

    #Convert current composition data to JSON for the pie chart
    current_composition_pie_chart_json=json.dumps([{"label": date, "value": val} for date, val in zip(current_composition_temp['index'], current_composition_temp['data'])])
    #Convert current composition data to JSON for the bar chart
    current_composition_bar_chart_json=json.dumps([{"x": date, "y": val} for date, val in zip(current_composition_temp['index'], current_composition_temp['data'])])

    #Compute the backtest of the strategy
    back_tested = back_test(Prices_df,Max_Vol,Max_Weight_Allowed,MktCap_df,Method,t,Nb_Month_1,Nb_Month_2,ThreeM_USD_libor)
    #Convert the backtest data to json
    back_tested_json = dataToJson(back_tested)



    #Create a description of the backtest data
    
    back_tested_df=back_tested.to_frame()
    back_tested_graph=back_tested_df.reset_index()
    back_tested_graph.columns=["date",name]
    data_graph_line=json.dumps([[date,val] for date, val in zip(back_tested_graph['date'], back_tested_graph[name])])
    current_composition_graph_df=current_composition_df.reset_index()
    current_composition_graph_df.columns=["tick","weights"]
    data_graph_pie=json.dumps([[tick, weight] for tick, weight in zip(current_composition_graph_df['tick'], current_composition_graph_df['weights'])])
    description=back_tested_df.describe()
    description.columns=["Description"]

    #misceleanous data for the web page

    backtest_date=t
    pricing_day= time.strftime("%d")
    pricing_month= time.strftime("%B")
    pricing_year= time.strftime("%Y")
    pricing_hour= time.strftime("%X")

    underlying="Nikkei 225"
    #get the benchmark
    if underlying=="Nikkei 225":
        benchmark=NKY225
        #this is complete tweaking, apparently yahoo is missing some data, so I tweaked the dataset so that we have the same data (ideal: use a merge so that we only select the same data in both dataset)
        benchmark=benchmark.head(t*20)
        benchmark=benchmark.reset_index()
        benchmark = json.dumps([[date,val] for date, val in zip(benchmark['Date'], benchmark['values'])])
    #next part is just for fun
    else:
        benchmark=SP500
        benchmark=benchmark.head(t*20-4)
        benchmark=benchmark.reset_index()
        benchmark = json.dumps([[date,val] for date, val in zip(benchmark['Date'], benchmark['values'])])

    number_component=len(current_composition_df)
    number_observation_bt=int(description.loc['count'])
    average_level_bt=float(description.loc['mean'])
    volatility_bt=float(description.loc['std'])*100
    maximum_bt=float(description.loc['max'])
    minimum_bt=float(description.loc['min'])
    pt25_bt=float(description.loc['25%'])
    pt50_bt=float(description.loc['50%'])
    pt75_bt=float(description.loc['75%'])

    data_hist=histo_func(back_tested_df)
    index_bins_js=json.dumps(data_hist[1])
    index_freq_js=json.dumps(data_hist[0])

    benchmark_hist=histo_func(NKY225.head(t*20))
    ben_bins_js=json.dumps(benchmark_hist[1])
    ben_freq_js=json.dumps(benchmark_hist[0])
    return flask.render_template('Index_Generator.html',ben_freq_js=ben_freq_js,ben_bins_js=ben_bins_js,index_freq_js=index_freq_js,index_bins_js=index_bins_js,benchmark=benchmark,data_graph_pie=data_graph_pie,data_graph_line=data_graph_line,pt75_bt=pt75_bt,pt50_bt=pt50_bt,pt25_bt=pt25_bt,maximum_bt=maximum_bt,minimum_bt=minimum_bt,volatility_bt=volatility_bt,average_level_bt=average_level_bt,number_observation_bt=number_observation_bt,backtest_date=backtest_date,number_component=number_component,underlying=underlying,name=name,pricing_day=pricing_day,pricing_month=pricing_month,pricing_year=pricing_year,pricing_hour=pricing_hour, pie_data=current_composition_pie_chart_json, current_data=current_composition_df.to_html(classes='weights'),current_composition_df=current_composition_df,bar_data=current_composition_bar_chart_json,back_tested_data=back_tested_json,describe_data=description.to_html(classes='weights'))

@Server.route('/pro/',methods=['GET','POST'])
def index_pro():

    if (flask.request.args.get('NbMonth1') is None):
        return flask.render_template('Index_Generator_Pro.html')
        
    Nb_Month_1 = int(flask.request.args.get('NbMonth1'))
    Nb_Month_2 = int(flask.request.args.get('NbMonth2'))
    #return the homepage if the method is blank
    if (flask.request.args.get('method') is None):
        return flask.render_template('Index_Generator.html')
    Method=flask.request.args.get('method')

    t=int(flask.request.args.get('backtest_len'))
    print t
    #the following lines prevent the code from bugging if the Ranking method is selected
    if Method=="Ranking":
        Max_Weight_Allowed=0
        Max_Vol=0
        name="Momentum"+" "+str(Method)+" "+str(Nb_Month_1)+"-"+str(Nb_Month_2)+" "+"Months"+" "+"Index"
        print Max_Vol
        print name
        
    else:
        Max_Vol=float(flask.request.args.get('vol_cap'))
        Max_Weight_Allowed=float(flask.request.args.get('max_weight'))
        name="Momentum"+" "+str(Method)+" "+"("+"Vol:"+" "+str(Max_Vol)+"%"+")"+" "+str(Nb_Month_1)+"-"+str(Nb_Month_2)+" "+"Months"+" "+"Index"
        print Max_Vol
        print name
        

    #compute the composition of the selected index as of now
    current_composition=optimal_weights(Prices_df,Method,Max_Vol,Max_Weight_Allowed,MktCap_df,Nb_Month_1,Nb_Month_2,ThreeM_USD_libor)
    #convert the weights into unit percentages
    current_composition=current_composition[current_composition>0]*100

    #Convert current composition data to json --> needed as the dataToJson doesn't output the right format for the graphs
    current_composition_df=current_composition.to_frame()
    current_composition_df.columns=["Weights"]
    current_composition_json=current_composition_df.to_json(date_format='iso',orient='split')
    current_composition_temp=json.loads(current_composition_json)

    #Convert current composition data to JSON for the pie chart
    current_composition_pie_chart_json=json.dumps([{"label": date, "value": val} for date, val in zip(current_composition_temp['index'], current_composition_temp['data'])])
    #Convert current composition data to JSON for the bar chart
    current_composition_bar_chart_json=json.dumps([{"x": date, "y": val} for date, val in zip(current_composition_temp['index'], current_composition_temp['data'])])

    #Compute the backtest of the strategy
    back_tested = back_test(Prices_df,Max_Vol,Max_Weight_Allowed,MktCap_df,Method,t,Nb_Month_1,Nb_Month_2,ThreeM_USD_libor)
    #Convert the backtest data to json
    back_tested_json = dataToJson(back_tested)



    #Create a description of the backtest data
    
    back_tested_df=back_tested.to_frame()
    back_tested_graph=back_tested_df.reset_index()
    back_tested_graph.columns=["date",name]
    data_graph_line=json.dumps([[date,val] for date, val in zip(back_tested_graph['date'], back_tested_graph[name])])
    current_composition_graph_df=current_composition_df.reset_index()
    current_composition_graph_df.columns=["tick","weights"]
    data_graph_pie=json.dumps([[tick, weight] for tick, weight in zip(current_composition_graph_df['tick'], current_composition_graph_df['weights'])])
    description=back_tested_df.describe()
    description.columns=["Description"]

    #misceleanous data for the web page

    backtest_date=t
    pricing_day= time.strftime("%d")
    pricing_month= time.strftime("%B")
    pricing_year= time.strftime("%Y")
    pricing_hour= time.strftime("%X")

    underlying="Nikkei 225"
    #get the benchmark
    if underlying=="Nikkei 225":
        benchmark=NKY225
        #this is complete tweaking, apparently yahoo is missing some data, so I tweaked the dataset so that we have the same data (ideal: use a merge so that we only select the same data in both dataset)
        benchmark=benchmark.head(t*20)
        benchmark=benchmark.reset_index()
        benchmark = json.dumps([[date,val] for date, val in zip(benchmark['Date'], benchmark['values'])])
    #next part is just for fun
    else:
        benchmark=SP500
        benchmark=benchmark.head(t*20-4)
        benchmark=benchmark.reset_index()
        benchmark = json.dumps([[date,val] for date, val in zip(benchmark['Date'], benchmark['values'])])

    number_component=len(current_composition_df)
    number_observation_bt=int(description.loc['count'])
    average_level_bt=float(description.loc['mean'])
    volatility_bt=float(description.loc['std'])*100
    maximum_bt=float(description.loc['max'])
    minimum_bt=float(description.loc['min'])
    pt25_bt=float(description.loc['25%'])
    pt50_bt=float(description.loc['50%'])
    pt75_bt=float(description.loc['75%'])

    data_hist=histo_func(back_tested_df)
    index_bins_js=json.dumps(data_hist[1])
    index_freq_js=json.dumps(data_hist[0])

    benchmark_hist=histo_func(NKY225.head(t*20))
    ben_bins_js=json.dumps(benchmark_hist[1])
    ben_freq_js=json.dumps(benchmark_hist[0])
    return flask.render_template('Index_Generator_Pro.html',ben_freq_js=ben_freq_js,ben_bins_js=ben_bins_js,index_freq_js=index_freq_js,index_bins_js=index_bins_js,benchmark=benchmark,data_graph_pie=data_graph_pie,data_graph_line=data_graph_line,pt75_bt=pt75_bt,pt50_bt=pt50_bt,pt25_bt=pt25_bt,maximum_bt=maximum_bt,minimum_bt=minimum_bt,volatility_bt=volatility_bt,average_level_bt=average_level_bt,number_observation_bt=number_observation_bt,backtest_date=backtest_date,number_component=number_component,underlying=underlying,name=name,pricing_day=pricing_day,pricing_month=pricing_month,pricing_year=pricing_year,pricing_hour=pricing_hour, pie_data=current_composition_pie_chart_json, current_data=current_composition_df.to_html(classes='weights'),current_composition_df=current_composition_df,bar_data=current_composition_bar_chart_json,back_tested_data=back_tested_json,describe_data=description.to_html(classes='weights'))

# route for handling the login page logic
@Server.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'admin' or request.form['password'] != 'admin':
            error = 'Invalid Credentials. Please try again.'
        else:
            return redirect('/pro/')
    return render_template('login.html', error=error)

if __name__ == '__main__':
    Server.debug=True
    Server.run()