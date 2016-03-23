                                #### Import function, data and hardcoded input #### 

import flask

from flask import flash,session,render_template, redirect, request,Response
import time
import matplotlib.finance as finance
import datetime
from pandas import read_csv
from functools import wraps

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
#NKY225=get_benchmark("^N225",2013,5,28,2016,3,01)

from pandas import *

# Import Python Library
import json
from urllib2 import urlopen  # python 2 syntax

# Import users' define function
from Function_Library import *

#define backtest globally so that we can download it ==> expect modification to create a specific DataFrame for the download
global output


# Import data -  New Import from CSV

Prices_df=read_csv('NKY225 - Prices.csv',sep=';',decimal=",")
Prices_df=Prices_df.set_index('Date')
#Prices_df=Prices_df.astype(float)
Prices_df.index.name=None
MktCap_df=read_csv('NKY225 - MktCap.csv',sep=';',decimal=",")
MktCap_df=MktCap_df.set_index('Date')
#MktCap_df=MktCap_df.astype(float)
MktCap_df.index.name=None

# Correspond to the backtest period.
global backtest_period

# Benchmark data: Nikkei 225
NKY225_df=read_csv('NKY225 Benchmark.csv',sep=';',decimal=",")
NKY225_df.columns=['Date','Index']
#NKY225_df.index.name=None
NKY225_a_df=NKY225_df.set_index('Date')


# Libor : risk free rate - try to find a way to get it online?
ThreeM_USD_libor = 0.00619

# Number of month to compute the momentum
global Nb_Month_1
global Nb_Month_2
#following lines commented to test user input
#Nb_Month_1 = 3
#Nb_Month_2 = 6



#initiate database
global Database
Database=DataFrame(columns=('Date','Name'))

# Constraint function : in our case, volatility is the constraint.


                                #### Server set-up #### 



# Create the application.
Server = flask.Flask(__name__)

#function to check if user logged in
def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            flash('Please login first.')
            return redirect('/login')
    return wrap
    
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

    backtest_period=int(flask.request.args.get('backtest_len'))
    
    #the following lines prevent the code from bugging if the Ranking method is selected
    if Method=="Ranking":
        Max_Weight_Allowed=0
        Max_Vol=0
        name="Momentum"+" "+str(Method)+" "+str(Nb_Month_1)+"-"+str(Nb_Month_2)+" "+"Months"+" "+"Index"
     
        
    else:
        Max_Vol=float(flask.request.args.get('vol_cap'))
        Max_Weight_Allowed=float(flask.request.args.get('max_weight'))
        name="Momentum"+" "+str(Method)+" "+"("+"Vol:"+" "+str(Max_Vol)+"%"+")"+" "+str(Nb_Month_1)+"-"+str(Nb_Month_2)+" "+"Months"+" "+"Index"
       
        

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
    back_tested = back_test(Prices_df,Max_Vol,Max_Weight_Allowed,MktCap_df,Method,backtest_period,Nb_Month_1,Nb_Month_2,ThreeM_USD_libor,vol_cap,freq,vol_frame)
    #Convert the backtest data to json
    back_tested_json = dataToJson(back_tested)



    #Create a description of the backtest data
    
    back_tested_df=back_tested.to_frame()
    back_tested_graph=back_tested_df.reset_index()
    back_tested_graph.columns=["date",name]
    #bidouille
    back_tested_graph["New Date"]=back_tested_graph["date"].map(lambda x: datetime.strptime(x, '%d/%m/%y'))
    ###
    data_graph_line=json.dumps([[date,val] for date, val in zip(back_tested_graph['New Date'], back_tested_graph[name])])
   
    current_composition_graph_df=current_composition_df.reset_index()
    current_composition_graph_df.columns=["tick","weights"]
    data_graph_pie=json.dumps([[tick, weight] for tick, weight in zip(current_composition_graph_df['tick'], current_composition_graph_df['weights'])])
    back_tested_df_return=Returns_df(back_tested_df)
   
    description=back_tested_df_return.describe()
    description.columns=["Description"]

    #misceleanous data for the web page

    pricing_day= time.strftime("%d")
    pricing_month= time.strftime("%B")
    pricing_year= time.strftime("%Y")
    pricing_hour= time.strftime("%X")

  
    underlying="Nikkei 225"
    #get the benchmark
    if underlying=="Nikkei 225":
        NKY225_df=NKY225_a_df.tail(backtest_period*20+1)
        NKY225_df['values']=NKY225_df['Index']/NKY225_df['Index'][0]
        del NKY225_df['Index']
        benchmark=NKY225_df
        #this is complete tweaking, apparently yahoo is missing some data, so I tweaked the dataset so that we have the same data (ideal: use a merge so that we only select the same data in both dataset)
        #benchmark=benchmark.head(backtest_period*20)
        benchmark=benchmark.reset_index()
        #bidouille
    	benchmark_graph["New Date"]=benchmark["date"].map(lambda x: datetime.strptime(x, '%d/%m/%y'))
    	
        benchmark = json.dumps([[date,val] for date, val in zip(benchmark['Date'], benchmark['values'])])
    #next part is just for fun
    #else:
     #   benchmark=SP500
      # benchmark=benchmark.reset_index()
       # benchmark = json.dumps([[date,val] for date, val in zip(benchmark['Date'], benchmark['values'])])

    number_component=len(current_composition_df)
    number_observation_bt=int(description.loc['count'])
    average_level_bt=float(description.loc['mean'])*100
    volatility_bt=float(description.loc['std'])*(252**0.5)*100
    maximum_bt=float(description.loc['max'])
    minimum_bt=float(description.loc['min'])*100
    pt25_bt=float(description.loc['25%'])
    pt50_bt=float(description.loc['50%'])
    pt75_bt=float(description.loc['75%'])

    data_hist=histo_func(back_tested_df)
    index_bins_js=json.dumps(data_hist[1])
    index_freq_js=json.dumps(data_hist[0])

    benchmark_hist=histo_func(NKY225.head(t*20))
    ben_bins_js=json.dumps(benchmark_hist[1])
    ben_freq_js=json.dumps(benchmark_hist[0])

    creation_time=str(pricing_hour) + " " + str(pricing_day)+" "+str(pricing_month)+" "+str(pricing_year)
    temp_data=[creation_time,name]

    Database.loc[len(Database)]=temp_data
    #Database_out=Database.set_index('Date')
    return flask.render_template('Index_Generator.html',Database=Database.to_html(classes='weights'),ben_freq_js=ben_freq_js,ben_bins_js=ben_bins_js,index_freq_js=index_freq_js,index_bins_js=index_bins_js,benchmark=benchmark,data_graph_pie=data_graph_pie,data_graph_line=data_graph_line,pt75_bt=pt75_bt,pt50_bt=pt50_bt,pt25_bt=pt25_bt,maximum_bt=maximum_bt,minimum_bt=minimum_bt,volatility_bt=volatility_bt,average_level_bt=average_level_bt,number_observation_bt=number_observation_bt,backtest_date=backtest_period,number_component=number_component,underlying=underlying,name=name,pricing_day=pricing_day,pricing_month=pricing_month,pricing_year=pricing_year,pricing_hour=pricing_hour, pie_data=current_composition_pie_chart_json, current_data=current_composition_df.to_html(classes='weights'),current_composition_df=current_composition_df,bar_data=current_composition_bar_chart_json,back_tested_data=back_tested_json,describe_data=description.to_html(classes='weights'))

#pro generate
@Server.route('/pro/',methods=['GET','POST'])
@login_required
def index_pro():
    if (flask.request.args.get('NbMonth1') is None):
        return flask.render_template('Index_Generator_Pro.html')
       
    Nb_Month_1 = int(flask.request.args.get('NbMonth1'))
    Nb_Month_2 = int(flask.request.args.get('NbMonth2'))
    #return the homepage if the method is blank
    if (flask.request.args.get('method') is None):
        return flask.render_template('Index_Generator_Pro.html')
    Method=flask.request.args.get('method')

    backtest_period=int(flask.request.args.get('backtest_len'))
    
     
    #the following lines prevent the code from bugging if the Ranking method is selected
    global name
    if Method=="Ranking":
        Max_Weight_Allowed=0
        Max_Vol=0
        name="Momentum"+" "+str(Method)+" "+str(Nb_Month_1)+"-"+str(Nb_Month_2)+" "+"Months"+" "+"Index"    
    else:
        Max_Vol=float(flask.request.args.get('vol_cap'))
        Max_Weight_Allowed=float(flask.request.args.get('max_weight'))
        name="Momentum"+" "+str(Method)+" "+"("+"Vol:"+" "+str(Max_Vol)+"%"+")"+" "+str(Nb_Month_1)+"-"+str(Nb_Month_2)+" "+"Months"+" "+"Index"
        

    vol_cap_imposed=flask.request.args.get('vol_capped')
    if (flask.request.args.get('vol_frame') is None):
        return flask.render_template('Index_Generator_Pro.html')

    if vol_cap_imposed=="vol_cap_yes":
        vol_cap = (float(flask.request.args.get('vol_cap_daily')))/1000
        vol_frame= int(flask.request.args.get('vol_frame'))
    else:
        vol_cap=1
        vol_frame=20
    

    freq=int(flask.request.args.get('rebalance_len'))
    #compute the composition of the selected index as of now
    current_composition=optimal_weights(Prices_df,Method,Max_Vol,Max_Weight_Allowed,MktCap_df,Nb_Month_1,Nb_Month_2,ThreeM_USD_libor)
    #convert the weights into unit percentages
    current_composition=current_composition[current_composition>0]*100
    

    #Convert current composition data to json --> needed as the dataToJson doesn't output the right format for the graphs
    current_composition_df=current_composition.to_frame()
    current_composition_df.columns=["Weights (%)"]
    current_composition_json=current_composition_df.to_json(date_format='iso',orient='split')
    current_composition_temp=json.loads(current_composition_json)

    #Convert current composition data to JSON for the pie chart
    current_composition_pie_chart_json=json.dumps([{"label": date, "value": val} for date, val in zip(current_composition_temp['index'], current_composition_temp['data'])])
    #Convert current composition data to JSON for the bar chart
    current_composition_bar_chart_json=json.dumps([{"x": date, "y": val} for date, val in zip(current_composition_temp['index'], current_composition_temp['data'])])
    
    #Compute the backtest of the strategy
    back_tested = back_test(Prices_df,Max_Vol,Max_Weight_Allowed,MktCap_df,Method,backtest_period,Nb_Month_1,Nb_Month_2,ThreeM_USD_libor,vol_cap,freq,vol_frame)
    description_df=OutputStats(back_tested,current_composition)
    #Convert the backtest data to json
    back_tested_json = dataToJson(back_tested)



    #Create a description of the backtest data
    
    back_tested_df=back_tested.to_frame()
    back_tested_graph=back_tested_df.reset_index()
    back_tested_graph.columns=["date",name]
    
    global output
    
    
    #bidouille
    back_tested_graph["New Date"]=back_tested_graph["date"].map(lambda x: datetime.strptime(x, '%d/%m/%y'))
    ###
    data_graph_line=json.dumps([[date,val] for date, val in zip(back_tested_graph['New Date'], back_tested_graph[name])])
    
    current_composition_graph_df=current_composition_df.reset_index()
    current_composition_graph_df.columns=["tick","weights"]
    data_graph_pie=json.dumps([[tick, weight] for tick, weight in zip(current_composition_graph_df['tick'], current_composition_graph_df['weights'])])
    back_tested_df_return=Returns_df(back_tested_df)
    

    description=back_tested_df_return.describe()
    description.columns=["Description"]

    back_tested_df_return_js=back_tested_df_return.reset_index()
   
    back_tested_df_return_js.columns=['date','returns']
    back_tested_df_return_js["New Date"]=back_tested_df_return_js["date"].map(lambda x: datetime.strptime(x, '%d/%m/%y'))
    back_tested_df_return_json=json.dumps([[date, returns] for date, returns in zip(back_tested_df_return_js['New Date'], back_tested_df_return_js['returns'])])
    #misceleanous data for the web page

    #backtest_date=t
    pricing_day= time.strftime("%d")
    pricing_month= time.strftime("%B")
    pricing_year= time.strftime("%Y")
    pricing_hour= time.strftime("%X")

    underlying="Nikkei 225"
    
	#get the benchmark
    if underlying=="Nikkei 225":
        
        NKY225_df=NKY225_a_df.tail(backtest_period*20+1)
        NKY225_df['values']=NKY225_df['Index']/NKY225_df['Index'][0]
        del NKY225_df['Index']
        
        benchmark=NKY225_df
        
        #this is complete tweaking, apparently yahoo is missing some data, so I tweaked the dataset so that we have the same data (ideal: use a merge so that we only select the same data in both dataset)
        
        benchmark_return=Returns_df(NKY225_a_df.tail(backtest_period*20+1))
        benchmark_return=benchmark_return.reset_index()
        benchmark_return["New Date"]=benchmark_return['Date'].map(lambda x: datetime.strptime(x, '%d/%m/%y'))
        
    	#benchmark["New Date"]=benchmark["date"].map(lambda x: datetime.strptime(x, '%d/%m/%y'))
                
        benchmark_return = json.dumps([[date,val] for date, val in zip(benchmark_return['New Date'], benchmark_return['Index'])])
        
		#bidouille
	    #back_tested_graph["New Date"]=back_tested_graph["date"].map(lambda x: datetime.strptime(x, '%d/%m/%y'))
	    ###
	    #data_graph_line=json.dumps([[date,val] for date, val in zip(back_tested_graph['New Date'], back_tested_graph[name])])

        #bidouille
    	benchmark=benchmark.reset_index()

    	benchmark["New Date"]=benchmark["Date"].map(lambda x: datetime.strptime(x, '%d/%m/%y'))

        
        benchmark = json.dumps([[date,val] for date, val in zip(benchmark['New Date'], benchmark['values'])])
    #next part is just for fun
    #else:
     #   benchmark=SP500
      #  benchmark=benchmark.head(del NKY225_2_df['Index']*20-4)
       # benchmark=benchmark.reset_index()
        #benchmark = json.dumps([[date,val] for date, val in zip(benchmark['Date'], benchmark['values'])])

    number_component=len(current_composition_df)
    number_observation_bt=int(description.loc['count'])
    average_level_bt=float(description.loc['mean'])*100
    volatility_bt=float(description.loc['std'])*(252**0.5)*100
    maximum_bt=float(description.loc['max'])
    minimum_bt=float(description.loc['min'])*100
    pt25_bt=float(description.loc['25%'])
    pt50_bt=float(description.loc['50%'])
    pt75_bt=float(description.loc['75%'])

    
    data_hist=histo_func(back_tested_df)
    index_bins_js=json.dumps(data_hist[1])
    index_freq_js=json.dumps(data_hist[0])

    benchmark_hist=histo_func(NKY225_a_df.head(backtest_period*20))
    ben_bins_js=json.dumps(benchmark_hist[1])
    ben_freq_js=json.dumps(benchmark_hist[0])

    creation_time=str(pricing_hour) + " " + str(pricing_day)+" "+str(pricing_month)+" "+str(pricing_year)
    temp_data=[creation_time,name]

    Database.loc[len(Database)]=temp_data

    data_norm_freq_js=json.dumps(normfunction(data_hist[1]))
    data_norm_bins_js=json.dumps(benchmark_hist[0])
    #Database_out=Database.set_index('Date')
    return flask.render_template('Index_Generator_Pro.html',data_norm_freq_js=data_norm_freq_js,benchmark_return=benchmark_return,back_tested_df_return=back_tested_df_return_json,Database=Database.to_html(classes='weights'),ben_freq_js=ben_freq_js,ben_bins_js=ben_bins_js,index_freq_js=index_freq_js,index_bins_js=index_bins_js,benchmark=benchmark,data_graph_pie=data_graph_pie,data_graph_line=data_graph_line,pt75_bt=pt75_bt,pt50_bt=pt50_bt,pt25_bt=pt25_bt,maximum_bt=maximum_bt,minimum_bt=minimum_bt,volatility_bt=volatility_bt,average_level_bt=average_level_bt,number_observation_bt=number_observation_bt,backtest_date=backtest_period,number_component=number_component,underlying=underlying,name=name,pricing_day=pricing_day,pricing_month=pricing_month,pricing_year=pricing_year,pricing_hour=pricing_hour, pie_data=current_composition_pie_chart_json, current_data=current_composition_df.to_html(classes='weights',float_format=lambda x: '%.3f' % x),current_composition_df=current_composition_df,bar_data=current_composition_bar_chart_json,back_tested_data=back_tested_json,description_df=description_df.to_html(classes='weights',float_format=lambda x: '%.3f' % x),describe_data=description.to_html(classes='weights',float_format=lambda x: '%.3f' % x))

#login page
Server.secret_key="secret_key"
# route for handling the login page logic
@Server.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'admin' or request.form['password'] != 'admin':
            error = 'Invalid Credentials. Please try again.'
        else:
            flask.session['logged_in'] = True
            
            return redirect('/pro/')
    return render_template('login.html', error=error)

#logout function
@Server.route('/logout')
@login_required
def logout():
    flask.session.pop('logged_in',None)
    flash('You have signed out')
    return redirect('/')
#download button function
@Server.route("/download")
@login_required
def downloadCSV():   
    csv = output.to_csv()
    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-disposition":
                 "attachment; filename= %s .csv" %name})
if __name__ == '__main__':
    Server.debug=True
    Server.run()