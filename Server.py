

 #### Import function, data and hardcoded input #### 

import flask
import time
import re
from flask import flash,session,render_template, redirect, request,Response,send_file
import time
import matplotlib.finance as finance
import datetime
from pandas import read_csv
from functools import wraps
from pandas import *
import json
from urllib2 import urlopen  # python 2 syntax

# Import users' define function
from Function_Library import *

#define backtest globally so that we can download it ==> expect modification to create a specific DataFrame for the download
global output


# Correspond to the backtest period.
global backtest_period

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
#20 days windows vol Variable
global basket_days_20_vol 
basket_days_20_vol=np.zeros(1)

#20 rolling vol Variable


global dilution
dilution=np.zeros(1)
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
    
    return flask.render_template('main.html')

@Server.route('/test/',methods=['GET','POST'])
def testing():
    
    return flask.render_template('test.html')

@Server.route('/main/',methods=['GET','POST'])
def index():
    time_start = time.clock()
    Parent_Index=flask.request.args.get('index')
    if (flask.request.args.get('liquidity') is None):
        return flask.render_template('Index_Generator.html')
    liquidity_thresh=int(flask.request.args.get('liquidity'))
    if (Parent_Index is None):
        return flask.render_template('Index_Generator.html')
    else:
        Parent_Index=str(flask.request.args.get('index'))
        Prices_df=get_data_price(Parent_Index,liquidity_thresh)
        MktCap_df=get_data_mkt(Parent_Index,liquidity_thresh)
        Benchmark_df=read_csv('%s - Benchmark.csv' %Parent_Index,sep=';',decimal=",")
        input_benchmark=Benchmark_df['Index']

    backtest_period=int(flask.request.args.get('backtest_len'))  
    Benchmark_Graph_df=Benchmark_df
    Benchmark_Graph_df=Benchmark_Graph_df.tail(backtest_period*20+1)
    Benchmark_Graph_df=Benchmark_Graph_df.reset_index()
    del Benchmark_Graph_df['index']
    Benchmark_Graph_df['values']=Benchmark_Graph_df['Index']/Benchmark_Graph_df['Index'][0]
    del Benchmark_Graph_df['Index']
    benchmark=Benchmark_Graph_df


    value_strat=flask.request.args.get('strategy')
    if (value_strat is None):
        return flask.render_template('Index_Generator.html') 
    if value_strat=='momentum_long_only':
        strat_name="LO Momentum"
        strategy='momentum'
        strat_list=("none","none", "none")
        position='long'
        Method=flask.request.args.get('method')
        if (flask.request.args.get('method') is None):
            return flask.render_template('Index_Generator.html')
        Constraint_type=flask.request.args.get('method_type')

    if value_strat=='alpha':
        strat_name="LO Alpha"
        strategy='alpha'
        strat_list=("none","none", "none")
        position='long'
        Method=flask.request.args.get('method')
        if (flask.request.args.get('method') is None):
            return flask.render_template('Index_Generator.html')
        Constraint_type=flask.request.args.get('method_type')

    if value_strat=='ls_alpha':
        strat_name="L/S Alpha"
        strategy='alpha'
        strat_list=("none","none", "none")
        position='ls'
        Method=flask.request.args.get('method')
        if (flask.request.args.get('method') is None):
            return flask.render_template('Index_Generator.html')
        Constraint_type=flask.request.args.get('method_type')

    if value_strat=='sharpe':
        strat_name="LO Sharpe"
        strategy='sharpe'
        strat_list=("none","none", "none")
        position='long'
        Method=flask.request.args.get('method')
        if (flask.request.args.get('method') is None):
            return flask.render_template('Index_Generator.html')
        Constraint_type=flask.request.args.get('method_type')

    if value_strat=='ls_sharpe':
        strat_name="L/S Sharpe"
        strategy='sharpe'
        strat_list=("none","none", "none")
        position='ls'
        Method=flask.request.args.get('method')
        if (flask.request.args.get('method') is None):
            return flask.render_template('Index_Generator.html')
        Constraint_type=flask.request.args.get('method_type')
          
    if value_strat=='multi_fact':
        strat_name="LO Multi Factorial"
        strategy='multi_fact'
        position='long'
        strat_list_unparse=str(flask.request.args.getlist('strategy_combination'))
        strat_list_parsed=re.findall("'([^']*)'", strat_list_unparse)
        strat_list=tuple(strat_list_parsed)
        Method=flask.request.args.get('method')
        if (flask.request.args.get('method') is None):
            return flask.render_template('Index_Generator.html')
        Constraint_type=flask.request.args.get('method_type')

    if value_strat=='lo_reverse_beta':
        strat_name="LO Reverse Beta"
        strategy='lo_reverse_beta'
        strat_list=("none","none", "none")
        position='long'
        Method=flask.request.args.get('method')
        if (flask.request.args.get('method') is None):
            return flask.render_template('Index_Generator.html')
        Constraint_type=flask.request.args.get('method_type')

    if value_strat=='lo_beta':
        strat_name="LO Beta"
        strategy='beta'
        strat_list=("none","none", "none")
        position='long'
        Method=flask.request.args.get('method')
        if (flask.request.args.get('method') is None):
            return flask.render_template('Index_Generator.html')
        Constraint_type=flask.request.args.get('method_type')

    if value_strat=='ls_beta':
        strat_name="L/S Beta"
        strategy='beta'
        strat_list=("none","none", "none")
        position='ls'
        Method=flask.request.args.get('method')
        if (flask.request.args.get('method') is None):
            return flask.render_template('Index_Generator.html')
        Constraint_type=flask.request.args.get('method_type')

    if value_strat=='reverse_beta':
        strat_name="L/S Reverse Beta"
        strategy='reverse_beta'
        strat_list=("none","none", "none")
        position='ls'
        Method="Ranking"
        Constraint_type='None'

    if value_strat=='reverse_beta_neutral':
        strat_name="L/S Reverse Beta Neutral"
        strategy='reverse_beta_neutral'
        strat_list=("none","none", "none")
        position='ls'
        Method="Ranking"
        Constraint_type='None'

    if value_strat=='momentum_long_short':
        strat_name="L/S Momentum"
        strat_list=("none","none", "none")
        position='ls'
        strategy='momentum'
        Method="Ranking"
        Constraint_type='None'
        

    if (flask.request.args.get('NbMonth1') is None):
        return flask.render_template('Index_Generator.html')
       
    Nb_Month_1 = int(flask.request.args.get('NbMonth1'))
    Nb_Month_2 = int(flask.request.args.get('NbMonth2'))
    
 ##identify the constraints method
    global name
    if Method=="Ranking":
        Max_Weight_Allowed=0
        Max_Vol=0
        Max_Beta=0
        Min_Beta=0
        name=str(strat_name)+" "+str(Method)+" "+"Index"  
          
    else:
        if Constraint_type=="Vol_C":
            Max_Beta=0
            Min_Beta=0
            Max_Vol=float(flask.request.args.get('vol_cap'))
            Max_Weight_Allowed=float(flask.request.args.get('max_weight'))
            name= str(strat_name)+" "+str(Method)+" "+"("+"Vol:"+" "+str(Max_Vol)+"%"+")"+" "+"Index"
            
        else:
            Max_Weight_Allowed=float(flask.request.args.get('max_weight'))
            Max_Vol=0
            Max_Beta=float(flask.request.args.get('max_beta'))
            Min_Beta=float(flask.request.args.get('min_beta'))
            name= str(strat_name)+" "+str(Method)+" "+"("+"Beta:"+" "+str(Min_Beta)+"-"+str(Max_Beta)+")"+" "+"Index"
            


##Vol Cap Part
    vol_cap_imposed=flask.request.args.get('vol_capped')
    if (flask.request.args.get('vol_frame') is None):
        return flask.render_template('Index_Generator_Pro.html')

    if vol_cap_imposed=="vol_cap_yes":
        vol_cap = (float(flask.request.args.get('vol_cap_daily')))/100
        vol_frame= int(flask.request.args.get('vol_frame'))
    else:
        vol_cap=10
        vol_frame=20

    leverage_in=flask.request.args.get('leverage')
    if (flask.request.args.get('leverage') is None):
        return flask.render_template('Index_Generator.html')

    if leverage_in=="yes_leverage":
        leverage=float(flask.request.args.get('leverage_value'))
    else:
        leverage=1

                                    ##compute the composition of the selected index as of now##

    current_composition=optimal_weights(strategy,Prices_df,input_benchmark,Method,Constraint_type,Max_Weight_Allowed,MktCap_df,Nb_Month_1,Nb_Month_2,ThreeM_USD_libor,position,Max_Vol,Min_Beta,Max_Beta,strat_list)
    #convert the weights into unit percentages
    current_composition=current_composition[current_composition<>0]*100
    #Convert current composition data to json --> needed as the dataToJson doesn't output the right format for the graphs
    current_composition_df=current_composition.to_frame()
    current_composition_df.columns=["Weights (%)"]
    current_composition_json=current_composition_df.to_json(date_format='iso',orient='split')
    current_composition_temp=json.loads(current_composition_json)
    #Convert current composition data to JSON for the pie chart
    current_composition_pie_chart_json=json.dumps([{"label": date, "value": val} for date, val in zip(current_composition_temp['index'], current_composition_temp['data'])])
    #Convert current composition data to JSON for the bar chart
    current_composition_bar_chart_json=json.dumps([{"x": date, "y": val} for date, val in zip(current_composition_temp['index'], current_composition_temp['data'])])
    
                                            ##Compute the backtest of the strategy##
    
    #get arguments
    freq=int(flask.request.args.get('rebalance_len'))
    back_tested = back_test(strategy,Prices_df,Method,Constraint_type,Max_Weight_Allowed,MktCap_df,backtest_period,Nb_Month_1,Nb_Month_2,ThreeM_USD_libor,vol_cap,freq,vol_frame,position,Max_Vol,input_benchmark,Min_Beta, Max_Beta,strat_list,leverage)
    print input_benchmark
    description_df=OutputStats(back_tested,current_composition,Benchmark_df)
    #Convert the _test data to json
    back_tested_json = dataToJson(back_tested)



    #Create a description of the backtest data
    
    back_tested_df=back_tested.to_frame()
    back_tested_graph=back_tested_df.reset_index()
    back_tested_graph.columns=["date",name]
   
    roll_vol_df=get_roll_vol(back_tested_df)

    if vol_cap<>10:
        dilution_df=get_dil(back_tested_df,vol_cap,vol_frame)
    else:
        dilution_df=DataFrame(np.zeros(len(back_tested_graph["date"])),index=back_tested_graph["date"]) 
        
    leverage_in=flask.request.args.get('leverage')
    
    
    #bidouille
    back_tested_graph["New Date"]=back_tested_graph["date"].map(lambda x: datetime.strptime(x, '%d/%m/%y'))
    global output

    output=back_tested_graph
    
    

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

    roll_vol_df=roll_vol_df.reset_index()
    roll_vol_df.columns=['date','value']
    roll_vol_df["New Date"]=roll_vol_df["date"].map(lambda x: datetime.strptime(x, '%d/%m/%y'))
    roll_vol_df_js=json.dumps([[date, returns] for date, returns in zip(roll_vol_df['New Date'], roll_vol_df['value'])])
    #misceleanous data for the web page
    dilution_df
    dilution_df=dilution_df.reset_index()
    dilution_df.columns=['date','value']
    dilution_df["New Date"]=dilution_df["date"].map(lambda x: datetime.strptime(x, '%d/%m/%y'))
    dilution_df_js=json.dumps([[date, returns] for date, returns in zip(dilution_df['New Date'], dilution_df['value'])])
    #backtest_date=t
    pricing_day= time.strftime("%d")
    pricing_month= time.strftime("%B")
    pricing_year= time.strftime("%Y")
    pricing_hour= time.strftime("%X")

        
        
    #this is complete tweaking, apparently yahoo is missing some data, so I tweaked the dataset so that we have the same data (ideal: use a merge so that we only select the same data in both dataset)
    
    
    benchmark=benchmark.set_index('Date')
    
    


    benchmark_return=Returns_df(benchmark)
    benchmark_return=benchmark_return.reset_index()
    

    benchmark_return["New Date"]=benchmark_return['Date'].map(lambda x: datetime.strptime(x, '%d/%m/%y'))
    
    #benchmark["New Date"]=benchmark["date"].map(lambda x: datetime.strptime(x, '%d/%m/%y'))
    

    benchmark_return = json.dumps([[date,val] for date, val in zip(benchmark_return['New Date'], benchmark_return['values'])])

    benchmark=benchmark.reset_index()
    benchmark["New Date"]=benchmark["Date"].map(lambda x: datetime.strptime(x, '%d/%m/%y'))

    
    benchmark = json.dumps([[date,val] for date, val in zip(benchmark['New Date'], benchmark['values'])])
    
    number_component=len(current_composition_df)

    number_observation_bt=int(description_df.loc['Number of Observations'])
    average_level_bt=float(description_df.loc['Avg. Return (ann,%)'])
    volatility_bt=float(description_df.loc['Volatility (ann,%)'])
    maximum_bt=float(description_df.loc['Maximum Drawdown (%)'])
    minimum_bt=float(description_df.loc['Sortino Ratio'])
    pt25_bt=float(description_df.loc['Sharpe Ratio'])
    pt50_bt=float(description_df.loc['Nb of Positive Returns'])
    pt75_bt=float(description_df.loc['Avg. Positive Returns (%)'])

    
    data_hist=histo_func(back_tested_df)
    index_bins_js=json.dumps(data_hist[1])
    index_freq_js=json.dumps(data_hist[0])


    benchmark_histo_graph=Benchmark_Graph_df.tail(backtest_period*20+1)
    benchmark_histo_graph=benchmark_histo_graph.set_index('Date')
    
    benchmark_hist=histo_func(benchmark_histo_graph)
    ben_bins_js=json.dumps(benchmark_hist[1])
    ben_freq_js=json.dumps(benchmark_hist[0])

    creation_time=str(pricing_hour) + " " + str(pricing_day)+" "+str(pricing_month)+" "+str(pricing_year)
    temp_data=[creation_time,name]

    Database.loc[len(Database)]=temp_data
    #compute and plot norm distribution
    data_norm_freq_js=json.dumps(normfunction(data_hist[1],backtest_period/6))
    data_norm_bins_js=json.dumps(benchmark_hist[0])
    #Database_out=Database.set_index('Date')
    plotrf_df=plotrf(back_tested_graph,ThreeM_USD_libor)

    rf_graph_line=json.dumps([[date,val] for date, val in zip(plotrf_df['New Date'], plotrf_df["value_rf"])])
    number_neg_comp=len(current_composition_df[current_composition_df<0])
    time_elapsed = (time.clock() - time_start)
    print time_elapsed
    return flask.render_template('Index_Generator.html',Database=Database.to_html(classes='weights'),ben_freq_js=ben_freq_js,ben_bins_js=ben_bins_js,\
        index_freq_js=index_freq_js,index_bins_js=index_bins_js,benchmark=benchmark,data_graph_pie=data_graph_pie,data_graph_line=data_graph_line,pt75_bt=pt75_bt,\
        pt50_bt=pt50_bt,pt25_bt=pt25_bt,maximum_bt=maximum_bt,minimum_bt=minimum_bt,volatility_bt=volatility_bt,average_level_bt=average_level_bt,\
        number_observation_bt=number_observation_bt,backtest_date=backtest_period,number_component=number_component,name=name,\
        pricing_day=pricing_day,pricing_month=pricing_month,pricing_year=pricing_year,pricing_hour=pricing_hour, pie_data=current_composition_pie_chart_json,\
         current_data=current_composition_df.to_html(classes='weights'),current_composition_df=current_composition_df,bar_data=current_composition_bar_chart_json,\
         back_tested_data=back_tested_json,describe_data=description.to_html(classes='weights'),underlying=Parent_Index,number_neg_comp=number_neg_comp)

#pro generate
@Server.route('/pro/',methods=['GET','POST'])
@login_required
def index_pro():
    time_start = time.clock()
    Parent_Index=flask.request.args.get('index')
    if (flask.request.args.get('liquidity') is None):
        return flask.render_template('Index_Generator_Pro.html')
    liquidity_thresh=int(flask.request.args.get('liquidity'))
    if (Parent_Index is None):
        return flask.render_template('Index_Generator_Pro.html')
    else:
        Parent_Index=str(flask.request.args.get('index'))
        Prices_df=get_data_price(Parent_Index,liquidity_thresh)
        MktCap_df=get_data_mkt(Parent_Index,liquidity_thresh)
        Benchmark_df=read_csv('%s - Benchmark.csv' %Parent_Index,sep=';',decimal=",")
        input_benchmark=Benchmark_df['Index']

    backtest_period=int(flask.request.args.get('backtest_len'))  
    Benchmark_Graph_df=Benchmark_df
    Benchmark_Graph_df=Benchmark_Graph_df.tail(backtest_period*20+1)
    Benchmark_Graph_df=Benchmark_Graph_df.reset_index()
    del Benchmark_Graph_df['index']
    Benchmark_Graph_df['values']=Benchmark_Graph_df['Index']/Benchmark_Graph_df['Index'][0]
    del Benchmark_Graph_df['Index']
    benchmark=Benchmark_Graph_df


    value_strat=flask.request.args.get('strategy')
    if (value_strat is None):
        return flask.render_template('Index_Generator_Pro.html') 
    if value_strat=='momentum_long_only':
        strat_name="LO Momentum"
        strategy='momentum'
        strat_list=("none","none", "none")
        position='long'
        Method=flask.request.args.get('method')
        if (flask.request.args.get('method') is None):
            return flask.render_template('Index_Generator_Pro.html')
        Constraint_type=flask.request.args.get('method_type')

    if value_strat=='alpha':
        strat_name="LO Alpha"
        strategy='alpha'
        strat_list=("none","none", "none")
        position='long'
        Method=flask.request.args.get('method')
        if (flask.request.args.get('method') is None):
            return flask.render_template('Index_Generator_Pro.html')
        Constraint_type=flask.request.args.get('method_type')

    if value_strat=='ls_alpha':
        strat_name="L/S Alpha"
        strategy='alpha'
        strat_list=("none","none", "none")
        position='ls'
        Method=flask.request.args.get('method')
        if (flask.request.args.get('method') is None):
            return flask.render_template('Index_Generator_Pro.html')
        Constraint_type=flask.request.args.get('method_type')

    if value_strat=='sharpe':
        strat_name="LO Sharpe"
        strategy='sharpe'
        strat_list=("none","none", "none")
        position='long'
        Method=flask.request.args.get('method')
        if (flask.request.args.get('method') is None):
            return flask.render_template('Index_Generator_Pro.html')
        Constraint_type=flask.request.args.get('method_type')

    if value_strat=='ls_sharpe':
        strat_name="L/S Sharpe"
        strategy='sharpe'
        strat_list=("none","none", "none")
        position='ls'
        Method=flask.request.args.get('method')
        if (flask.request.args.get('method') is None):
            return flask.render_template('Index_Generator_Pro.html')
        Constraint_type=flask.request.args.get('method_type')
          
    if value_strat=='multi_fact':
        strat_name="LO Multi Factorial"
        strategy='multi_fact'
        position='long'
        strat_list_unparse=str(flask.request.args.getlist('strategy_combination'))
        strat_list_parsed=re.findall("'([^']*)'", strat_list_unparse)
        strat_list=tuple(strat_list_parsed)
        Method=flask.request.args.get('method')
        if (flask.request.args.get('method') is None):
            return flask.render_template('Index_Generator_Pro.html')
        Constraint_type=flask.request.args.get('method_type')

    if value_strat=='lo_reverse_beta':
        strat_name="LO Reverse Beta"
        strategy='lo_reverse_beta'
        strat_list=("none","none", "none")
        position='long'
        Method=flask.request.args.get('method')
        if (flask.request.args.get('method') is None):
            return flask.render_template('Index_Generator_Pro.html')
        Constraint_type=flask.request.args.get('method_type')

    if value_strat=='lo_beta':
        strat_name="LO Beta"
        strategy='beta'
        strat_list=("none","none", "none")
        position='long'
        Method=flask.request.args.get('method')
        if (flask.request.args.get('method') is None):
            return flask.render_template('Index_Generator_Pro.html')
        Constraint_type=flask.request.args.get('method_type')

    if value_strat=='ls_beta':
        strat_name="L/S Beta"
        strategy='beta'
        strat_list=("none","none", "none")
        position='ls'
        Method=flask.request.args.get('method')
        if (flask.request.args.get('method') is None):
            return flask.render_template('Index_Generator_Pro.html')
        Constraint_type=flask.request.args.get('method_type')

    if value_strat=='reverse_beta':
        strat_name="L/S Reverse Beta"
        strategy='reverse_beta'
        strat_list=("none","none", "none")
        position='ls'
        Method="Ranking"
        Constraint_type='None'

    if value_strat=='reverse_beta_neutral':
        strat_name="L/S Reverse Beta Neutral"
        strategy='reverse_beta_neutral'
        strat_list=("none","none", "none")
        position='ls'
        Method="Ranking"
        Constraint_type='None'

    if value_strat=='momentum_long_short':
        strat_name="L/S Momentum"
        strat_list=("none","none", "none")
        position='ls'
        strategy='momentum'
        Method="Ranking"
        Constraint_type='None'
        

    if (flask.request.args.get('NbMonth1') is None):
        return flask.render_template('Index_Generator_Pro.html')
       
    Nb_Month_1 = int(flask.request.args.get('NbMonth1'))
    Nb_Month_2 = int(flask.request.args.get('NbMonth2'))
    
 ##identify the constraints method
    global name
    if Method=="Ranking":
        Max_Weight_Allowed=0
        Max_Vol=0
        Max_Beta=0
        Min_Beta=0
        name=str(strat_name)+" "+str(Method)+" "+"Index"  
          
    else:
        if Constraint_type=="Vol_C":
            Max_Beta=0
            Min_Beta=0
            Max_Vol=float(flask.request.args.get('vol_cap'))
            Max_Weight_Allowed=float(flask.request.args.get('max_weight'))
            name= str(strat_name)+" "+str(Method)+" "+"("+"Vol:"+" "+str(Max_Vol)+"%"+")"+" "+"Index"
            
        else:
            Max_Weight_Allowed=float(flask.request.args.get('max_weight'))
            Max_Vol=0
            Max_Beta=float(flask.request.args.get('max_beta'))
            Min_Beta=float(flask.request.args.get('min_beta'))
            name= str(strat_name)+" "+str(Method)+" "+"("+"Beta:"+" "+str(Min_Beta)+"-"+str(Max_Beta)+")"+" "+"Index"
            


##Vol Cap Part
    vol_cap_imposed=flask.request.args.get('vol_capped')
    if (flask.request.args.get('vol_frame') is None):
        return flask.render_template('Index_Generator_Pro.html')
    if vol_cap_imposed=="vol_cap_yes":
        vol_cap = (float(flask.request.args.get('vol_cap_daily')))/100
        vol_frame= int(flask.request.args.get('vol_frame'))
    else:
        vol_cap=10
        vol_frame=20

    leverage_in=flask.request.args.get('leverage')
    if (flask.request.args.get('leverage') is None):
        return flask.render_template('Index_Generator_Pro.html')

    if leverage_in=="yes_leverage":
        leverage=float(flask.request.args.get('leverage_value'))
    else:
        leverage=1

                                    ##compute the composition of the selected index as of now##

    current_composition=optimal_weights(strategy,Prices_df,input_benchmark,Method,Constraint_type,Max_Weight_Allowed,MktCap_df,Nb_Month_1,Nb_Month_2,ThreeM_USD_libor,position,Max_Vol,Min_Beta,Max_Beta,strat_list)
    #convert the weights into unit percentages
    current_composition=current_composition[current_composition<>0]*100
    #Convert current composition data to json --> needed as the dataToJson doesn't output the right format for the graphs
    current_composition_df=current_composition.to_frame()
    current_composition_df.columns=["Weights (%)"]
    current_composition_json=current_composition_df.to_json(date_format='iso',orient='split')
    current_composition_temp=json.loads(current_composition_json)
    #Convert current composition data to JSON for the pie chart
    current_composition_pie_chart_json=json.dumps([{"label": date, "value": val} for date, val in zip(current_composition_temp['index'], current_composition_temp['data'])])
    #Convert current composition data to JSON for the bar chart
    current_composition_bar_chart_json=json.dumps([{"x": date, "y": val} for date, val in zip(current_composition_temp['index'], current_composition_temp['data'])])
    
                                            ##Compute the backtest of the strategy##
    
    #get arguments
    freq=int(flask.request.args.get('rebalance_len'))
    back_tested = back_test(strategy,Prices_df,Method,Constraint_type,Max_Weight_Allowed,MktCap_df,backtest_period,Nb_Month_1,Nb_Month_2,ThreeM_USD_libor,vol_cap,freq,vol_frame,position,Max_Vol,input_benchmark,Min_Beta, Max_Beta,strat_list,leverage)
    print input_benchmark
    description_df=OutputStats(back_tested,current_composition,Benchmark_df)
    #Convert the backtest data to json
    back_tested_json = dataToJson(back_tested)



    #Create a description of the backtest data
    
    back_tested_df=back_tested.to_frame()
    back_tested_graph=back_tested_df.reset_index()
    back_tested_graph.columns=["date",name]
   
    roll_vol_df=get_roll_vol(back_tested_df)

    if vol_cap<>10:
        dilution_df=get_dil(back_tested_df,vol_cap,vol_frame)
    else:
        dilution_df=DataFrame(np.zeros(len(back_tested_graph["date"])),index=back_tested_graph["date"]) 
        
    
    
    #bidouille
    back_tested_graph["New Date"]=back_tested_graph["date"].map(lambda x: datetime.strptime(x, '%d/%m/%y'))
    global output

    output=back_tested_graph
    
    

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

    roll_vol_df=roll_vol_df.reset_index()
    roll_vol_df.columns=['date','value']
    roll_vol_df["New Date"]=roll_vol_df["date"].map(lambda x: datetime.strptime(x, '%d/%m/%y'))
    roll_vol_df_js=json.dumps([[date, returns] for date, returns in zip(roll_vol_df['New Date'], roll_vol_df['value'])])
    #misceleanous data for the web page
    dilution_df
    dilution_df=dilution_df.reset_index()
    dilution_df.columns=['date','value']
    dilution_df["New Date"]=dilution_df["date"].map(lambda x: datetime.strptime(x, '%d/%m/%y'))
    dilution_df_js=json.dumps([[date, returns] for date, returns in zip(dilution_df['New Date'], dilution_df['value'])])
    #backtest_date=t
    pricing_day= time.strftime("%d")
    pricing_month= time.strftime("%B")
    pricing_year= time.strftime("%Y")
    pricing_hour= time.strftime("%X")

        
    #this is complete tweaking, apparently yahoo is missing some data, so I tweaked the dataset so that we have the same data (ideal: use a merge so that we only select the same data in both dataset)
    
    
    benchmark=benchmark.set_index('Date')
    
    


    benchmark_return=Returns_df(benchmark)
    benchmark_return=benchmark_return.reset_index()
    

    benchmark_return["New Date"]=benchmark_return['Date'].map(lambda x: datetime.strptime(x, '%d/%m/%y'))
    
    #benchmark["New Date"]=benchmark["date"].map(lambda x: datetime.strptime(x, '%d/%m/%y'))
    

    benchmark_return = json.dumps([[date,val] for date, val in zip(benchmark_return['New Date'], benchmark_return['values'])])
    
    #bidouille
    #back_tested_graph["New Date"]=back_tested_graph["date"].map(lambda x: datetime.strptime(x, '%d/%m/%y'))
    ###
    #data_graph_line=json.dumps([[date,val] for date, val in zip(back_tested_graph['New Date'], back_tested_graph[name])])

    #bidouille
    benchmark=benchmark.reset_index()
    benchmark["New Date"]=benchmark["Date"].map(lambda x: datetime.strptime(x, '%d/%m/%y'))

    
    benchmark = json.dumps([[date,val] for date, val in zip(benchmark['New Date'], benchmark['values'])])
    
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


    benchmark_histo_graph=Benchmark_Graph_df.tail(backtest_period*20+1)
    benchmark_histo_graph=benchmark_histo_graph.set_index('Date')
    
    benchmark_hist=histo_func(benchmark_histo_graph)
    ben_bins_js=json.dumps(benchmark_hist[1])
    ben_freq_js=json.dumps(benchmark_hist[0])

    creation_time=str(pricing_hour) + " " + str(pricing_day)+" "+str(pricing_month)+" "+str(pricing_year)
    temp_data=[creation_time,name]

    Database.loc[len(Database)]=temp_data
    #compute and plot norm distribution

    data_norm_freq_js=json.dumps(normfunction(data_hist[1],back_tested_df_return))
    data_norm_bins_js=json.dumps(benchmark_hist[0])
    #Database_out=Database.set_index('Date')
    plotrf_df=plotrf(back_tested_graph,ThreeM_USD_libor)

    rf_graph_line=json.dumps([[date,val] for date, val in zip(plotrf_df['New Date'], plotrf_df["value_rf"])])
  
    time_elapsed = (time.clock() - time_start)
    print time_elapsed
    
    return flask.render_template('Index_Generator_Pro.html',rf_graph_line=rf_graph_line,\
        data_norm_freq_js=data_norm_freq_js,benchmark_return=benchmark_return,back_tested_df_return=back_tested_df_return_json,\
        Database=Database.to_html(classes='weights'),ben_freq_js=ben_freq_js,ben_bins_js=ben_bins_js,index_freq_js=index_freq_js,\
        index_bins_js=index_bins_js,benchmark=benchmark,data_graph_pie=data_graph_pie,data_graph_line=data_graph_line,\
        pt75_bt=pt75_bt,pt50_bt=pt50_bt,pt25_bt=pt25_bt,maximum_bt=maximum_bt,minimum_bt=minimum_bt,volatility_bt=volatility_bt,\
        average_level_bt=average_level_bt,number_observation_bt=number_observation_bt,backtest_date=backtest_period,\
        number_component=number_component,underlying=Parent_Index,name=name,pricing_day=pricing_day,pricing_month=pricing_month,\
        pricing_year=pricing_year,pricing_hour=pricing_hour, pie_data=current_composition_pie_chart_json, \
        current_data=current_composition_df.to_html(classes='weights',float_format=lambda x: '%.3f' % x),\
        current_composition_df=current_composition_df,bar_data=current_composition_bar_chart_json,\
        back_tested_data=back_tested_json,description_df=description_df.to_html(classes='weights',float_format=lambda x: '%.3f' % x),\
        describe_data=description.to_html(classes='weights',float_format=lambda x: '%.3f' % x),roll_vol_df_js=roll_vol_df_js,\
        dilution_df_js=dilution_df_js)

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
    Database=DataFrame()
    flask.session.pop('logged_in',None)
    flash('You have signed out')
    return redirect('/')
@Server.route("/download_doc")
def downloadDOC(): 
    del output['date']
    del output['dayrf'] 
    csv = output.to_csv()
    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-disposition":
                 "attachment; filename= %s .csv" %name})
#download button function
@Server.route("/download")
@login_required
def downloadCSV():  
    del output['date']
    del output['dayrf']  
    csv = output.to_csv()
    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-disposition":
                 "attachment; filename= %s .csv" %name})
if __name__ == '__main__':
    Server.debug=True
    Server.run()   