# Import python libraries
from __future__ import division
import os

import datetime
from pandas import *
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


                                #### General Function used ##### 
def normfunction(bins,n):
    sigma=0.01
    mu=0
    # Plot between -10 and 10 with .001 steps.
    
    result=[]

    for i in bins:

        value=n*(1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (i - mu)**2 / (2 * sigma**2)))
        
        result.append(value)
    
    return result
#compute a simple function to output libor (must be a simpler way)
def plotrf(back_tested_df,libor):
    lenght=len(back_tested_df)
    back_tested_df["dayrf"]=DataFrame(range(0,lenght))
    back_tested_df["value_rf"]=back_tested_df["dayrf"].map(lambda x: np.exp(libor*(x/250)))
    return back_tested_df


# # Compute returns from a dataframe of prices.

def Returns_df(Prices_df):
    df_return=Prices_df/Prices_df.shift(1)-1
    df_return=df_return.ix[1:]
    return df_return

def histo_func(df):
    df_return=df/df.shift(1)-1
    df_return=df_return.ix[1:]
    hist_data=np.histogram(df_return,bins=[-0.05,-0.04,-0.03,-0.02,-0.01,0,0.01,0.02,0.03,0.04,0.05])
    return hist_data

def dataToJson(a):
    b=a.to_frame()
    b.columns=["values"]
    b=b.reset_index()
    H=b.drop('index',1)
    d=json.loads(H.to_json(date_format='iso',orient='split'))
    return json.dumps([{"x": date, "y": val} for date, val in zip(d['index'], d['data'])]) 



                                #### Z-Scores Function ##### 


### MOMENTUM STRATEGY

# Nb_Month_1 = First period = 3 months for us
# Nb_Month_2 = Second period = 6 months for us

# Price_Momentum returns the momentum of a security for a certain period -> will be used in the MSCI_Momentum Function
def Price_Momentum(Prices_df,Nb_Month,ThreeM_USD_libor):
    
    Price_Momentum = (Prices_df.iloc[-20]/(Prices_df.iloc[-Nb_Month*20])) - 1 - ThreeM_USD_libor

    return Price_Momentum

# MSCI_Momentum returns Dataframe containing the Ranked Z-score per security in the universe 
def MSCI_Momentum(Prices_df,Nb_Month_1,Nb_Month_2,ThreeM_USD_libor):

      # Compute returns
    df_return=Returns_df(Prices_df)
    
        #compute annualized vol from returns
    vol=((np.var(df_return)**0.5)*(len(df_return)**0.5))
    
    # Compute momentum using above momentum function
    three_m_m = Price_Momentum(Prices_df,Nb_Month_1,ThreeM_USD_libor)
    six_m_m = Price_Momentum(Prices_df,Nb_Month_2,ThreeM_USD_libor)

    three_m_m = three_m_m.fillna(0)
    six_m_m=six_m_m.fillna(0)
    
    # Risk Contrainted to vol and convert to frame
    rc_three_m_m=(three_m_m/vol)
    rc_six_m_m=(six_m_m/vol)
    
    # Winderized our RC distributions  : 
    rc_three_m_m[ rc_three_m_m >= 3] = 3
    rc_three_m_m[rc_three_m_m <= -3] = -3

    rc_six_m_m[ rc_six_m_m >= 3] = 3
    rc_six_m_m[rc_six_m_m <= -3] = -3

    
    #calcul of momentum scores
    z_score=rc_six_m_m*0.5+rc_three_m_m*0.5

    z_df=z_score.to_frame().fillna(0)
    
    z_df.columns = ['z_score']
    momentum_z_score_df = z_df.iloc[:,0].map(lambda x: x+1 if x>0 else (1-x)**(-1)).to_frame()
    momentum_z_score_df.columns = ['momentum_z_score']

    # Ranked Momentum  
    ranked_momentum_z_score_df=momentum_z_score_df.sort(['momentum_z_score'],ascending=[False])
    
    return ranked_momentum_z_score_df
    
def MSCI_Momentum_No_Ranking(Prices_df,Nb_Month_1,Nb_Month_2,ThreeM_USD_libor):
          # Compute returns
    df_return=Returns_df(Prices_df)
    
        #compute annualized vol from returns
    vol=((np.var(df_return)**0.5)*(len(df_return)**0.5))
    
    # Compute momentum using above momentum function
    three_m_m = Price_Momentum(Prices_df,Nb_Month_1,ThreeM_USD_libor)
    six_m_m = Price_Momentum(Prices_df,Nb_Month_2,ThreeM_USD_libor)

    three_m_m = three_m_m.fillna(0)
    six_m_m=six_m_m.fillna(0)
    
    # Risk Contrainted to vol and convert to frame
    rc_three_m_m=(three_m_m/vol)
    rc_six_m_m=(six_m_m/vol)
    
    #Winderized our RC distributions

    rc_three_m_m[ rc_three_m_m >= 3] = 3
    rc_three_m_m[rc_three_m_m <= -3] = -3

    rc_six_m_m[ rc_six_m_m >= 3] = 3
    rc_six_m_m[rc_six_m_m <= -3] = -3
    
    #calcul of momentum scores
    z_score=rc_six_m_m*0.5+rc_three_m_m*0.5

    z_df=z_score.to_frame().fillna(0)
    
    z_df.columns = ['z_score']
    momentum_z_score_df = z_df.iloc[:,0].map(lambda x: x+1 if x>0 else (1-x)**(-1)).to_frame()
    momentum_z_score_df.columns = ['momentum_z_score']
    return momentum_z_score_df
    

### CARRY STRATEGY
# To be implemented


                                #### PRE-REQUIRED FUNCTION FOR OTPIMIZATION PROCESSES ##### 


### Function Needed for the non-Constrained method.

# Round Off rules as per the MSCI Methodology used in the Number_of_Securities_Index function.

def MyRound(x, base):
    return int(base * round(float(x)/base))

def RoundOff(NumSec):
    if NumSec < 100: 
        return MyRound(NumSec,10)
    
    elif NumSec < 300:
        return MyRound(NumSec,25)
        
    else:
        return MyRound(NumSec,50)    

# Number_of_Securities_Index returns the index composition as per the MSCI number of securities algorithm
# Returns a dataframe with columns Z-Scores, Mkt_Cap and Weights of the final index, with the tickers as Index.
# Will be used for the Non-Constrained method.

def Number_of_Securities_Index(Ranked_Zscore_df, MktCap_df):

    # MktCap => corresponds to the one trunk when using in backtest function!
    MktCap_df=MktCap_df.tail(1)
    MktCap_df=MktCap_df.transpose()
    MktCap_df.columns=['MktCap']

    #Data_Universe = DataFrame with 2 columns : Z-Score (Ranked) / MktCap with Tickers as index
    Data_Universe=Ranked_Zscore_df.join(MktCap_df, how='inner')
    Data_Universe=Data_Universe.sort(['momentum_z_score'],ascending=[False])

    #Compute the Number of Securities in the Universe
    Nb_Sec_Universe=len(Data_Universe["momentum_z_score"])
    
    #Define Final_Index
    Final_Index = DataFrame(columns=['momentum_z_score','MktCap'])
    
    #Initialize
    Row=0
    
    # If Nb_Sec_Universe <= 25, Nb Sec Index =25 
    if Nb_Sec_Universe <= 25:

        # Number_of_Securities_Index = Nb_Sec_Universe
        Final_Index=Data_Universe
        
    # Otherwise -->  Other conditions to be met   
    else:

        # Condition on the Market Cap of the index 
        while Final_Index['MktCap'].sum() < 0.3 * Data_Universe['MktCap'].sum():
            Final_Index=Final_Index.append(Data_Universe.iloc[Row])
            Row=Row+1

        #condition on the number of sec when having 30% of universe market cap
        if len(Final_Index['MktCap']) <=25:
            Number_of_Securities_Index=RoundOff(25)
            Final_Index=Data_Universe.head(Number_of_Securities_Index)
            
        elif len(Final_Index['MktCap']) <=0.10*Nb_Sec_Universe:
            Number_of_Securities_Index=RoundOff(0.10*Nb_Sec_Universe)
            Final_Index=Data_Universe.head(Number_of_Securities_Index)

        else:
            
            Temp_Nb_Sec=RoundOff(len(Final_Index['MktCap']))

            if Temp_Nb_Sec <= 0.40 * Nb_Sec_Universe:
            
                Final_Index=Data_Universe.head(Temp_Nb_Sec)
            
            else:
                
                #Reduce the number of sec in the index to achieve 40% of universe nb sec
                while Temp_Nb_Sec > 0.40*Nb_Sec_Universe:
                    Temp_Nb_Sec=Temp_Nb_Sec-1

                Final_Index= Data_Universe.head(Temp_Nb_Sec)
                
                while Final_Index['MktCap'].sum()<0.2*Data_Universe['MktCap'].sum():
                    Temp_Nb_Sec=Temp_Nb_Sec+1

                Number_of_Securities_Index=RoundOff(Temp_Nb_Sec)
                Final_Index=Data_Universe.head(Number_of_Securities_Index)
    
                
    return len(Final_Index)


### Function Needed for the Constrained method

# Specify Jacobians of Target Function and Constraint Function
#(Optional, but highly recommended as it makes the optimisation more likely to work)
# It would be nice to automatise this passage (that is, to make the program select
# automatically the corresponding jacobian, if available, but I don't know how to 
# do it)


## Vol

# x : first derivatves vector
def index_vol(Prices_df,x):
    
    df_return=Returns_df(Prices_df)
   
    #Compute CovMatrix
    Cov_Matrix=np.cov(df_return, rowvar=0)
    Cov_Matrix=np.nan_to_num(Cov_Matrix)
    
    return np.dot(np.dot(x,Cov_Matrix),np.transpose(x))

#
# x : first derivatves vector
def index_vol_jacobian(Prices_df,x):
    
    df_return=Returns_df(Prices_df)

    #Compute CovMatrix
    Cov_Matrix=np.cov(df_return, rowvar=0)
    Cov_Matrix=np.nan_to_num(Cov_Matrix)
    
    return np.dot(x,Cov_Matrix*2)
    
## Beta

#Return the Beta of a portfolio with weights x
def index_beta(Prices_df,Benchmark_df,x):
    df_return=np.nan_to_num(np.array(Returns_df(Prices_df)))
    Benchmark_return=np.array(Returns_df(Benchmark_df))
    beta=np.zeros(len(df_return[0]))
    design_matrix=np.transpose(np.vstack((np.ones(len(df_return[:,0])),Benchmark_return)))
    
    for i in range(len(df_return[0])):
        beta[i]=np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(design_matrix),design_matrix)),np.transpose(design_matrix)),df_return[:,i])[1]
    
    return np.dot(x,beta)

#Jacobian of Beta Constraint function, it actually returns the vector of the Betas (maybe can be useful)
def index_beta_jacobian(Prices_df,Benchmark_df):
    df_return=np.nan_to_num(np.array(Returns_df(Prices_df)))
    Benchmark_return=np.array(Returns_df(Benchmark_df))
    beta=np.zeros(len(df_return[0]))
    design_matrix=np.transpose(np.vstack((np.ones(len(df_return[:,0])),Benchmark_return)))
    
    for i in range(len(df_return[0])):
        beta[i]=np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(design_matrix),design_matrix)),np.transpose(design_matrix)),df_return[:,i])[1]
    
    return beta
    

    





                                #### OTPIMIZATION FUNCTION ##### 

# Depending on the selected method, the optimization function will apply different optimization processes and returns a
# dataframe containig 

#Note: 4 arguments added: Constraint_Type(either "Volatility" or "Beta"), Benchmark_df (dataframe of benchmark levels),
# min_Beta, max_Beta.
#Max_Vol is now optional

def optimal_weights(Prices_df,Benchmark_df,Method,Constraint_Type,Max_Weight_Allowed,MktCap_df,Nb_Month_1,Nb_Month_2,ThreeM_USD_libor,position,Max_Vol=100,min_Beta=0, max_Beta=1):
    
    #df_return=Returns_df(Prices_df)
    #Ranking the inputed dataset
    Ranked_Zscore_df=MSCI_Momentum(Prices_df,Nb_Month_1,Nb_Month_2,ThreeM_USD_libor)
    # If Method is selected as Ranking.
    nbr_sec=Number_of_Securities_Index(Ranked_Zscore_df,MktCap_df)
    #Composition

    if Method=="Ranking":   
    #if position if long then nothing changes 
        if position=='long':
            # Compute the optimal index as per the MSCI Methodology
            Optimal_Index=Ranked_Zscore_df
            # Rank the computed Z-scores
            Optimal_Index["Ranking"] = (-Optimal_Index['momentum_z_score']).argsort()
            #generate weights
            Optimal_Index['Weights'] = Optimal_Index['Ranking'].map(lambda x: 1 if x <nbr_sec  else 0)
            Optimal_Index['Weights']=Optimal_Index['Weights']/np.sum(Optimal_Index['Weights']) 
            
            #Return Composition
            Composition=Series(Optimal_Index["Weights"],index=Optimal_Index["Weights"].index)
            #if position if long/short then 50 best/50 worst
        else:
            Optimal_Index=Ranked_Zscore_df
            Optimal_Index["Ranking"] = (-Optimal_Index['momentum_z_score']).argsort() 
            Optimal_Index['Position'] = 1
            Optimal_Index['Position'][ Optimal_Index["Ranking"] < 50] = 1
            Optimal_Index['Position'][(Optimal_Index["Ranking"]  > 50) & (Optimal_Index["Ranking"]  <= 175)] = 0
            Optimal_Index['Position'][Optimal_Index["Ranking"]  > 175] = -1
            Optimal_Index['Weights']=Optimal_Index['Position']/100
            Composition=Series(Optimal_Index["Weights"],index=Optimal_Index["Weights"].index)

    elif Method=="Constrained":
        
        
        Non_Ranked_Zscore_df=MSCI_Momentum_No_Ranking(Prices_df,Nb_Month_1,Nb_Month_2,ThreeM_USD_libor)

        #Initiate Weight Array
        
        if position== "long":
            Weights_0=np.ones(len(Non_Ranked_Zscore_df))
            Weights_0=Weights_0/np.sum(Weights_0)
        else:
            Weights_0=-np.ones(len(Non_Ranked_Zscore_df))
            Weights_0=Weights_0/np.sum(Weights_0)
        #Define Constraints
        
        if Constraint_Type=="Volatility":
            daily_max_var=(Max_Vol/100)**2.0/256.0
            
            if position== "long":
                Constraints=({'type': 'ineq',
                'fun' : lambda x : daily_max_var-index_vol(Prices_df,x),
                'jac' : lambda x : -index_vol_jacobian(Prices_df,x)},
                {'type' : 'eq', 
                 'fun' : lambda x : np.sum(x)-1,
                'jac' : lambda x : np.ones(len(x))
                })
            else:
                Constraints=({'type': 'ineq',
                'fun' : lambda x : daily_max_var-index_vol(Prices_df,x),
                'jac' : lambda x : -index_vol_jacobian(Prices_df,x)},
                {'type' : 'eq', 
                 'fun' : lambda x : 0.-np.sum(x),
                'jac' : lambda x : -np.ones(len(x))
                })
        
        elif Constraint_Type=="Beta":
            
            if position== "long":
                Constraints=({'type': 'ineq',
                'fun' : lambda x : max_Beta-index_beta(Prices_df,Benchmark_df,x),
                'jac' : lambda x : -index_beta_jacobian(Prices_df,Benchmark_df)},
                {'type': 'ineq',
                'fun' : lambda x : index_beta(Prices_df,Benchmark_df,x)-min_Beta,
                'jac' : lambda x : index_beta_jacobian(Prices_df,Benchmark_df)},            
                {'type' : 'eq', 
                 'fun' : lambda x : np.sum(x)-1,
                'jac' : lambda x : np.ones(len(x))
                })
            else:
                Constraints=({'type': 'ineq',
                'fun' : lambda x : max_Beta-index_beta(Prices_df,Benchmark_df,x),
                'jac' : lambda x : -index_beta_jacobian(Prices_df,Benchmark_df)},
                {'type': 'ineq',
                'fun' : lambda x : index_beta(Prices_df,Benchmark_df,x)-min_Beta,
                'jac' : lambda x : index_beta_jacobian(Prices_df,Benchmark_df)},            
                {'type' : 'eq', 
                 'fun' : lambda x : 0.-np.sum(x),
                'jac' : lambda x : -np.ones(len(x))
                })                
                
                
#            Constraints=(
#            {'type': 'eq',
#            'fun' : lambda x : index_beta(Prices_df,Benchmark_df,x)-min_Beta,
#            'jac' : lambda x : index_beta_jacobian(Prices_df,Benchmark_df)},            
#            {'type' : 'eq', 
#             'fun' : lambda x : np.sum(x)-1,
#            'jac' : lambda x : np.ones(len(x))
#            })
            
        elif Constraint_Type == "Mixed":
            daily_max_var=(Max_Vol/100)**2.0/256.0
            
            if position=="long":
                Constraints=({'type': 'ineq',
                'fun' : lambda x : max_Beta-index_beta(Prices_df,Benchmark_df,x),
                'jac' : lambda x : -index_beta_jacobian(Prices_df,Benchmark_df)},
                {'type': 'ineq',
                'fun' : lambda x : index_beta(Prices_df,Benchmark_df,x)-min_Beta,
                'jac' : lambda x : index_beta_jacobian(Prices_df,Benchmark_df)},
                {'type': 'ineq',
                'fun' : lambda x : daily_max_var-index_vol(Prices_df,x),
                'jac' : lambda x : -index_vol_jacobian(Prices_df,x)},            
                {'type' : 'eq', 
                 'fun' : lambda x : np.sum(x)-1,
                'jac' : lambda x : np.ones(len(x))
                })
            else: 
                Constraints=({'type': 'ineq',
                'fun' : lambda x : max_Beta-index_beta(Prices_df,Benchmark_df,x),
                'jac' : lambda x : -index_beta_jacobian(Prices_df,Benchmark_df)},
                {'type': 'ineq',
                'fun' : lambda x : index_beta(Prices_df,Benchmark_df,x)-min_Beta,
                'jac' : lambda x : index_beta_jacobian(Prices_df,Benchmark_df)},
                {'type': 'ineq',
                'fun' : lambda x : daily_max_var-index_vol(Prices_df,x),
                'jac' : lambda x : -index_vol_jacobian(Prices_df,x)},            
                {'type' : 'eq', 
                 'fun' : lambda x : 0.-np.sum(x),
                'jac' : lambda x : -np.ones(len(x))
                })
          
        if position== "long":
        #Define minimum value for weights (always 0) and maximum (in range (0,1] )
            bnds=[(0,(Max_Weight_Allowed/100.))]*len(Weights_0)
        else:
            bnds=[((-Max_Weight_Allowed/100.),(Max_Weight_Allowed/100.))]*len(Weights_0)
        
        #Define Target - Used in the optimization function following
        def target_fun(x):
            
            total_score=-np.dot(x,Non_Ranked_Zscore_df)
            return total_score
        def target_fun_derivative(x):
            
            return -Non_Ranked_Zscore_df
    
        #Define Target Function
        res=minimize(target_fun, Weights_0, jac = target_fun_derivative,
                   constraints=Constraints, method="SLSQP",
                   bounds=bnds,
                   options={'disp': True, "maxiter":1000})
        
        #Extract Weights
        Weights=res.x
            
        # Function to remove the components with unsignificant weights
        f= lambda x : x if x>10**-6 else 0.0
        f=np.vectorize(f)
        Weights=f(Weights)
   
        #Normalise
        Weights=Weights/np.sum(Weights)
        
       
        Composition=Series(Weights,index=Ranked_Zscore_df.index)
        Composition.name = "Weights"
        
    
    return Composition
    

                                #### BACKTEST FUNCTION ##### 


# The following procedure runs the optimization process at different period in the past.
# It returns the performance of the index over this period.

#Arguments to be checked when calling the function
  
def back_test(Prices_df,Method,Constraint_Type,Max_Weight_Allowed,MktCap_df,t,Nb_Month_1,Nb_Month_2,ThreeM_USD_libor,vol_cap, freq, vol_time,position,Max_Vol=100,Benchmark_df=0,min_Beta=0, max_Beta=1):
      
    # vol_time : new input : number of days used to compute previous volatility

    df_return=Returns_df(Prices_df)

    # try to find what the starting point is
    starting_point=len(df_return)-t*20    
    
    #set up counter for loop    
    cnt=0
    
    while (cnt)*freq<=t*20:
    #set the dataset for backtest
        Prices_df_bt=Prices_df.head(starting_point+freq*cnt)
        Benchmark_df_bt=Benchmark_df.head(starting_point+freq*cnt)
        #Compute Optimal Composition
        #Weights_bt=optimal_weights(Prices_df_bt,Method,Max_Vol,Max_Weight_Allowed,MktCap_df,Nb_Month_1,Nb_Month_2,ThreeM_USD_libor,position)
        Weights_bt=optimal_weights(Prices_df_bt,Benchmark_df_bt,Method,Constraint_Type,Max_Weight_Allowed,MktCap_df,Nb_Month_1,Nb_Month_2,ThreeM_USD_libor,position,Max_Vol,min_Beta, max_Beta)
    
        #compute returns
        next_returns=df_return.ix[(starting_point+freq*cnt):].fillna(0)
        next_20_returns=next_returns.head(freq)
        optimum_return=np.dot(next_20_returns,Weights_bt)
        
        #create return series
        if cnt == 0:
            return_series=Series(optimum_return,index=next_20_returns.index)
        
        else:
            return_series_prov=Series(optimum_return,index=next_20_returns.index)
            return_series=np.hstack([return_series,return_series_prov])
        
        #update counter
        cnt=cnt+1
    
    #end loop
    
    return_series_date=Series(return_series,index=df_return.tail(t*20).index)
    
    #creating base 1
    base_1_backtest=np.ones(len(return_series_date)+1)
    
    #undiluted represents the quote of risky securities in the index (the rest is assumed cash)
    undiluted=1.0
    
    for i in range(1,len(base_1_backtest)):
        if i>vol_time:
        
            #get "past" return at time i
            information=return_series_date.head(i)
            #get last [vol_time] returns to compute vol
            period=information.tail(vol_time)
            #compute vol
            hist_vol=np.std(period)*(250.0/vol_time)**0.5
                    
            #dilute if vol above vol_cap
            if hist_vol>vol_cap:
                undiluted=min(vol_cap/hist_vol,undiluted)
            #end if
        #compute base 1
        base_1_backtest[i]=base_1_backtest[i-1]*(1+return_series_date[i-1]*undiluted)
        #print undiluted
     
    base_1_backtest_date=Series(base_1_backtest,index=df_return.tail(t*20+1).index)  
    
    return base_1_backtest_date

                                #### OUTPUT STATISTICS FUNCTION ##### 

# General function which displays the Summary Statistics table

# Only1 function to call returning a full dataframe with all statistics: OutputStats(back_tested,current_composition)

# Function which returns the number of components : input dataframe of stocks and weigths
def NbofComponents(current_composition_df):
    return len(current_composition_df)


def AvgAnnualReturn(back_tested_df):
    Perf = back_tested_df['Returns'][len(back_tested_df)-1]/back_tested_df['Returns'][0]
    
    return Perf ** (252 / len(back_tested_df)) - 1


def AnnVolatility(back_tested_returns_df):
    Vol = np.std(back_tested_returns_df['Returns'])
    return Vol * (252)**0.5

def UpsideVol(back_tested_returns_df):
    NegReturns_df=back_tested_returns_df[back_tested_returns_df['Returns']>0]
    Vol = np.std(NegReturns_df['Returns'])
    return Vol * (252)**0.5

def DownsideVol(back_tested_returns_df):
    NegReturns_df=back_tested_returns_df[back_tested_returns_df['Returns']<0]
    Vol = np.std(NegReturns_df['Returns'])
    return Vol * (252)**0.5


# We assume r=0 riskfreerate
def SharpeRatio(back_tested_returns_df,back_tested_df):
    return AvgAnnualReturn(back_tested_df)/AnnVolatility(back_tested_returns_df)

# We assume r=0 riskfreerate
def SortinoRatio(back_tested_returns_df,back_tested_df):
    return AvgAnnualReturn(back_tested_df)/DownsideVol(back_tested_returns_df)


def MaximumDD(back_tested_df):
    mdd = 0
    peak = back_tested_df['Returns'][0]
    for x in back_tested_df['Returns']:
        if x > peak:
            peak = x
        dd = (x - peak ) / peak
        if dd < mdd:
            mdd = dd
            
    return abs(mdd)

# General function which displays the Summary Statistics table
# Input: back_tested series, current composition
# Output: Databframe

def OutputStats(back_tested,current_composition):

    # Compute returns using Returns_df function
    back_tested_returns=Returns_df(back_tested)

    # Transform series into dataframe
    back_tested_df=back_tested.to_frame()
    back_tested_df.columns=['Returns']
    back_tested_returns=Returns_df(back_tested)
    back_tested_returns_df=back_tested_returns.to_frame()
    back_tested_returns_df.columns=['Returns']
    current_composition_df=current_composition.to_frame()
    # Current composition with removes 0 weights
    current_composition_df=current_composition_df[current_composition_df["Weights"]!=0]

    # Create the table
    Stats=['Index Nb of Components','Number of Observations', 'Avg. Return (ann,%)','Volatility (ann,%)',
           'Maximum Drawdown (%)','Sharpe Ratio','Sortino Ratio','Nb of Negative Returns','Avg. Negative Returns (%)',
           'Negative Volatility (%)','Nb of Positive Returns', 'Avg. Positive Returns (%)', 'Positive Volatility (%)']
    Stats_Output_df=DataFrame({'Stat':Stats})
    Stats_Output_df=Stats_Output_df.set_index('Stat')
    Stats_Output_df['Statistics']=0.0
    Stats_Output_df.index.name=None

    # Fill the table with numbers
    Stats_Output_df['Statistics']['Index Nb of Components']=NbofComponents(current_composition_df)
    Stats_Output_df['Statistics']['Number of Observations']=len(back_tested_df)
    Stats_Output_df['Statistics']['Avg. Return (ann,%)']=AvgAnnualReturn(back_tested_df)*100
    Stats_Output_df['Statistics']['Volatility (ann,%)']=AnnVolatility(back_tested_returns_df)*100
    Stats_Output_df['Statistics']['Maximum Drawdown (%)']=MaximumDD(back_tested_df)*100
    Stats_Output_df['Statistics']['Sharpe Ratio']=SharpeRatio(back_tested_returns_df,back_tested_df)
    Stats_Output_df['Statistics']['Sortino Ratio']=SortinoRatio(back_tested_returns_df,back_tested_df)
    Stats_Output_df['Statistics']['Nb of Negative Returns']=len(back_tested_returns_df[back_tested_returns_df['Returns']<0])
    Stats_Output_df['Statistics']['Avg. Negative Returns (%)']=back_tested_returns_df[back_tested_returns_df['Returns']<0].mean()*100
    Stats_Output_df['Statistics']['Negative Volatility (%)']=DownsideVol(back_tested_returns_df)*100
    Stats_Output_df['Statistics']['Nb of Positive Returns']=len(back_tested_returns_df[back_tested_returns_df['Returns']>0])
    Stats_Output_df['Statistics']['Avg. Positive Returns (%)']=back_tested_returns_df[back_tested_returns_df['Returns']>0].mean()*100
    Stats_Output_df['Statistics']['Positive Volatility (%)']=UpsideVol(back_tested_returns_df)*100

    return Stats_Output_df

