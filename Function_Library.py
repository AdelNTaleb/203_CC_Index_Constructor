# Import python libraries
import os

import datetime
from pandas import *
import numpy as np
from scipy.optimize import minimize


                                #### General Function used ##### 


# # Compute returns from a dataframe of prices.

def Returns_df(Prices_df):
    df_return=Prices_df/Prices_df.shift(1)-1
    df_return=df_return.ix[1:]
    return df_return

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
    
    # COME BACK HERE  : Winderized our RC distributions

    #rc_six_m_m = rc_six_m_m.iloc[0,:].map(lambda x: 3.0 if x>3.0 else x)
    #rc_six_m_m = rc_six_m_m.map(lambda x: -3.0 if np.abs(x)>3 else x)
    
    #rc_three_m_m = rc_three_m_m.iloc[0,:].map(lambda x: 3.0 if x>3.0 else x)
    #rc_three_m_m = rc_three_m_m.map(lambda x: -3.0 if np.abs(x)>3 else x)
    
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
    
    # COME BACK HERE  : Winderized our RC distributions

    #rc_six_m_m = rc_six_m_m.iloc[0,:].map(lambda x: 3.0 if x>3.0 else x)
    #rc_six_m_m = rc_six_m_m.map(lambda x: -3.0 if np.abs(x)>3 else x)
    
    #rc_three_m_m = rc_three_m_m.iloc[0,:].map(lambda x: 3.0 if x>3.0 else x)
    #rc_three_m_m = rc_three_m_m.map(lambda x: -3.0 if np.abs(x)>3 else x)
    
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
                
                while Final_Index["MarketCap"].sum()<0.2*Data_Universe["MarketCap"].sum():
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




                                #### OTPIMIZATION FUNCTION ##### 

# Depending on the selected method, the optimization function will apply different optimization processes and returns a
# dataframe containig 

def optimal_weights(Prices_df,Method,Max_Vol,Max_Weight_Allowed,MktCap_df,Nb_Month_1,Nb_Month_2,ThreeM_USD_libor):
    
    df_return=Returns_df(Prices_df)
    #Ranking the inputed dataset
    Ranked_Zscore_df=MSCI_Momentum(Prices_df,Nb_Month_1,Nb_Month_2,ThreeM_USD_libor)
    # If Method is selected as Ranking.
    nbr_sec=Number_of_Securities_Index(Ranked_Zscore_df,MktCap_df)
    #Composition

    if Method=="Ranking":    
      
        # Compute the optimal index as per the MSCI Methodology
        Optimal_Index=Ranked_Zscore_df
        # Rank the computed Z-scores
        Optimal_Index["Ranking"] = (-Optimal_Index['momentum_z_score']).argsort()
        #generate weights
        Optimal_Index['Weights'] = Optimal_Index['Ranking'].map(lambda x: 1 if x <nbr_sec  else 0)
        Optimal_Index['Weights']=Optimal_Index['Weights']/np.sum(Optimal_Index['Weights']) 
        
        #Return Composition
        Composition=Series(Optimal_Index["Weights"],index=Optimal_Index["Weights"].index)
        
        
    elif Method=="Constrained":
        
        
        Non_Ranked_Zscore_df=MSCI_Momentum_No_Ranking(Prices_df,Nb_Month_1,Nb_Month_2,ThreeM_USD_libor)
        constraint_function=index_vol
        constraint_jacobian=index_vol_jacobian    
        #Initiate Weight Array
        Weights_0=np.ones(len(Non_Ranked_Zscore_df))
        Weights_0=Weights_0/np.sum(Weights_0)
                 
        #Define Constraints
        daily_max_var=(Max_Vol/100)**2.0/256.0
        Constraints=({'type': 'ineq',
        'fun' : lambda x : daily_max_var-constraint_function(Prices_df,x),
        'jac' : lambda x : -constraint_jacobian(Prices_df,x)},
        {'type' : 'eq', 
         'fun' : lambda x : np.sum(x)-1,
        'jac' : lambda x : np.ones(len(x))
        })
        
        #Define minimum value for weights (always 0) and maximum (in range (0,1] )
        bnds=[(0,(Max_Weight_Allowed/100))]*len(Weights_0)
        
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
    
    return Composition
    




                                #### BACKTEST FUNCTION ##### 


# The following procedure runs the optimization process at different period in the past.
# It returns the performance of the index over this period.


def back_test(Prices_df,Max_Vol,Max_Weight_Allowed,MktCap_df,Method,t,Nb_Month_1,Nb_Month_2,ThreeM_USD_libor):
      
    df_return=Returns_df(Prices_df)

    # try to find what the "u" is
    u=len(df_return)-t*20
    df_return_bt=df_return.ix[:-t*20]
    
    #set the dataset for backtest
    Prices_df_bt=Prices_df.ix[:-t*20]
        #Compute Optimal Composition
    Weights_bt=optimal_weights(Prices_df_bt,Method,Max_Vol,Max_Weight_Allowed,MktCap_df,Nb_Month_1,Nb_Month_2,ThreeM_USD_libor)
                  
        #compute returns
    next_returns=df_return.ix[u:].fillna(0)
    next_20_returns=next_returns.head(20)
    

    optimum_return=np.dot(next_20_returns,Weights_bt)
    return_series=Series(optimum_return,index=next_20_returns.index)

    for j in range(t-1,0,-1):
            
        s=len(df_return)-j*20  
        df_return_bt=df_return.ix[:-j*20]
        
        #set the dataset for backtest
        Prices_df_bt=Prices_df.ix[:-j*20] 
        
        #Compute Optimal Composition
        Weights_bt=optimal_weights(Prices_df_bt,Method,Max_Vol,Max_Weight_Allowed,MktCap_df,Nb_Month_1,Nb_Month_2,ThreeM_USD_libor)
        #uncomment the following line to test the backtest
        #print Weights_bt[Weights_bt>0]   
        #compute returns

        next_returns=df_return.ix[s:].fillna(0)
        next_20_returns=next_returns.head(20)
        

        optimum_return=np.dot(next_20_returns,Weights_bt)

        return_series_prov=Series(optimum_return,index=next_20_returns.index)
        return_series=np.hstack([return_series,return_series_prov])
        

    return_series_date=Series(return_series,index=df_return.ix[u:].index)    
    

    base_1_backtest=np.ones(len(return_series_date)+1)
    
    for i in range(1,len(base_1_backtest)):
        base_1_backtest[i]=base_1_backtest[i-1]*(1+return_series_date[i-1])
     
    base_1_backtest_date=Series(base_1_backtest,index=df_return.ix[u-1:].index) 
    
    return base_1_backtest_date

