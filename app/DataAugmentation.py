import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error
from sdv.tabular import CTGAN

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from pandas.core.common import SettingWithCopyWarning

#find optimal number of synthetic samples that can be generated
#calculated by minimizing the avg mse of the syntheticDataSize evaluation functions 
def sgd(dataframe,target_name,discrete_columns,sampleSize = 200, lr = 0.01, num_iters = 5):
    start_iter = 0
    prevScore = 0
    theta = sampleSize
    for iter in range(start_iter + 1, num_iters + 1):
        score, grad = getGradients(dataframe,target_name,discrete_columns,prevScore,theta)
        prevScore = score
        theta = theta - (lr * grad)
        
    return theta

def getGradients(dataframe,target_name,discrete_columns,prevScore,sampleSize):
    scores = []
    scores.append(prevScore)
    currentScore = compareSyntheticDataSize(dataframe,target_name,discrete_columns,sampleSize)
    scores.append(currentScore)
    gradients = np.gradient(scores)
    return currentScore, gradients[0]
    

def generate_data(dataframe, sampleSize):
    if sampleSize == 0 or pd.isna(sampleSize) :
        return pd.DataFrame()
    model = CTGAN()
    model.fit(dataframe)
    data = model.sample(int(sampleSize))
    return data

@ignore_warnings(category=ConvergenceWarning)
@ignore_warnings(category=SettingWithCopyWarning)
def compareSyntheticDataSize(dataframe, target_name, discrete_columns,size,title = ""):
    
    synthetic_data = generate_data(dataframe,size)
    leaky_data = pd.concat([dataframe,synthetic_data], axis = 1)
    
    cleanedMixedData,target = processData(leaky_data,target_name,discrete_columns)
    feat_train, feat_test, target_train, target_test = train_test_split(cleanedMixedData,target, test_size=0.10, random_state=(42))
    
    # establish baseline on models  
    lin_model = LinearRegression().fit(feat_train, target_train)
    # mlp_model = MLPRegressor().fit(feat_train, target_train)
    decision_model = DecisionTreeRegressor().fit(feat_train, target_train)
    
    # models = [lin_model, mlp_model, decision_model]
    
    models = [lin_model, decision_model]
    
    training_scores = [model.score(feat_train, target_train) for model in models]
    test_scores = [model.score(feat_test, target_test) for model in models]
    
    training_predictions = [model.predict(feat_train) for model in models]
    test_predictions = [model.predict(feat_test) for model in models]
    
    # training_mses = [mean_squared_error(target_train,prediction) for prediction in training_predictions]
    # test_mses = [mean_squared_error(target_test,prediction) for prediction in test_predictions]
    
    avg_training_score = sum(training_scores) / len(training_scores)
    avg_test_score = sum(test_scores) / len(test_scores)
    # avg_training_mse = sum(training_mses) / len(training_mses)
    # avg_test_mse = sum(test_mses) / len(test_mses)
    
    return avg_test_score

def processData(dataframe, target, categorical_vars = []):
    
    # drop na, null, etc     
    dataframe = dataframe.replace('%','', regex=True)
    dataframe = dataframe.replace('-','', regex=True)
    indices_to_keep = ~dataframe.isin([np.nan, np.inf, -np.inf]).any(1)
    dataframe = dataframe[indices_to_keep]
    
    dataframe = pd.get_dummies(dataframe,columns=categorical_vars, prefix='dmy')
    
    #Drop unencoded variables
    dataframe = dataframe.drop(categorical_vars, axis = 1, errors = 'ignore')
    
    dataframe = dataframe.apply(pd.to_numeric, errors = 'ignore')
    
    dataframe = dataframe.reset_index().dropna()
    dataframe = dataframe.drop('index', axis = 1)
    
    y = dataframe[target]
    y = y.drop(categorical_vars, errors = 'ignore')
    y = y.apply(pd.to_numeric, errors = 'ignore')
    
    return dataframe, y

def generateOptimalDataSamples(dataframe,optimalSamples):
    if not optimalSamples:
        optimalSamples = 0
    syntheticData = generate_data(dataframe,int(optimalSamples))
    return syntheticData
