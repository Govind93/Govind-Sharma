## Demographic dynamics _Disaster Risk reduction ( Naive bayesian Algorithm)

# Python with Pandas 
import numpy as np      
import pandas as pd 
## Read the dataset
dt = pd.read_excel('F:/MY FILE/New Results/NE Data/NE Raw Data/Census/Assam.xlsx')
dt.info()
dt.describe()  

Tr2011 = dt.iloc[:, 32:48]
names = dt.iloc[:,32].to_frame() 
dtt = Tr2011.sum(axis = 0, skipna = True)

#### Calculate the ratio for each observation
Tpop = Tr2011[['TPOP2011']]/dtt[[1]]
TFP = Tr2011[['TFP2011']]/dtt[[3]]
TMP = Tr2011[['TMP2011']]/dtt[[2]]
TFW = Tr2011[['TFW2011']]/dtt[[13]]
TMW = Tr2011[['TMW2011']]/dtt[[12]] 
Tpop = Tpop.values
TFP = TFP.values
TMP = TMP.values
TFW = TFW.values
TMW = TMW.values

## Wrt Total Population 
TpopMax = Tpop.max()
TpopMin = Tpop.min()
TpopMean = (TpopMax + TpopMin)/2 
 ## Prior probabiity (Wrt Total Population)
Prob_H = Tpop/TpopMax    # put the value of  TpopMax
Prob_M = abs((TpopMean - Tpop)/ TpopMean)
Prob_L = abs((TpopMax - Tpop)/ TpopMax)
  
########## Composite index  ##########################################
x = [Tpop]; p = [TFP]; q = [TMP]; r = [TFW]; s = [TMW]  
matrix = [(x*5)+(p*4)+(q*3)+(r*2)+(s*1) for x,p,q,r,s in zip(x,p,q,r,s)]
################################################################    
  #CI = np.matrix.T
CI = matrix
CI_Max = np.max(matrix)
CI_Min = np.min(matrix)
CI_Mean = (CI_Max + CI_Min)/2
 
 ## Conditional Probalility 
Prob_CI_H = CI / CI_Max   
Prob_CI_M = abs((CI_Mean - CI) / CI_Mean) 
Prob_CI_L = abs((CI_Max - CI) / CI_Max)   
 
 ## Numerator
NUM_H = (Prob_H * Prob_CI_H)
NUM_M = (Prob_M * Prob_CI_M)
NUM_L = (Prob_L * Prob_CI_L) 
   
## Denominator   
Deno = (NUM_H + NUM_M + NUM_L)   
## Naive bayes Thm (Posterior Probability)
Prob_H_CI = NUM_H/ Deno
Prob_M_CI = NUM_M/ Deno
Prob_L_CI = NUM_L/ Deno

verify_result = Prob_H_CI + Prob_M_CI + Prob_L_CI
        
frames1 = [names, Prob_H_CI]
frames2 = [names,Prob_M_CI]
frames3 = [names,Prob_L_CI]
result_H = np.concatenate(frames1, axis=None)
result_M = np.concatenate(frames2, axis=None)
result_L = np.concatenate(frames3, axis=None)     
frames = [result_H, result_M, result_L]

my_df = pd.DataFrame(frames)
df = my_df.T
prob_H_M_L = df.to_csv('F:/AssamProb.csv')

######## calcualte the NCDVI with [(P(H),P(M),P(L)] ##############################
CDVI = [(Prob_H_CI - (Prob_M_CI + Prob_L_CI))]
mx = np.max(CDVI); mn = np.min(CDVI);
NCDVI = [(CDVI - mn)/ (mx-mn)]
frm = [names, NCDVI]
ncdvi = np.concatenate(frm, axis=None)
ncdvi = pd.DataFrame(ncdvi)
exp = ncdvi.to_csv('F:/AssamNCDVI.csv')
###########################################################

######################################################################################
############# Build Multiple models to predict Naive bayesian based (CDVI) ###########
## import module
import numpy as np 
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, BayesianRidge, ElasticNet
from sklearn.linear_model import  Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score 
#from subprocess import check_output
#from sklearn.model_selection import cross_val_score
###### Read the data
data= pd.read_excel('C:/Users/dell/Desktop/North-East India/Raw Data/Assam.xlsx')
names = data.columns
data.info()
data.describe()
data.head()
##data.drop('MPOP', axis = 1, inplace = True)
##data = data.drop(data[(data['CDVI - DM']<0.50) & (data['TOT']<30)].index)
######################### Splitting the data #######################################
X = data.iloc[:, 1:6]
y = data.iloc[:, 6]
y = pd.DataFrame(y)

X_trn = X[:19]
y_trn = y[:19]
X_tst = X[19:26]
y_tst = y[19:26]
y_tst = pd.DataFrame(y_tst)
X_tst = pd.DataFrame(X_tst) 
################################ Future Scaling ##################################
scaler = StandardScaler()
scaler.fit(X_trn)
X_trn = scaler.transform(X_trn)
X_tst = scaler.transform(X_tst)

################ Random Forest Regressor ##########################################
md = RandomForestRegressor(n_jobs=-1)
clss = RandomForestRegressor(n_estimators=30, criterion='mse', max_depth=30) 
clss.fit(X_trn, y_trn)
scoreOfModel1 = clss.score(X_trn, y_trn)
print("Model Score RFR: ",scoreOfModel1)    

pred1 = clss.predict(X_tst)
pred1 = pd.DataFrame(pred1)
###################################################################################
#################### Decision Tree Regressor ######################################
clss = DecisionTreeRegressor()
clss.fit(X_trn, y_trn)
scoreOfModel2 = clss.score(X_trn, y_trn)
print("Model Score DTR: ",scoreOfModel2)

pred2 = clss.predict(X_tst)
pred2 = pd.DataFrame(pred2)
####################################################################################
###################### Support vector regressor ####################################
sv = SVR(kernel = 'rbf', C=1.0) # radial basis function(rbf)
sv.fit(X_trn, y_trn)
scoreOfModel3 = sv.score(X_trn, y_trn)
print("Model Score SVR: ",scoreOfModel3)

pred3 = sv.predict(X_tst)
pred3 = pd.DataFrame(pred3)
####################################################################################
################### Multiple Linear Regression #####################################
reg = LinearRegression()
reg.fit(X_trn, y_trn)

pred4 = reg.predict(X_tst)
pred4 = pd.DataFrame(pred4) 
print('r2 score MLR:', {r2_score(y_tst, pred4)}) #model Evaluation
###################################################################################
######################### Ridge Regression (L2) ###################################
rg = Ridge(alpha=1.0)
rg.fit(X_trn, y_trn)
rg.score(X_trn, y_trn)
rg.__dict__

pred5 = rg.predict(X_tst)
pred5 = pd.DataFrame(pred5)
print('r2 score Ridge:', {r2_score(y_tst, pred5)}) #model Evaluation

##################################################################################
#########################  Lasso regression (L1) #################################
ls = Lasso(alpha=0.1)
ls.fit(X_trn, y_trn)
ls.score(X_trn, y_trn)
ls.intercept_
ls.coef_
ls.__dict__
pred6 = ls.predict(X_tst)
pred6 = pd.DataFrame(pred6)
print('r2 score:', {r2_score(y_tst, pred6)})
############### Bayesian regression ###############################################
Bs = BayesianRidge() 
Bs.fit(X_trn, y_trn) 
Bs.coef_
Bs.intercept_

pred7 = Bs.predict(X_tst)
pred7 = pd.DataFrame(pred7)
print('r2 score BSR:', {r2_score(y_tst, pred7)}) #model Evaluation

############## ElasticNet Regression (L1 + L2 penalized model) ###########
## hyperparameter that determines strength of a Ridge,lasso, elastcNet regression 
# we use alpha as our hyperparameter lambda ## l1 for lasso
enet = ElasticNet(alpha=0.005, l1_ratio=0.7)
enet.fit(X_trn, y_trn)
enet.alpha
pred8 = enet.predict(X_tst)
pred8 = pd.DataFrame(pred8)
print('r2 score:', {r2_score(y_tst, pred8)})

################################################################################
print("Model predicted RFR: ",pred1)
print("Model predicted DTR: ",pred2)
print("Model predicted SVR: ",pred3)
print("Model predicted MLR: ",pred4)
print("Model predicted Ridge: ",pred5)
print("Model predicted Lasso: ",pred6)
print("Model predicted Bayesian ridge Reg: ",pred7)
print("Model predicted ElasticNet:", pred8)

al = pd.concat([pred1,pred2,pred3,pred4,pred5,pred6,pred7,pred8, y_tst], axis =1)
al.to_csv('F:/Assam_pred_8_Algo.csv')
#################################################################################



