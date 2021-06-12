##  Naïve Bayesian Modelling Based Hazard Vulnerability in the Context of Demographic Dynamics for North East region in India.

Main project features: Main objective is to find out the most vulnerable region and check the sustainability in North East using Naïve Bayesian modelling.

Research Methodology Description:
This module has been segmented into six segments, that is, identification of influential factors, normalization of influential factors, such as Total Population,
Total Female Population, Total Male Population, Total Female working Population, and Total male Working Population using census data.

Applied ML: Naïve Bayesian modelling using Python.

Results:
Based on this study, the highest risk values (probability score) occurred for the Assam districts such as Dhubri, Barpeta, Sonitpur and Cachar. 
In addition, the highest risk values for mentioned districts like as 0.732, 0.671, 0.734, and 0.694. These districts are significantly affected in the scenario
of natural hazards and disaster.



## Naive bayesian algorithm to predict most vulnerable regions (Natural hazards) in North east India in the context of Dempgramic dynamics
# import the module
import numpy as np      
import pandas as pd

# read the dataset
dt = pd.read_excel('Assam_Census.xlsx') ## File 1

dt.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 26 entries, 0 to 25
Data columns (total 48 columns):
Dist2011     26 non-null object
TPOP2011     26 non-null int64
TMP2011      26 non-null int64
TFP2011      26 non-null int64
MSC2011      26 non-null int64
FSC2011      26 non-null int64
MST2011      26 non-null int64
FST2011      26 non-null int64
MLIT2011     26 non-null int64
FLIT2011     26 non-null int64
MILL2011     26 non-null int64
FILL2011     26 non-null int64
TMW2011      26 non-null int64
TFW2011      26 non-null int64
NONWM2011    26 non-null int64
NONWF2011    26 non-null int64
dtypes: float64(30), int64(15), object(3)
memory usage: 9.9+ KB

dt.describe()
Out[4]: 
             TPOP91          TMP91  ...      NONWM2011     NONWF2011
count  2.200000e+01      22.000000  ...      26.000000  2.600000e+01
mean   9.004642e+05  465558.681818  ...  280068.653846  4.486859e+05
std    3.252570e+05  168244.092904  ...  125015.738533  2.229502e+05
min    4.149100e+05  215176.000000  ...   54479.000000  7.459500e+04
25%    7.102075e+05  365508.750000  ...  215185.000000  3.286745e+05
50%    7.914410e+05  411957.500000  ...  245074.000000  3.764280e+05
75%    1.151911e+06  591191.750000  ...  312311.250000  5.153452e+05
max    1.687449e+06  871978.000000  ...  672384.000000  1.171386e+06

[8 rows x 45 columns]

Tr2011 = dt.iloc[:, 32:48]

names = dt.iloc[:,32].to_frame() 
dtt = Tr2011.sum(axis = 0, skipna = True)

names.head()
Out[7]: 
     Dist2011
0      DHUBRI
1   KOKRAJHAR
2  BONGAIGAON
3    GOALPARA
4     BARPETA

dtt.head()
Out[8]: 
Dist2011    DHUBRIKOKRAJHARBONGAIGAONGOALPARABARPETANALBAR...
TPOP2011                                             30723414
TMP2011                                              15694583
TFP2011                                              15028831
MSC2011                                               1127209
dtype: object

##### Calculate the ratio for each observation

Tpop = Tr2011[['TPOP2011']]/dtt[[1]]
TFP = Tr2011[['TFP2011']]/dtt[[3]]
TMP = Tr2011[['TMP2011']]/dtt[[2]]
TFW = Tr2011[['TFW2011']]/dtt[[13]]
TMW = Tr2011[['TMW2011']]/dtt[[12]] 

Tpop
Out[11]: 
array([[0.06344536],
       [0.02887511],
       [0.02404694],
       [0.03281481],
       [0.0551248 ],
       [0.02511567],
       [0.04939366],
       [0.03022125],
       [0.06262683],
       [0.03391996],
       [0.02233258],
       [0.03116265],
       [0.09190932],
       [0.03472557],
       [0.03555126],
       [0.03746491],
       [0.04317017],
       [0.04322205],
       [0.03112652],
       [0.03999184],
       [0.02145907],
       [0.05652422],
       [0.03092348],
       [0.00696869],
       [0.02706952],
       [0.04081376]])
       
## Wrt Total Population 
       
 TpopMax = Tpop.max()
TpopMin = Tpop.min()
TpopMean = (TpopMax + TpopMin)/2

TpopMax
Out[13]: 0.09190931710909471

TpopMin
Out[14]: 0.0069686916955257645

TpopMean
Out[15]: 0.049439004402310235

## Prior probabiity (Wrt Total Population)

Prob_H = Tpop/TpopMax    # put the value of  TpopMax
Prob_M = abs((TpopMean - Tpop)/ TpopMean)
Prob_L = abs((TpopMax - Tpop)/ TpopMax)

Prob_H
Out[17]: 
array([[0.69030388],
       [0.31416958],
       [0.26163764],
       [0.35703464],
       [0.59977378],
       [0.27326572],
       [0.53741738],
       [0.32881597],
       [0.68139805],
       [0.369059  ],
       [0.24298491],
       [0.33905866],
       [1.        ],
       [0.37782424],
       [0.38680798],
       [0.4076291 ],
       [0.46970396],
       [0.47026845],
       [0.33866557],
       [0.43512286],
       [0.23348094],
       [0.61499989],
       [0.33645647],
       [0.07582138],
       [0.2945242 ],
       [0.44406552]])

########## Composite index  #############################################

  x = [Tpop]; p = [TFP]; q = [TMP]; r = [TFW]; s = [TMW]  
matrix = [(x*5)+(p*4)+(q*3)+(r*2)+(s*1) for x,p,q,r,s in zip(x,p,q,r,s)]
##########################################################################

matrix
Out[19]: 
[array([[0.90976521],
        [0.43694364],
        [0.34681357],
        [0.48242613],
        [0.7835897 ],
        [0.3603857 ],
        [0.76262995],
        [0.43830133],
        [0.95411255],
        [0.5286561 ],
        [0.36755712],
        [0.46020534],
        [1.32113585],
        [0.55517742],
        [0.57060965],
        [0.58366488],
        [0.67637661],
        [0.67543808],
        [0.48428107],
        [0.56525349],
        [0.30356677],
        [0.81644542],
        [0.48641832],
        [0.10728294],
        [0.42049649],
        [0.60246668]])]     
        
CI = matrix
CI_Max = np.max(matrix)
CI_Min = np.min(matrix)
CI_Mean = (CI_Max + CI_Min)/2

CI_Max
Out[21]: 1.3211358543552831

CI_Min
Out[22]: 0.10728294104473408

CI_Mean
Out[23]: 0.7142093977000086

 ## Conditional Probalility 

Prob_CI_H = CI / CI_Max   
Prob_CI_M = abs((CI_Mean - CI) / CI_Mean) 
Prob_CI_L = abs((CI_Max - CI) / CI_Max)

Prob_CI_H
Out[25]: 
array([[[0.68862351],
        [0.33073332],
        [0.26251166],
        [0.36516012],
        [0.59311819],
        [0.27278474],
        [0.57725324],
        [0.33176098],
        [0.7221911 ],
        [0.40015271],
        [0.27821297],
        [0.34834066],
        [1.        ],
        [0.42022735],
        [0.43190838],
        [0.44179021],
        [0.51196598],
        [0.51125558],
        [0.36656417],
        [0.42785417],
        [0.2297771 ],
        [0.61798748],
        [0.36818191],
        [0.08120508],
        [0.31828406],
        [0.45602175]]])

 ## Numerator

NUM_H = (Prob_H * Prob_CI_H)
NUM_M = (Prob_M * Prob_CI_M)
NUM_L = (Prob_L * Prob_CI_L)

NUM_H
Out[27]: 
array([[[0.47535948],
        [0.10390635],
        [0.06868293],
        [0.13037481],
        [0.35573674],
        [0.07454272],
        [0.31022592],
        [0.10908831],
        [0.4920996 ],
        [0.14767996],
        [0.06760155],
        [0.11810792],
        [1.        ],
        [0.15877208],
        [0.16706561],
        [0.18008654],
        [0.24047244],
        [0.24042737],
        [0.12414266],
        [0.18616913],
        [0.05364857],
        [0.38006223],
        [0.12387718],
        [0.00615708],
        [0.09374236],
        [0.20250353]]])
        
## Denominator   
        
  Deno = (NUM_H + NUM_M + NUM_L)

Deno
Out[29]: 
array([[[0.64936278],
        [0.72438517],
        [0.87741911],
        [0.64768066],
        [0.52975353],
        [0.84676825],
        [0.5058434 ],
        [0.70776571],
        [0.67021127],
        [0.60770091],
        [0.88012128],
        [0.68028902],
        [1.7300057 ],
        [0.58576048],
        [0.57189427],
        [0.55502342],
        [0.50599172],
        [0.50615713],
        [0.66230176],
        [0.54921448],
        [0.96943749],
        [0.54765163],
        [0.66256357],
        [1.5852934 ],
        [0.76074964],
        [0.53221558]]])
        
 ## Naive bayes Theorem (Posterior Probability)
      
Prob_H_CI = NUM_H/ Deno
Prob_M_CI = NUM_M/ Deno
Prob_L_CI = NUM_L/ Deno

Prob_H_CI
Out[31]: 
array([[[0.73203993],
        [0.14344074],
        [0.07827836],
        [0.2012949 ],
        [0.67151366],
        [0.08803202],
        [0.61328451],
        [0.15413053],
        [0.73424549],
        [0.24301422],
        [0.07680936],
        [0.17361432],
        [0.57803278],
        [0.27105291],
        [0.29212674],
        [0.32446656],
        [0.47524976],
        [0.4750054 ],
        [0.18744124],
        [0.33897346],
        [0.0553399 ],
        [0.69398539],
        [0.18696649],
        [0.00388388],
        [0.12322366],
        [0.38049156]]])
        
     verify_result = Prob_H_CI + Prob_M_CI + Prob_L_CI
     
     verify_result
Out[33]: 
array([[[1.],
        [1.],
        [1.],
        [1.]]])
        
        
    frames1 = [names, Prob_H_CI]
frames2 = [names,Prob_M_CI]
frames3 = [names,Prob_L_CI]
result_H = np.concatenate(frames1, axis=None)
result_M = np.concatenate(frames2, axis=None)
result_L = np.concatenate(frames3, axis=None)     
frames = [result_H, result_M, result_L]

frames
Out[35]: 
[array(['DHUBRI', 'KOKRAJHAR', 'BONGAIGAON',
        'Kamrup Metropolitan', 0.7320399278834355, 0.14344074199052684,
        0.0782783629751842],
       dtype=object),
 array(['DHUBRI', 'KOKRAJHAR', 'BONGAIGAON', 0.11945742599867147, 0.2229136907900462,
        0.3011132859084573],
       dtype=object),
 array(['DHUBRI', 'KOKRAJHAR', 'BONGAIGAON', 0.14850264611789307, 0.6336455672194269,
        0.6206083511163585,], dtype=object)]
        
        
        my_df = pd.DataFrame(frames)
df = my_df.T
prob_H_M_L = df.to_csv('F:/AssamProb.csv')

df
Out[37]: 
                      0                    1                    2
0                DHUBRI               DHUBRI               DHUBRI
1             KOKRAJHAR            KOKRAJHAR            KOKRAJHAR
2            BONGAIGAON           BONGAIGAON           BONGAIGAON
3              GOALPARA             GOALPARA             GOALPARA

4              0.73204             0.119457             0.148503
5             0.143441             0.222914             0.633646
6            0.0782784             0.301113             0.620608
7             0.201295             0.168487             0.630218


CDVI = [(Prob_H_CI - (Prob_M_CI + Prob_L_CI))]
mx = np.max(CDVI); mn = np.min(CDVI);
NCDVI = [(CDVI - mn)/ (mx-mn)]
frm = [names, NCDVI]
ncdvi = np.concatenate(frm, axis=None)
ncdvi = pd.DataFrame(ncdvi)
exp = ncdvi.to_csv('F:/AssamNCDVI.csv')

ncdvi
Out[39]: 
                      0
0                DHUBRI
1             KOKRAJHAR
2            BONGAIGAON
3              GOALPARA
4               BARPETA

5              0.99698
6             0.191079
7              0.10186
8             0.270292
9             0.914109


############# Build Multiple models to predict Naive bayesian based (CDVI) ###########

import numpy as np 
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, BayesianRidge, ElasticNet
from sklearn.linear_model import  Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# read the dataset
data= pd.read_excel('Assam.xlsx')  ## File 2

names = data.columns

data.head()
Out[43]: 
     District     TOTP    FPOP    MPOP    TWFP    TWMP  CDVI - DM
0      DHUBRI  1949258  951410  997848  144921  524898   0.996906
1   KOKRAJHAR   887142  434237  452905  104809  236322   0.191315
2  BONGAIGAON   738804  362986  375818   58264  197542   0.101849
3    GOALPARA  1008183  494891  513292   95455  267118   0.270420
4     BARPETA  1693622  826618  867004  116527  445297   0.913938

data.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 26 entries, 0 to 25
Data columns (total 7 columns):
District     26 non-null object
TOTP         26 non-null int64
FPOP         26 non-null int64
MPOP         26 non-null int64
TWFP         26 non-null int64
TWMP         26 non-null int64
CDVI - DM    26 non-null float64
dtypes: float64(1), int64(5), object(1)
memory usage: 1.5+

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
scoreOfModel4 = reg.score(X_trn, y_trn)

pred4 = reg.predict(X_tst)
pred4 = pd.DataFrame(pred4) 
print('r2 score MLR:', {r2_score(y_tst, pred4)}) #model Evaluation
###################################################################################
######################### Ridge Regression (L2) ###################################
rg = Ridge(alpha=1.0)
rg.fit(X_trn, y_trn)
rg.score(X_trn, y_trn)
rg.__dict__
scoreOfModel5 = rg.score(X_trn, y_trn)

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
scoreOfModel6 = ls.score(X_trn, y_trn)

pred6 = ls.predict(X_tst)
pred6 = pd.DataFrame(pred6)
print('r2 score:', {r2_score(y_tst, pred6)})
############### Bayesian regression ###############################################
Bs = BayesianRidge() 
Bs.fit(X_trn, y_trn) 
Bs.coef_
Bs.intercept_
scoreOfModel7 = Bs.score(X_trn, y_trn)

pred7 = Bs.predict(X_tst)
pred7 = pd.DataFrame(pred7)
print('r2 score BSR:', {r2_score(y_tst, pred7)}) #model Evaluation

############## ElasticNet Regression (L1 + L2 penalized model) ###########
## hyperparameter that determines strength of a Ridge,lasso, elastcNet regression 
# we use alpha as our hyperparameter lambda ## l1 for lasso
enet = ElasticNet(alpha=0.005, l1_ratio=0.7)
enet.fit(X_trn, y_trn)
enet.alpha
scoreOfModel8 = enet.score(X_trn, y_trn)

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

""" Comparing multiple models """
########################################################################################
models = pd.DataFrame({'Models': ['RFR','DTR','SVR','MLR','Ridge','Lasso', 'Bayesian Reg', 'E_Net'],   
    'Score': [scoreOfModel1,scoreOfModel2, scoreOfModel3, scoreOfModel4,
              scoreOfModel5, scoreOfModel6, scoreOfModel7, scoreOfModel8] })
models.sort_values(by='Score', ascending=False)
Model Score RFR:  0.9960304659730086
Model Score DTR:  0.9999999627623446
Model Score SVR:  0.9441311411367278
r2 score MLR: {0.5832025256283164}
r2 score Ridge: {0.7908128232597118}
r2 score: {0.6584885907827149}
r2 score BSR: {0.7959034039856475}
r2 score: {0.7880180238787802}
Model predicted RFR:            
0  0.472721
1  0.106310
2  0.930598
3  0.248909
4  0.106310
5  0.175586
6  0.644064
Model predicted DTR:            
0  0.439752
1  0.100177
2  0.913938
3  0.232522
4  0.100177
5  0.191315
6  0.646323
Model predicted SVR:            
0  0.464426
1  0.234200
2  0.864212
3  0.311225
4  0.441231
5  0.207043
6  0.549665
Model predicted MLR:            
0  0.267335
1  0.135543
2  0.615542
3  0.128377
4 -0.049331
5  0.083017
6  0.820158
Model predicted Ridge:            
0  0.357399
1  0.078817
2  0.650726
3  0.340330
4 -0.119258
5  0.254323
6  0.436612
Model predicted Lasso:            
0  0.430079
1  0.260855
2  0.603414
3  0.379477
4  0.132566
5  0.336399
6  0.496463
Model predicted Bayesian ridge Reg:            
0  0.364181
1  0.091109
2  0.649801
3  0.341836
4 -0.103960
5  0.258877
6  0.440005
Model predicted ElasticNet:           
0  0.356781
1  0.078319
2  0.648839
3  0.336908
4 -0.120599
5  0.251242
6  0.428349

## Model scores
Out[48]: 
         Models     Score
1           DTR  1.000000
0           RFR  0.996030
2           SVR  0.944131
3           MLR  0.857221
7         E_Net  0.758489
4         Ridge  0.755402
6  Bayesian Reg  0.754097
5         Lasso  0.626048


#### write the output file
al = pd.concat([pred1,pred2,pred3,pred4,pred5,pred6,pred7,pred8, y_tst], axis =1)

al.to_csv('F:/Assam_pred_8_Algo.csv')
