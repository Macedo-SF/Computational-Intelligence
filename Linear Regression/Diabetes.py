import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from math import sqrt
from sklearn import linear_model
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
from statsmodels.stats.diagnostic import normal_ad

# fig directory
# decided to save the figures instead of showing, comment plt.close() and uncomment plt.show() to switch

my_path = 'C:/Users/Saulo/source/repos/IC/Linear Regression' # set ur path in order to save the figures or do as stated above and use show()

# testing functions

def my3dPlot(x,y, model,score,mean,root,name1,name2):
    x1=x[:, 0]
    y1=x[:, 1]
    z1=pd.Series(y.flatten())
    x2, y2 =np.meshgrid(x1, y1)
    model_viz = np.array([x2.flatten(), y2.flatten()]).T
    predicted = model.predict(model_viz)
    # plotting fixes

    plt.style.use('default')

    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')

    axes = [ax1, ax2, ax3]

    for ax in axes:
        ax.plot(x1, y1, z1, color='k', zorder=15, linestyle='none', marker='o', alpha=0.5)
        ax.scatter(x2, y2, predicted, facecolor=(0,0,0,0), s=20, edgecolor='#70b3f0')
        ax.set_xlabel(name1, fontsize=12)
        ax.set_ylabel(name2, fontsize=12)
        ax.set_zlabel('Diabetes', fontsize=12)
        ax.locator_params(nbins=4, axis='x')
        ax.locator_params(nbins=5, axis='x')

    ax1.text2D(0.2, 0.32, '', fontsize=13, ha='center', va='center',
               transform=ax1.transAxes, color='grey', alpha=0.5)
    ax2.text2D(0.3, 0.42, '', fontsize=13, ha='center', va='center',
               transform=ax2.transAxes, color='grey', alpha=0.5)
    ax3.text2D(0.85, 0.85, '', fontsize=13, ha='center', va='center',
               transform=ax3.transAxes, color='grey', alpha=0.5)

    ax1.view_init(elev=28, azim=120)
    ax2.view_init(elev=4, azim=114)
    ax3.view_init(elev=60, azim=165)

    fig.suptitle('$R^2 = %.5f$' % score, fontsize=20)

    fig.tight_layout()
    print(name1+' and '+name2+': \nR2: ', score, '\nMean Absolute Error: ',
         mean,'\nRoot Mean Square Error: ', root)
    fig.savefig(my_path+'/Figures/'+name1+'_'+name2+'_Graph'+'.png')
    plt.close()
    #plt.show()

def my2dPlot(x,y,pred,name,score,mean,root):
    fig = plt.figure()
    plt.scatter(x,y,  color='blue')
    plt.plot(x, pred, color='black', linewidth=3)
    plt.title(name+' vs Diabetes', fontsize=14)
    plt.xlabel(name, fontsize=14)
    plt.ylabel('Diabetes', fontsize=14)
    plt.grid(True)
    print(name+': \nR2: ', score, '\nMean Absolute Error: ', mean,'\nRoot Mean Square Error: ', root)
    fig.savefig(my_path+'/Figures/'+name+'_Graph'+'.png')
    plt.close()
    #plt.show()

def myResiduals2(y,pred,name1,name2):
    res = pd.Series(y.flatten()-pred)
    print('Anderson-Darling Normal: ',normal_ad(res)[1],'\n\n') # if < 0.05 -> bad
    fig = plt.figure()
    plt.hist(res)
    plt.title(name1+' and '+name2+' Residuals')
    fig.savefig(my_path+'/Figures/'+name1+'_'+name2+'_ResidualHistogram'+'.png')
    plt.close()
    #plt.show()
    fig = plt.figure()
    plt.scatter(pred,res, color='green', s=50, alpha=.6)
    plt.hlines(y=0, xmin=min(pred), xmax=max(pred), color='black')
    plt.ylabel(name1+'and '+name2+' Residuals')
    plt.xlabel(name1+'and '+name2+' Prediction')
    plt.title ('Residuals vs Preditions ('+name1+' - '+name2+')')
    fig.savefig(my_path+'/Figures/'+name1+'_'+name2+'_ResidualGraph'+'.png')
    plt.close()
    #plt.show()
def myResiduals(y,pred,name):
    res = pd.Series(y.flatten()-pred)
    print('Anderson-Darling Normal: ',normal_ad(res)[1],'\n\n') # if < 0.05 -> bad
    fig = plt.figure()
    plt.hist(res)
    plt.title(name+' Residuals')
    fig.savefig(my_path+'/Figures/'+name+'_ResidualHistogram'+'.png')
    plt.close()
    #plt.show()
    fig = plt.figure()
    plt.scatter(pred,res, color='green', s=50, alpha=.6)
    plt.hlines(y=0, xmin=min(pred), xmax=max(pred), color='black')
    plt.ylabel(name+' Residuals')
    plt.xlabel(name+' Prediction')
    plt.title ('Residuals vs Preditions ('+name+')')
    fig.savefig(my_path+'/Figures/'+name+'_ResidualGraph'+'.png')
    plt.close()
    #plt.show()

def myDistribution2(var,name1,name2):
    fig = plt.figure()
    plt.hist(var)
    plt.title(name1+' and '+name2+' Distribution')
    fig.savefig(my_path+'/Figures/'+name1+'_'+name2+'_Distribution'+'.png')
    plt.close()
    #plt.show()

def myDistribution(var,name):
    fig = plt.figure()
    plt.hist(var)
    plt.title(name+' Distribution')
    fig.savefig(my_path+'/Figures/'+name+'_Distribution'+'.png')
    plt.close()
    #plt.show()

# test end

# Load the diabetes dataset
diabetes = load_diabetes()
# diabetes
table = pd.DataFrame(diabetes.data)
table.columns = diabetes.feature_names
table['y'] = diabetes.target
y = table['y']
#myDistribution(y,"y")
#print(table.head()) # not enough space in screen to see all columns and lines
#print(table.corr())
# splice data
# example
    # splitting variable
    # splitting training data
    # splitting testing data
    # fitting linear model with training data
    # predicting with test data
    # score, r^2
    # mean absolute error
    # root mean squared error
    # y=Slope*x+Intercept
# example end
# data *******************************************************************************************************

# age

Xa = table['age']
Xa_training = Xa[:-20].array.to_numpy().reshape(-1,1)
Xa_test = Xa[-20:].array.to_numpy().reshape(-1,1)
ya_training = y[:-20]
ya_test = y[-20:].array.to_numpy().reshape(-1,1)

regrA = linear_model.LinearRegression()
regrA.fit(Xa_training, ya_training)
diabetes_ya_pred = regrA.predict(Xa_test)
scoreA = regrA.score(Xa_test,ya_test)
maeA = sum(abs(ya_test - diabetes_ya_pred.reshape(-1,1)))/ya_test.size
rmseA = sqrt(mean_squared_error(ya_test,diabetes_ya_pred.reshape(-1,1)))
slopeA = regrA.coef_
interceptA = regrA.intercept_

# sex

Xs = table['sex']
Xs_training = Xs[:-20].array.to_numpy().reshape(-1,1)
Xs_test = Xs[-20:].array.to_numpy().reshape(-1,1)
ys_training = y[:-20]
ys_test = y[-20:].array.to_numpy().reshape(-1,1)

regrS = linear_model.LinearRegression()
regrS.fit(Xs_training, ys_training)
diabetes_ys_pred = regrS.predict(Xs_test)
scoreS = regrS.score(Xs_test,ys_test)
maeS =sum(abs(ys_test - diabetes_ys_pred.reshape(-1,1)))/ys_test.size
rmseS = sqrt(mean_squared_error(ys_test,diabetes_ys_pred.reshape(-1,1)))
slopeS = regrS.coef_
interceptS = regrS.intercept_

# body mass index

Xb = table['bmi']
Xb_training = Xb[:-20].array.to_numpy().reshape(-1,1)
Xb_test = Xb[-20:].array.to_numpy().reshape(-1,1)
yb_training = y[:-20]
yb_test = y[-20:].array.to_numpy().reshape(-1,1)

regrB = linear_model.LinearRegression()
regrB.fit(Xb_training, yb_training)
diabetes_yb_pred = regrB.predict(Xb_test)
scoreB = regrB.score(Xb_test,yb_test)
maeB = sum(abs(yb_test - diabetes_yb_pred.reshape(-1,1)))/yb_test.size
rmseB = sqrt(mean_squared_error(yb_test,diabetes_yb_pred.reshape(-1,1)))
slopeB = regrB.coef_
interceptB = regrB.intercept_

# blood pressure

Xbp = table['bp']
Xbp_training = Xbp[:-20].array.to_numpy().reshape(-1,1)
Xbp_test = Xbp[-20:].array.to_numpy().reshape(-1,1)
ybp_training = y[:-20]
ybp_test = y[-20:].array.to_numpy().reshape(-1,1)

regrBP = linear_model.LinearRegression()
regrBP.fit(Xbp_training, ybp_training)
diabetes_ybp_pred = regrBP.predict(Xbp_test)
scoreBP = regrBP.score(Xbp_test,ybp_test)
maeBP = sum(abs(ybp_test - diabetes_ybp_pred.reshape(-1,1)))/ybp_test.size
rmseBP = sqrt(mean_squared_error(ybp_test,diabetes_ybp_pred.reshape(-1,1)))
slopeBP = regrBP.coef_
interceptBP = regrBP.intercept_

# s1

Xs1 = table['s1']
Xs1_training = Xs1[:-20].array.to_numpy().reshape(-1,1)
Xs1_test = Xs1[-20:].array.to_numpy().reshape(-1,1)
ys1_training = y[:-20]
ys1_test = y[-20:].array.to_numpy().reshape(-1,1)

regrs1 = linear_model.LinearRegression()
regrs1.fit(Xs1_training, ys1_training)
diabetes_ys1_pred = regrs1.predict(Xs1_test)
scores1 = regrs1.score(Xs1_test,ys1_test)
maes1 =sum(abs(ys1_test - diabetes_ys1_pred.reshape(-1,1)))/ys1_test.size
rmses1 = sqrt(mean_squared_error(ys1_test,diabetes_ys1_pred.reshape(-1,1)))
slopes1 = regrs1.coef_
intercepts1 = regrs1.intercept_

# s2

Xs2 = table['s2']
Xs2_training = Xs2[:-20].array.to_numpy().reshape(-1,1)
Xs2_test = Xs2[-20:].array.to_numpy().reshape(-1,1)
ys2_training = y[:-20]
ys2_test = y[-20:].array.to_numpy().reshape(-1,1)

regrs2 = linear_model.LinearRegression()
regrs2.fit(Xs2_training, ys2_training)
diabetes_ys2_pred = regrs2.predict(Xs2_test)
scores2 = regrs2.score(Xs2_test,ys2_test)
maes2 =sum(abs(ys2_test - diabetes_ys2_pred.reshape(-1,1)))/ys2_test.size
rmses2 = sqrt(mean_squared_error(ys2_test,diabetes_ys2_pred.reshape(-1,1)))
slopes2 = regrs2.coef_
intercepts2 = regrs2.intercept_

# s3

Xs3 = table['s3']
Xs3_training = Xs3[:-20].array.to_numpy().reshape(-1,1)
Xs3_test = Xs3[-20:].array.to_numpy().reshape(-1,1)
ys3_training = y[:-20]
ys3_test = y[-20:].array.to_numpy().reshape(-1,1)

regrs3 = linear_model.LinearRegression()
regrs3.fit(Xs3_training, ys3_training)
diabetes_ys3_pred = regrs3.predict(Xs3_test)
scores3 = regrs3.score(Xs3_test,ys3_test)
maes3 =sum(abs(ys3_test - diabetes_ys3_pred.reshape(-1,1)))/ys3_test.size
rmses3 = sqrt(mean_squared_error(ys3_test,diabetes_ys3_pred.reshape(-1,1)))
slopes3 = regrs3.coef_
intercepts3 = regrs3.intercept_

# s4

Xs4 = table['s4']
Xs4_training = Xs4[:-20].array.to_numpy().reshape(-1,1)
Xs4_test = Xs4[-20:].array.to_numpy().reshape(-1,1)
ys4_training = y[:-20]
ys4_test = y[-20:].array.to_numpy().reshape(-1,1)

regrs4 = linear_model.LinearRegression()
regrs4.fit(Xs4_training, ys4_training)
diabetes_ys4_pred = regrs4.predict(Xs4_test)
scores4 = regrs4.score(Xs4_test,ys4_test)
maes4 =sum(abs(ys4_test - diabetes_ys4_pred.reshape(-1,1)))/ys4_test.size
rmses4 = sqrt(mean_squared_error(ys4_test,diabetes_ys4_pred.reshape(-1,1)))
slopes4 = regrs4.coef_
intercepts4 = regrs4.intercept_

# s5

Xs5 = table['s5']
Xs5_training = Xs5[:-20].array.to_numpy().reshape(-1,1)
Xs5_test = Xs5[-20:].array.to_numpy().reshape(-1,1)
ys5_training = y[:-20]
ys5_test = y[-20:].array.to_numpy().reshape(-1,1)

regrs5 = linear_model.LinearRegression()
regrs5.fit(Xs5_training, ys5_training)
diabetes_ys5_pred = regrs5.predict(Xs5_test)
scores5 = regrs5.score(Xs5_test,ys5_test)
maes5 =sum(abs(ys5_test - diabetes_ys5_pred.reshape(-1,1)))/ys5_test.size
rmses5 = sqrt(mean_squared_error(ys5_test,diabetes_ys5_pred.reshape(-1,1)))
slopes5 = regrs5.coef_
intercepts5 = regrs5.intercept_

# s6

Xs6 = table['s6']
Xs6_training = Xs6[:-20].array.to_numpy().reshape(-1,1)
Xs6_test = Xs6[-20:].array.to_numpy().reshape(-1,1)
ys6_training = y[:-20]
ys6_test = y[-20:].array.to_numpy().reshape(-1,1)

regrs6 = linear_model.LinearRegression()
regrs6.fit(Xs6_training, ys6_training)
diabetes_ys6_pred = regrs6.predict(Xs6_test)
scores6 = regrs6.score(Xs6_test,ys6_test)
maes6 =sum(abs(ys6_test - diabetes_ys6_pred.reshape(-1,1)))/ys6_test.size
rmses6 = sqrt(mean_squared_error(ys6_test,diabetes_ys6_pred.reshape(-1,1)))
slopes6 = regrs6.coef_
intercepts6 = regrs6.intercept_

# age and body mass index

Xab = table[['age','bmi']]
Xab_training = Xab[:-20].to_numpy().reshape(-1,2)
Xab_test = Xab[-20:].to_numpy().reshape(-1,2)
yab_training = y[:-20]
yab_test = y[-20:].to_numpy().reshape(-1,1)

regrAB = linear_model.LinearRegression()
regrAB.fit(Xab_training, yab_training)
diabetes_yab_pred = regrAB.predict(Xab_test)
scoreAB = regrAB.score(Xab_test,yab_test)
maeAB = sum(abs(yab_test - diabetes_yab_pred.reshape(-1,1)))/yab_test.size
rmseAB =sqrt(mean_squared_error(yab_test,diabetes_yab_pred.reshape(-1,1)))
slopeAB = regrAB.coef_
interceptAB = regrAB.intercept_

# bmi and bp

Xbb = table[['bmi','bp']]
Xbb_training = Xbb[:-20].to_numpy().reshape(-1,2)
Xbb_test = Xbb[-20:].to_numpy().reshape(-1,2)
ybb_training = y[:-20]
ybb_test = y[-20:].to_numpy().reshape(-1,1)

regrBB = linear_model.LinearRegression()
regrBB.fit(Xbb_training, ybb_training)
diabetes_ybb_pred = regrBB.predict(Xbb_test)
scoreBB = regrBB.score(Xbb_test,ybb_test)
maeBB = sum(abs(ybb_test - diabetes_ybb_pred.reshape(-1,1)))/ybb_test.size
rmseBB =sqrt(mean_squared_error(ybb_test,diabetes_ybb_pred.reshape(-1,1)))
slopeBB = regrBB.coef_
interceptBB = regrBB.intercept_

# bmi and s5

Xb5 = table[['bmi','s5']]
Xb5_training = Xb5[:-20].to_numpy().reshape(-1,2)
Xb5_test = Xb5[-20:].to_numpy().reshape(-1,2)
yb5_training = y[:-20]
yb5_test = y[-20:].to_numpy().reshape(-1,1)

regrb5 = linear_model.LinearRegression()
regrb5.fit(Xb5_training, yb5_training)
diabetes_yb5_pred = regrb5.predict(Xb5_test)
scoreb5 = regrb5.score(Xb5_test,yb5_test)
maeb5 = sum(abs(yb5_test - diabetes_yb5_pred.reshape(-1,1)))/yb5_test.size
rmseb5 =sqrt(mean_squared_error(yb5_test,diabetes_yb5_pred.reshape(-1,1)))
slopeb5 = regrb5.coef_
interceptb5 = regrb5.intercept_

# bmi and s3

Xb3 = table[['bmi','s3']]
Xb3_training = Xb3[:-20].to_numpy().reshape(-1,2)
Xb3_test = Xb3[-20:].to_numpy().reshape(-1,2)
yb3_training = y[:-20]
yb3_test = y[-20:].to_numpy().reshape(-1,1)

regrb3 = linear_model.LinearRegression()
regrb3.fit(Xb3_training, yb3_training)
diabetes_yb3_pred = regrb3.predict(Xb3_test)
scoreb3 = regrb3.score(Xb3_test,yb3_test)
maeb3 = sum(abs(yb3_test - diabetes_yb3_pred.reshape(-1,1)))/yb3_test.size
rmseb3 =sqrt(mean_squared_error(yb3_test,diabetes_yb3_pred.reshape(-1,1)))
slopeb3 = regrb3.coef_
interceptb3 = regrb3.intercept_

# not using due to having a need for 4d plotting
"""
# bmi, bp and s5

Xbb5 = table[['bmi','bp','s5']]
Xbb5_training = Xbb5[:-20].to_numpy().reshape(-1,3)
Xbb5_test = Xbb5[-20:].to_numpy().reshape(-1,3)
ybb5_training = y[:-20]
ybb5_test = y[-20:].to_numpy().reshape(-1,1)

regrbb5 = linear_model.LinearRegression()
regrbb5.fit(Xbb5_training, ybb5_training)
diabetes_ybb5_pred = regrbb5.predict(Xbb5_test)
scorebb5 = regrbb5.score(Xbb5_test,ybb5_test)
maebb5 = sum(abs(ybb5_test - diabetes_ybb5_pred.reshape(-1,1)))/ybb5_test.size
rmsebb5 =sqrt(mean_squared_error(ybb5_test,diabetes_ybb5_pred.reshape(-1,1)))
slopebb5 = regrbb5.coef_
interceptbb5 = regrbb5.intercept_
#
print('BMI, Blood Pressure and s5: \nR2: ', scorebb5, '\nMean Absolute Error: ',
        maebb5,'\nRoot Mean Square Error: ', rmsebb5)
myDistribution(Xbb5, 'BMI, Blood Pressure and s5')
myResiduals(ybb5_test,diabetes_ybb5_pred, 'BMI, Blood Pressure and s5')
#

# bmi, s3 and s5

Xb35 = table[['bmi','s3','s5']]
Xb35_training = Xb35[:-20].to_numpy().reshape(-1,3)
Xb35_test = Xb35[-20:].to_numpy().reshape(-1,3)
yb35_training = y[:-20]
yb35_test = y[-20:].to_numpy().reshape(-1,1)

regrb35 = linear_model.LinearRegression()
regrb35.fit(Xb35_training, yb35_training)
diabetes_yb35_pred = regrb35.predict(Xb35_test)
scoreb35 = regrb35.score(Xb35_test,yb35_test)
maeb35 = sum(abs(yb35_test - diabetes_yb35_pred.reshape(-1,1)))/yb35_test.size
rmseb35 =sqrt(mean_squared_error(yb35_test,diabetes_yb35_pred.reshape(-1,1)))
slopeb35 = regrb35.coef_
interceptb35 = regrb35.intercept_
#
print('BMI, s3 and s5: \nR2: ', scoreb35, '\nMean Absolute Error: ',
        maeb35,'\nRoot Mean Square Error: ', rmseb35)
myDistribution(Xb35, 'BMI, s3 and s5')
myResiduals(yb35_test,diabetes_yb35_pred, 'BMI, s3 and s5')
#
"""

# plotting ***************************************************************************************************

# age

my2dPlot(Xa_test,ya_test,diabetes_ya_pred,'Age',scoreA,maeA,rmseA)
myDistribution(Xa,'Age')
myResiduals(ya_test,diabetes_ya_pred,'Age')

# sex 

my2dPlot(Xs_test,ys_test,diabetes_ys_pred,'Sex',scoreS,maeS,rmseS)
myDistribution(Xs,'Sex')
myResiduals(ys_test,diabetes_ys_pred,'Sex')

# bmi 

my2dPlot(Xb_test,yb_test,diabetes_yb_pred,'Body Mass Index',scoreB,maeB,rmseB)
myDistribution(Xb,'Body Mass Index')
myResiduals(yb_test,diabetes_yb_pred,'Body Mass Index')

# bp 

my2dPlot(Xbp_test,ybp_test,diabetes_ybp_pred,'Blood Pressure',scoreBP,maeBP,rmseBP)
myDistribution(Xbp,'Blood Pressure')
myResiduals(ybp_test,diabetes_ybp_pred,'Blood Pressure')

# s1 

my2dPlot(Xs1_test,ys1_test,diabetes_ys1_pred,'s1',scores1,maes1,rmses1)
myDistribution(Xs1,'s1')
myResiduals(ys1_test,diabetes_ys1_pred,'s1')

# s2 

my2dPlot(Xs2_test,ys2_test,diabetes_ys2_pred,'s2',scores2,maes2,rmses2)
myDistribution(Xs2,'s2')
myResiduals(ys2_test,diabetes_ys2_pred,'s2')

# s3 

my2dPlot(Xs3_test,ys3_test,diabetes_ys3_pred,'s3',scores3,maes3,rmses3)
myDistribution(Xs3,'s3')
myResiduals(ys3_test,diabetes_ys3_pred,'s3')

# s4 

my2dPlot(Xs4_test,ys4_test,diabetes_ys4_pred,'s4',scores4,maes4,rmses4)
myDistribution(Xs4,'s4')
myResiduals(ys4_test,diabetes_ys4_pred,'s4')

# s5 

my2dPlot(Xs5_test,ys5_test,diabetes_ys5_pred,'s5',scores5,maes5,rmses5)
myDistribution(Xs5,'s5')
myResiduals(ys5_test,diabetes_ys5_pred,'s5')

# s6 

my2dPlot(Xs6_test,ys6_test,diabetes_ys6_pred,'s6',scores6,maes6,rmses6)
myDistribution(Xs6,'s6')
myResiduals(ys6_test,diabetes_ys6_pred,'s6')

# bmi, s5 and bp ***4d, do not plot
"""
plt.plot(Xbb5_test, diabetes_ybb5_pred, color='green', linewidth=3)
plt.title('BMI-s5-BP vs Diabetes', fontsize=14)
plt.xlabel('Body Mass Index(BMI)-s5-Blood Pressure(BP)', fontsize=14)
plt.ylabel('Diabetes', fontsize=14)
plt.grid(True)
print('\n\nBody Mass Index, s5 and Blood Pressure: \nR2: ', scorebb5, '\nMean Absolute Error: ', 
      maebb5,'\nRoot Mean Square Error: ', rmsebb5)
plt.show()
"""
# age and bmi 3d plot

my3dPlot(Xab_test,yab_test,regrAB,scoreAB,maeAB,rmseAB,'Age','Body Mass Index')
myDistribution2(Xab,'Age','Body Mass Index')
myResiduals2(yab_test,diabetes_yab_pred,'Age','Body Mass Index')

# bmi and bp 3d plot

my3dPlot(Xbb_test,ybb_test,regrBB,scoreBB,maeBB,rmseBB,'Body Mass Index','Blood Pressure')
myDistribution2(Xbb,'Body Mass Index','Blood Pressure')
myResiduals2(ybb_test,diabetes_ybb_pred,'Body Mass Index','Blood Pressure')

# bmi and s3 3d plot

my3dPlot(Xb3_test,yb3_test,regrb3,scoreb3,maeb3,rmseb3,'Body Mass Index','s3')
myDistribution2(Xb3,'Body Mass Index','s3')
myResiduals2(yb3_test,diabetes_yb3_pred,'Body Mass Index','s3')

# bmi and s5 3d plot

my3dPlot(Xb5_test,yb5_test,regrb5,scoreb5,maeb5,rmseb5,'Body Mass Index','s5')
myDistribution2(Xb5,'Body Mass Index','s5')
myResiduals2(yb5_test,diabetes_yb5_pred,'Body Mass Index','s5')

# plotting end ***********************************************************************************************


# bmi,s5 e bp seem to be the best indicators
    # bmi combined with s5 seems like the best option
# assuming causality of the correlation whereas it might not be
# couldnt find s1-s6 definitions, seems to be 'tc ldl hdl tch ltg glu', but they are not all discernible
# score = r^2, if <0 or >1 , might be wrong model or bad data
    # the closer to 1, the least the residue
        # if r^2 = 1, it's probably for the worst