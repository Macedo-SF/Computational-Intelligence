import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from math import sqrt
from sklearn import linear_model
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
#linha teste
# Load the diabetes dataset
diabetes = load_diabetes()
# diabetes
tabela = pd.DataFrame(diabetes.data)
tabela.columns = diabetes.feature_names
tabela['y'] = diabetes.target
y = tabela['y']
# importante ********************************************************
#print(sum(y)/442) # posso pegar a média, assumir que é diabetes, dividir o erro por ela e dar uma acurácia de 71 a 75%?
# importante ********************************************************
#print(tabela.head()) # not enough space in screen to see all columns
# splice data
# exemplo
    #separa coluna de dados
    #separa dados de treinamento do modelo(independente e dependente)
    #separa dados de teste do modelo(independente e dependente)
    #modelo de regressão
    #regressão
    #predição
    #score
    #mean absolute error
    #root mean squared error
    #y=Slope*x+Intercept
# fim exemplo
# dados *******************************************************************************************************
# age
Xa = tabela['age']
Xa_treinamento = Xa[:-20].array.to_numpy().reshape(-1,1)
Xa_teste = Xa[-20:].array.to_numpy().reshape(-1,1)
ya_treinamento = y[:-20]
ya_teste = y[-20:].array.to_numpy().reshape(-1,1)

regrA = linear_model.LinearRegression()
regrA.fit(Xa_treinamento, ya_treinamento)
diabetes_ya_pred = regrA.predict(Xa_teste)
scoreA = regrA.score(Xa_teste,ya_teste)
maeA = sum(abs(ya_teste - diabetes_ya_pred.reshape(-1,1)))/ya_teste.size
rmseA = sqrt(mean_squared_error(ya_teste,diabetes_ya_pred.reshape(-1,1)))
slopeA = regrA.coef_
interceptA = regrA.intercept_
# sex
Xs = tabela['sex']
Xs_treinamento = Xs[:-20].array.to_numpy().reshape(-1,1)
Xs_teste = Xs[-20:].array.to_numpy().reshape(-1,1)
ys_treinamento = y[:-20]
ys_teste = y[-20:].array.to_numpy().reshape(-1,1)

regrS = linear_model.LinearRegression()
regrS.fit(Xs_treinamento, ys_treinamento)
diabetes_ys_pred = regrS.predict(Xs_teste)
scoreS = regrS.score(Xs_teste,ys_teste)
maeS =sum(abs(ys_teste - diabetes_ys_pred.reshape(-1,1)))/ys_teste.size
rmseS = sqrt(mean_squared_error(ys_teste,diabetes_ys_pred.reshape(-1,1)))
slopeS = regrS.coef_
interceptS = regrS.intercept_
# body mass index
Xb = tabela['bmi']
Xb_treinamento = Xb[:-20].array.to_numpy().reshape(-1,1)
Xb_teste = Xb[-20:].array.to_numpy().reshape(-1,1)
yb_treinamento = y[:-20]
yb_teste = y[-20:].array.to_numpy().reshape(-1,1)

regrB = linear_model.LinearRegression()
regrB.fit(Xb_treinamento, yb_treinamento)
diabetes_yb_pred = regrB.predict(Xb_teste)
scoreB = regrB.score(Xb_teste,yb_teste)
maeB = sum(abs(yb_teste - diabetes_yb_pred.reshape(-1,1)))/yb_teste.size
rmseB = sqrt(mean_squared_error(yb_teste,diabetes_yb_pred.reshape(-1,1)))
slopeB = regrB.coef_
interceptB = regrB.intercept_
# blood pressure
Xbp = tabela['bp']
Xbp_treinamento = Xbp[:-20].array.to_numpy().reshape(-1,1)
Xbp_teste = Xbp[-20:].array.to_numpy().reshape(-1,1)
ybp_treinamento = y[:-20]
ybp_teste = y[-20:].array.to_numpy().reshape(-1,1)

regrBP = linear_model.LinearRegression()
regrBP.fit(Xbp_treinamento, ybp_treinamento)
diabetes_ybp_pred = regrBP.predict(Xbp_teste)
scoreBP = regrBP.score(Xbp_teste,ybp_teste)
maeBP = sum(abs(ybp_teste - diabetes_ybp_pred.reshape(-1,1)))/ybp_teste.size
rmseBP = sqrt(mean_squared_error(ybp_teste,diabetes_ybp_pred.reshape(-1,1)))
slopeBP = regrBP.coef_
interceptBP = regrBP.intercept_
# s1
Xs1 = tabela['s1']
Xs1_treinamento = Xs1[:-20].array.to_numpy().reshape(-1,1)
Xs1_teste = Xs1[-20:].array.to_numpy().reshape(-1,1)
ys1_treinamento = y[:-20]
ys1_teste = y[-20:].array.to_numpy().reshape(-1,1)

regrS1 = linear_model.LinearRegression()
regrS1.fit(Xs1_treinamento, ys1_treinamento)
diabetes_ys1_pred = regrS1.predict(Xs1_teste)
scoreS1 = regrS1.score(Xs1_teste,ys1_teste)
maeS1 =sum(abs(ys1_teste - diabetes_ys1_pred.reshape(-1,1)))/ys1_teste.size
rmseS1 = sqrt(mean_squared_error(ys1_teste,diabetes_ys1_pred.reshape(-1,1)))
slopeS1 = regrS1.coef_
interceptS1 = regrS1.intercept_
# s2
Xs2 = tabela['s2']
Xs2_treinamento = Xs2[:-20].array.to_numpy().reshape(-1,1)
Xs2_teste = Xs2[-20:].array.to_numpy().reshape(-1,1)
ys2_treinamento = y[:-20]
ys2_teste = y[-20:].array.to_numpy().reshape(-1,1)

regrS2 = linear_model.LinearRegression()
regrS2.fit(Xs2_treinamento, ys2_treinamento)
diabetes_ys2_pred = regrS2.predict(Xs2_teste)
scoreS2 = regrS2.score(Xs2_teste,ys2_teste)
maeS2 =sum(abs(ys2_teste - diabetes_ys2_pred.reshape(-1,1)))/ys2_teste.size
rmseS2 = sqrt(mean_squared_error(ys2_teste,diabetes_ys2_pred.reshape(-1,1)))
slopeS2 = regrS2.coef_
interceptS2 = regrS2.intercept_
# s3
Xs3 = tabela['s3']
Xs3_treinamento = Xs3[:-20].array.to_numpy().reshape(-1,1)
Xs3_teste = Xs3[-20:].array.to_numpy().reshape(-1,1)
ys3_treinamento = y[:-20]
ys3_teste = y[-20:].array.to_numpy().reshape(-1,1)

regrs3 = linear_model.LinearRegression()
regrs3.fit(Xs3_treinamento, ys3_treinamento)
diabetes_ys3_pred = regrs3.predict(Xs3_teste)
scores3 = regrs3.score(Xs3_teste,ys3_teste)
maes3 =sum(abs(ys3_teste - diabetes_ys3_pred.reshape(-1,1)))/ys3_teste.size
rmses3 = sqrt(mean_squared_error(ys3_teste,diabetes_ys3_pred.reshape(-1,1)))
slopes3 = regrs3.coef_
intercepts3 = regrs3.intercept_
# s4
Xs4 = tabela['s4']
Xs4_treinamento = Xs4[:-20].array.to_numpy().reshape(-1,1)
Xs4_teste = Xs4[-20:].array.to_numpy().reshape(-1,1)
ys4_treinamento = y[:-20]
ys4_teste = y[-20:].array.to_numpy().reshape(-1,1)

regrs4 = linear_model.LinearRegression()
regrs4.fit(Xs4_treinamento, ys4_treinamento)
diabetes_ys4_pred = regrs4.predict(Xs4_teste)
scores4 = regrs4.score(Xs4_teste,ys4_teste)
maes4 =sum(abs(ys4_teste - diabetes_ys4_pred.reshape(-1,1)))/ys4_teste.size
rmses4 = sqrt(mean_squared_error(ys4_teste,diabetes_ys4_pred.reshape(-1,1)))
slopes4 = regrs4.coef_
intercepts4 = regrs4.intercept_
# s5
Xs5 = tabela['s5']
Xs5_treinamento = Xs5[:-20].array.to_numpy().reshape(-1,1)
Xs5_teste = Xs5[-20:].array.to_numpy().reshape(-1,1)
ys5_treinamento = y[:-20]
ys5_teste = y[-20:].array.to_numpy().reshape(-1,1)

regrs5 = linear_model.LinearRegression()
regrs5.fit(Xs5_treinamento, ys5_treinamento)
diabetes_ys5_pred = regrs5.predict(Xs5_teste)
scores5 = regrs5.score(Xs5_teste,ys5_teste)
maes5 =sum(abs(ys5_teste - diabetes_ys5_pred.reshape(-1,1)))/ys5_teste.size
rmses5 = sqrt(mean_squared_error(ys5_teste,diabetes_ys5_pred.reshape(-1,1)))
slopes5 = regrs5.coef_
intercepts5 = regrs5.intercept_
# s6
Xs6 = tabela['s6']
Xs6_treinamento = Xs6[:-20].array.to_numpy().reshape(-1,1)
Xs6_teste = Xs6[-20:].array.to_numpy().reshape(-1,1)
ys6_treinamento = y[:-20]
ys6_teste = y[-20:].array.to_numpy().reshape(-1,1)

regrs6 = linear_model.LinearRegression()
regrs6.fit(Xs6_treinamento, ys6_treinamento)
diabetes_ys6_pred = regrs6.predict(Xs6_teste)
scores6 = regrs6.score(Xs6_teste,ys6_teste)
maes6 =sum(abs(ys6_teste - diabetes_ys6_pred.reshape(-1,1)))/ys6_teste.size
rmses6 = sqrt(mean_squared_error(ys6_teste,diabetes_ys6_pred.reshape(-1,1)))
slopes6 = regrs6.coef_
intercepts6 = regrs6.intercept_
# age and body mass index
Xab = tabela[['age','bmi']]
Xab_treinamento = Xab[:-20].to_numpy().reshape(-1,2)
Xab_teste = Xab[-20:].to_numpy().reshape(-1,2)
yab_treinamento = y[:-20]
yab_teste = y[-20:].to_numpy().reshape(-1,1)

regrAB = linear_model.LinearRegression()
regrAB.fit(Xab_treinamento, yab_treinamento)
diabetes_yab_pred = regrAB.predict(Xab_teste)
scoreAB = regrAB.score(Xab_teste,yab_teste)
maeAB = sum(abs(yab_teste - diabetes_yab_pred.reshape(-1,1)))/yab_teste.size
rmseAB =sqrt(mean_squared_error(yab_teste,diabetes_yab_pred.reshape(-1,1)))
slopeAB = regrAB.coef_
interceptAB = regrAB.intercept_
# bmi and bp
Xbb = tabela[['bmi','bp']]
Xbb_treinamento = Xbb[:-20].to_numpy().reshape(-1,2)
Xbb_teste = Xbb[-20:].to_numpy().reshape(-1,2)
ybb_treinamento = y[:-20]
ybb_teste = y[-20:].to_numpy().reshape(-1,1)

regrBB = linear_model.LinearRegression()
regrBB.fit(Xbb_treinamento, ybb_treinamento)
diabetes_ybb_pred = regrBB.predict(Xbb_teste)
scoreBB = regrBB.score(Xbb_teste,ybb_teste)
maeBB = sum(abs(ybb_teste - diabetes_ybb_pred.reshape(-1,1)))/ybb_teste.size
rmseBB =sqrt(mean_squared_error(ybb_teste,diabetes_ybb_pred.reshape(-1,1)))
slopeBB = regrBB.coef_
interceptBB = regrBB.intercept_
# bmi and s5
Xb5 = tabela[['bmi','s5']]
Xb5_treinamento = Xb5[:-20].to_numpy().reshape(-1,2)
Xb5_teste = Xb5[-20:].to_numpy().reshape(-1,2)
yb5_treinamento = y[:-20]
yb5_teste = y[-20:].to_numpy().reshape(-1,1)

regrb5 = linear_model.LinearRegression()
regrb5.fit(Xb5_treinamento, yb5_treinamento)
diabetes_yb5_pred = regrb5.predict(Xb5_teste)
scoreb5 = regrb5.score(Xb5_teste,yb5_teste)
maeb5 = sum(abs(yb5_teste - diabetes_yb5_pred.reshape(-1,1)))/yb5_teste.size
rmseb5 =sqrt(mean_squared_error(yb5_teste,diabetes_yb5_pred.reshape(-1,1)))
slopeb5 = regrb5.coef_
interceptb5 = regrb5.intercept_
# bmi and s3
Xb3 = tabela[['bmi','s3']]
Xb3_treinamento = Xb3[:-20].to_numpy().reshape(-1,2)
Xb3_teste = Xb3[-20:].to_numpy().reshape(-1,2)
yb3_treinamento = y[:-20]
yb3_teste = y[-20:].to_numpy().reshape(-1,1)

regrb3 = linear_model.LinearRegression()
regrb3.fit(Xb3_treinamento, yb3_treinamento)
diabetes_yb3_pred = regrb3.predict(Xb3_teste)
scoreb3 = regrb3.score(Xb3_teste,yb3_teste)
maeb3 = sum(abs(yb3_teste - diabetes_yb3_pred.reshape(-1,1)))/yb3_teste.size
rmseb3 =sqrt(mean_squared_error(yb3_teste,diabetes_yb3_pred.reshape(-1,1)))
slopeb3 = regrb3.coef_
interceptb3 = regrb3.intercept_
# bmi, bp and s5
Xbb5 = tabela[['bmi','bp','s5']]
Xbb5_treinamento = Xbb5[:-20].to_numpy().reshape(-1,3)
Xbb5_teste = Xbb5[-20:].to_numpy().reshape(-1,3)
ybb5_treinamento = y[:-20]
ybb5_teste = y[-20:].to_numpy().reshape(-1,1)

regrbb5 = linear_model.LinearRegression()
regrbb5.fit(Xbb5_treinamento, ybb5_treinamento)
diabetes_ybb5_pred = regrbb5.predict(Xbb5_teste)
scorebb5 = regrbb5.score(Xbb5_teste,ybb5_teste)
maebb5 = sum(abs(ybb5_teste - diabetes_ybb5_pred.reshape(-1,1)))/ybb5_teste.size
rmsebb5 =sqrt(mean_squared_error(ybb5_teste,diabetes_ybb5_pred.reshape(-1,1)))
slopebb5 = regrbb5.coef_
interceptbb5 = regrbb5.intercept_

# plotagem ***************************************************************************************************

# age
plt.scatter(Xa_teste,ya_teste,  color='red')
plt.plot(Xa_teste, diabetes_ya_pred, color='pink', linewidth=3)
plt.title('Age vs Diabetes', fontsize=14)
plt.xlabel('Age', fontsize=14)
plt.ylabel('Diabetes', fontsize=14)
plt.grid(True)
print('Age: \nR2: ', scoreA, '\nMean Absolute Error: ', maeA,'\nRoot Mean Square Error: ', rmseA)
plt.show()
# sex 
plt.scatter(Xs_teste,ys_teste,  color='orange')
plt.plot(Xs_teste, diabetes_ys_pred, color='yellow', linewidth=3)
plt.title('Sex vs Diabetes', fontsize=14)
plt.xlabel('Sex', fontsize=14)
plt.ylabel('Diabetes', fontsize=14)
plt.grid(True)
print('\n\nSex: \nR2: ', scoreS, '\nMean Absolute Erro: ', maeS,'\nRoot Mean Square Error: ', rmseS)
plt.show() 
# bmi 
plt.scatter(Xb_teste,yb_teste,  color='gray')
plt.plot(Xb_teste, diabetes_yb_pred, color='black', linewidth=3)
plt.title('Body Mass Index vs Diabetes', fontsize=14)
plt.xlabel('Body Mass Index(BMI)', fontsize=14)
plt.ylabel('Diabetes', fontsize=14)
plt.grid(True)
print('\n\nBody Mass Index: \nR2: ', scoreB, '\nMean Absolute Error: ', maeB,'\nRoot Mean Square Error: ', rmseB)
plt.show()
# bp 
plt.scatter(Xbp_teste,ybp_teste,  color='gray')
plt.plot(Xbp_teste, diabetes_ybp_pred, color='black', linewidth=3)
plt.title('Blood Pressure vs Diabetes', fontsize=14)
plt.xlabel('Blood Pressure', fontsize=14)
plt.ylabel('Diabetes', fontsize=14)
plt.grid(True)
print('\n\nBlood Pressure: \nR2: ', scoreBP, '\nMean Absolute Error: ', maeBP,'\nRoot Mean Square Error: ', rmseBP)
plt.show()
# s1 
plt.scatter(Xs1_teste,ys1_teste,  color='orange')
plt.plot(Xs1_teste, diabetes_ys1_pred, color='yellow', linewidth=3)
plt.title('S1 vs Diabetes', fontsize=14)
plt.xlabel('S1', fontsize=14)
plt.ylabel('Diabetes', fontsize=14)
plt.grid(True)
print('\n\nS1: \nR2: ', scoreS1, '\nMean Absolute Error: ', maeS1,'\nRoot Mean Square Error: ', rmseS1)
plt.show()
# s2 
plt.scatter(Xs2_teste,ys2_teste,  color='orange')
plt.plot(Xs2_teste, diabetes_ys2_pred, color='yellow', linewidth=3)
plt.title('S2 vs Diabetes', fontsize=14)
plt.xlabel('S2', fontsize=14)
plt.ylabel('Diabetes', fontsize=14)
plt.grid(True)
print('\n\nS2: \nR2: ', scoreS2, '\nMean Absolute Error: ', maeS2,'\nRoot Mean Square Error: ', rmseS2)
plt.show()
# s3 
plt.scatter(Xs3_teste,ys3_teste,  color='orange')
plt.plot(Xs3_teste, diabetes_ys3_pred, color='yellow', linewidth=3)
plt.title('s3 vs Diabetes', fontsize=14)
plt.xlabel('s3', fontsize=14)
plt.ylabel('Diabetes', fontsize=14)
plt.grid(True)
print('\n\ns3: \nR2: ', scores3, '\nMean Absolute Error: ', maes3,'\nRoot Mean Square Error: ', rmses3)
plt.show()
# s4 
plt.scatter(Xs4_teste,ys4_teste,  color='orange')
plt.plot(Xs4_teste, diabetes_ys4_pred, color='yellow', linewidth=3)
plt.title('s4 vs Diabetes', fontsize=14)
plt.xlabel('s4', fontsize=14)
plt.ylabel('Diabetes', fontsize=14)
plt.grid(True)
print('\n\ns4: \nR2: ', scores4, '\nMean Absolute Error: ', maes4,'\nRoot Mean Square Error: ', rmses4)
plt.show()
# s5 
plt.scatter(Xs5_teste,ys5_teste,  color='orange')
plt.plot(Xs5_teste, diabetes_ys5_pred, color='yellow', linewidth=3)
plt.title('s5 vs Diabetes', fontsize=14)
plt.xlabel('s5', fontsize=14)
plt.ylabel('Diabetes', fontsize=14)
plt.grid(True)
print('\n\ns5: \nR2: ', scores5, '\nMean Absolute Error: ', maes5,'\nRoot Mean Square Error: ', rmses5)
plt.show()
# s6 
plt.scatter(Xs6_teste,ys6_teste,  color='orange')
plt.plot(Xs6_teste, diabetes_ys6_pred, color='yellow', linewidth=3)
plt.title('s6 vs Diabetes', fontsize=14)
plt.xlabel('s6', fontsize=14)
plt.ylabel('Diabetes', fontsize=14)
plt.grid(True)
print('\n\ns6: \nR2: ', scores6, '\nMean Absolute Error: ', maes6,'\nRoot Mean Square Error: ', rmses6)
plt.show()
# bmi, s5 and bp ***4d, nem rola
plt.plot(Xbb5_teste, diabetes_ybb5_pred, color='green', linewidth=3)
plt.title('BMI-s5-BP vs Diabetes', fontsize=14)
plt.xlabel('Body Mass Index(BMI)-s5-Blood Pressure(BP)', fontsize=14)
plt.ylabel('Diabetes', fontsize=14)
plt.grid(True)
print('\n\nBody Mass Index, s5 and Blood Pressure: \nR2: ', scorebb5, '\nMean Absolute Error: ', 
      maebb5,'\nRoot Mean Square Error: ', rmsebb5)
plt.show()

# age and bmi 3d plot

x1=Xab_teste[:, 0]
y1=Xab_teste[:, 1]
z1=pd.Series(yab_teste.flatten())
x2, y2 =np.meshgrid(x1, y1)
model_viz = np.array([x2.flatten(), y2.flatten()]).T
predicted = regrAB.predict(model_viz)
#usados na plotagem

plt.style.use('default')

fig = plt.figure(figsize=(12, 4))

ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')

axes = [ax1, ax2, ax3]

for ax in axes:
    ax.plot(x1, y1, z1, color='k', zorder=15, linestyle='none', marker='o', alpha=0.5)
    ax.scatter(x2, y2, predicted, facecolor=(0,0,0,0), s=20, edgecolor='#70b3f0')
    ax.set_xlabel('Age', fontsize=12)
    ax.set_ylabel('Body Mass Index', fontsize=12)
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

fig.suptitle('$R^2 = %.5f$' % scoreAB, fontsize=20)

fig.tight_layout()
print('\n\nAge and Body Mass Index: \nR2: ', scoreAB, '\nMean Absolute Error: ',
     maeAB,'\nRoot Mean Square Error: ', rmseAB)
plt.show()

# bmi and bp 3d plot

x1=Xbb_teste[:, 0]
y1=Xbb_teste[:, 1]
z1=pd.Series(ybb_teste.flatten())
x2, y2 =np.meshgrid(x1, y1)
model_viz = np.array([x2.flatten(), y2.flatten()]).T
predicted = regrBB.predict(model_viz)
#usados na plotagem

plt.style.use('default')

fig = plt.figure(figsize=(12, 4))

ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')

axes = [ax1, ax2, ax3]

for ax in axes:
    ax.plot(x1, y1, z1, color='k', zorder=15, linestyle='none', marker='o', alpha=0.5)
    ax.scatter(x2, y2, predicted, facecolor=(0,0,0,0), s=20, edgecolor='#70b3f0')
    ax.set_xlabel('Body Mass Index', fontsize=12)
    ax.set_ylabel('Blood Pressure', fontsize=12)
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

fig.suptitle('$R^2 = %.5f$' % scoreBB, fontsize=20)

fig.tight_layout()
print('\n\nBody Mass Index and Blood Pressure: \nR2: ', scoreBB, '\nMean Absolute Error: ',
     maeBB,'\nRoot Mean Square Error: ', rmseBB)
plt.show()

# bmi and s5 3d plot

x1=Xb5_teste[:, 0]
y1=Xb5_teste[:, 1]
z1=pd.Series(yb5_teste.flatten())
x2, y2 =np.meshgrid(x1, y1)
model_viz = np.array([x2.flatten(), y2.flatten()]).T
predicted = regrb5.predict(model_viz)
#usados na plotagem

plt.style.use('default')

fig = plt.figure(figsize=(12, 4))

ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')

axes = [ax1, ax2, ax3]

for ax in axes:
    ax.plot(x1, y1, z1, color='k', zorder=15, linestyle='none', marker='o', alpha=0.5)
    ax.scatter(x2, y2, predicted, facecolor=(0,0,0,0), s=20, edgecolor='#70b3f0')
    ax.set_xlabel('Body Mass Index', fontsize=12)
    ax.set_ylabel('s5', fontsize=12)
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

fig.suptitle('$R^2 = %.5f$' % scoreb5, fontsize=20)

fig.tight_layout()
print('\n\nBody Mass Index and s5: \nR2: ', scoreb5, '\nMean Absolute Error: ',
     maeb5,'\nRoot Mean Square Error: ', rmseb5)
plt.show()
# bmi and s3 3d plot

x1=Xb3_teste[:, 0]
y1=Xb3_teste[:, 1]
z1=pd.Series(yb3_teste.flatten())
x2, y2 =np.meshgrid(x1, y1)
model_viz = np.array([x2.flatten(), y2.flatten()]).T
predicted = regrb3.predict(model_viz)
#usados na plotagem

plt.style.use('default')

fig = plt.figure(figsize=(12, 4))

ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')

axes = [ax1, ax2, ax3]

for ax in axes:
    ax.plot(x1, y1, z1, color='k', zorder=15, linestyle='none', marker='o', alpha=0.5)
    ax.scatter(x2, y2, predicted, facecolor=(0,0,0,0), s=20, edgecolor='#70b3f0')
    ax.set_xlabel('Body Mass Index', fontsize=12)
    ax.set_ylabel('s3', fontsize=12)
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

fig.suptitle('$R^2 = %.5f$' % scoreb5, fontsize=20)

fig.tight_layout()
print('\n\nBody Mass Index and s3: \nR2: ', scoreb3, '\nMean Absolute Error: ',
     maeb3,'\nRoot Mean Square Error: ', rmseb3)
plt.show()
# fim plotagem ***********************************************************************************************


# bmi,s5 e bp são os melhores indicadores
    # bmi combinado com s5 traz o melhor resultado
# não posso afirmar que existe uma relação linear dos dados, não é minha area
    #correlação não implica causalidade
# não achei s1-s6, parece ser tc ldl hdl tch ltg glu, mas nem todos são discerniveis
# erro medio absoluto e raiz do mse
# score = r^2, se fora de 0:1, pode ser modelo errado
    # quanto mais proximo de 1, menor o resíduo
        # se for r2=1, algo tá errado