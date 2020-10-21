#iris dataset: contains three different iris petals
#Setosa, Versicolour and Virginica 
#lenghts and widths of sepals and petals (150x4)

import sys
import pandas as pd
import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from collections import Counter

# fig directory
# comment plt.close() and uncomment plt.show() to switch

my_path = 'C:/Users/Saulo/source/repos/IC/Classification' # set ur path in order to save the figures or do as stated above and use show()

# functions

#data scatter //use pcolormesh to make it beautiful, maybe?
def scatter(y,data1,data2,name1,name2):

    #dataset to n-array
    data = pd.concat([data1, data2], axis=1, sort=False)
    #print(data.head(150))
    data=data.to_numpy()
    #2d plot
    plt.subplots()
    plt.scatter(data[:, 0], data[:, 1], c=y, cmap=plt.cm.Set1)
    plt.xlabel(name1)
    plt.ylabel(name2)
    plt.grid(True)
    plt.savefig(my_path+'/Figures/'+name1+'_'+name2+'_Scatter'+'.png')
    plt.close()

#multi-layer perceptron
def mlpAnalysis(y,data1,data2,name1,name2):

    #dataset to n-array
    data = pd.concat([data1, data2], axis=1, sort=False)
    #print(data.head(150))
    data=data.to_numpy()
    #split rows, training and test
    clf = MLPClassifier(alpha=0.01,max_iter=2000)
    yt=np.concatenate([y[:40], y[51:90], y[101:140]])
    xt = np.concatenate([data[:40,:], data[51:90,:], data[101:140,:]])
    yv=np.concatenate([y[40:50], y[90:100], y[140:150]])
    xv = np.concatenate([data[40:50,:], data[90:100,:], data[140:150,:]])
    #fit, predict, print
    clf.fit(xt, yt)
    yp=clf.predict(xv)
    print('____________________________________________________________\n')
    print(name1,' and ',name2,' Multi-Layer Perceptron:','\n____________________________________________________________\n')
    print('Prediction: \n\n',yp,'\n')
    print('Real: \n\n',yv,'\n____________________________________________________________\n')
    #mean accuracy, clf.score(xv,yv)
    comp = yp == yv
    c= Counter(comp)
    print('Accuracy: ',c[1]/(c[0]+c[1]),'\n')
    print(c,'\n____________________________________________________________\n')
    #confusion matrix
    titles_options = [("Confusion matrix", None), ("Confusion matrix, normalized", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(clf, xv, yv, cmap=plt.cm.Blues, normalize=normalize)
        disp.ax_.set_title(title)
        #print(title)
        #print(disp.confusion_matrix)
        plt.savefig(my_path+'/Figures/'+name1+'_'+name2+'_mlPerceptron_'+title+'.png')
        plt.close()
    #classification report
    print('Classification Report: \n\n',classification_report(yv, yp),'\n____________________________________________________________\n')

def mlpAnalysis3(y,data1,data2,data3,name1,name2,name3):

    #dataset to n-array
    data = pd.concat([data1, data2, data3], axis=1, sort=False)
    #print(data.head(150))
    data=data.to_numpy()
    #split rows, training and test
    clf = MLPClassifier(alpha=0.01,max_iter=2000)
    yt=np.concatenate([y[:40], y[51:90], y[101:140]])
    xt = np.concatenate([data[:40,:], data[51:90,:], data[101:140,:]])
    yv=np.concatenate([y[40:50], y[90:100], y[140:150]])
    xv = np.concatenate([data[40:50,:], data[90:100,:], data[140:150,:]])
    #fit, predict, print
    clf.fit(xt, yt)
    yp=clf.predict(xv)
    print('____________________________________________________________\n')
    print(name1,', ',name2,' and ',name3,' Multi-Layer Perceptron:','\n____________________________________________________________\n')
    print('Prediction: \n\n',yp,'\n')
    print('Real: \n\n',yv,'\n____________________________________________________________\n')
    #mean accuracy, clf.score(xv,yv)
    comp = yp == yv
    c= Counter(comp)
    print('Accuracy: ',c[1]/(c[0]+c[1]),'\n')
    print(c,'\n____________________________________________________________\n')
    #confusion matrix
    titles_options = [("Confusion matrix", None), ("Confusion matrix, normalized", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(clf, xv, yv, cmap=plt.cm.Blues, normalize=normalize)
        disp.ax_.set_title(title)
        #print(title)
        #print(disp.confusion_matrix)
        plt.savefig(my_path+'/Figures/'+name1+'_'+name2+'_'+name3+'_mlPerceptron_'+title+'.png')
        plt.close()
    #classification report
    print('Classification Report: \n\n',classification_report(yv, yp)
          ,'\n____________________________________________________________\n')

def mlpAnalysis4(y,data1,data2,data3,data4,name1,name2,name3,name4):

    #dataset to n-array
    data = pd.concat([data1, data2, data3, data4], axis=1, sort=False)
    #print(data.head(150))
    data=data.to_numpy()
    #split rows, training and test
    clf = MLPClassifier(alpha=0.01,max_iter=2000)
    yt=np.concatenate([y[:40], y[51:90], y[101:140]])
    xt = np.concatenate([data[:40,:], data[51:90,:], data[101:140,:]])
    yv=np.concatenate([y[40:50], y[90:100], y[140:150]])
    xv = np.concatenate([data[40:50,:], data[90:100,:], data[140:150,:]])
    #fit, predict, print
    clf.fit(xt, yt)
    yp=clf.predict(xv)
    print('____________________________________________________________\n')
    print(name1,', ',name2,', ',name3,' and ',name4,' Multi-Layer Perceptron:','\n____________________________________________________________\n')
    print('Prediction: \n\n',yp,'\n')
    print('Real: \n\n',yv,'\n____________________________________________________________\n')
    #mean accuracy, clf.score(xv,yv)
    comp = yp == yv
    c= Counter(comp)
    print('Accuracy: ',c[1]/(c[0]+c[1]),'\n')
    print(c,'\n____________________________________________________________\n')
    #confusion matrix
    titles_options = [("Confusion matrix", None), ("Confusion matrix, normalized", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(clf, xv, yv, cmap=plt.cm.Blues, normalize=normalize)
        disp.ax_.set_title(title)
        #print(title)
        #print(disp.confusion_matrix)
        plt.savefig(my_path+'/Figures/'+name1+'_'+name2+'_'+name3+'_'+name4+'_mlPerceptron_'+title+'.png')
        plt.close()
    #classification report
    print('Classification Report: \n\n',classification_report(yv, yp)
          ,'\n____________________________________________________________\n')

#k-neighbors //my approach was rather crude
def kneighAnalysis(y,data1,data2,name1,name2):

    #dataset to n-array
    data = pd.concat([data1, data2], axis=1, sort=False)
    #print(data.head(150))
    data=data.to_numpy()
    #split rows, training and test
    clf = KNeighborsClassifier()
    yt=np.concatenate([y[:40], y[51:90], y[101:140]])
    xt = np.concatenate([data[:40,:], data[51:90,:], data[101:140,:]])
    yv=np.concatenate([y[40:50], y[90:100], y[140:150]])
    xv = np.concatenate([data[40:50,:], data[90:100,:], data[140:150,:]])
    #fit, predict, print
    clf.fit(xt, yt)
    yp=clf.predict(xv)
    print('____________________________________________________________\n')
    print(name1,' and ',name2,' k-Neighbors:','\n____________________________________________________________\n')
    print('Prediction: \n\n',yp,'\n')
    print('Real: \n\n',yv,'\n____________________________________________________________\n')
    #mean accuracy, clf.score(xv,yv)
    comp = yp == yv
    c= Counter(comp)
    print('Accuracy: ',c[1]/(c[0]+c[1]),'\n')
    print(c,'\n____________________________________________________________\n')
    #confusion matrix
    titles_options = [("Confusion matrix", None), ("Confusion matrix, normalized", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(clf, xv, yv, cmap=plt.cm.Blues, normalize=normalize)
        disp.ax_.set_title(title)
        #print(title)
        #print(disp.confusion_matrix)
        plt.savefig(my_path+'/Figures/'+name1+'_'+name2+'_k-neighrbors_'+title+'.png')
        plt.close()
    #classification report
    print('Classification Report: \n\n',classification_report(yv, yp),'\n____________________________________________________________\n')

def kneighAnalysis3(y,data1,data2,data3,name1,name2,name3):

    #dataset to n-array
    data = pd.concat([data1, data2, data3], axis=1, sort=False)
    #print(data.head(150))
    data=data.to_numpy()
    #split rows, training and test
    clf = KNeighborsClassifier()
    yt=np.concatenate([y[:40], y[51:90], y[101:140]])
    xt = np.concatenate([data[:40,:], data[51:90,:], data[101:140,:]])
    yv=np.concatenate([y[40:50], y[90:100], y[140:150]])
    xv = np.concatenate([data[40:50,:], data[90:100,:], data[140:150,:]])
    #fit, predict, print
    clf.fit(xt, yt)
    yp=clf.predict(xv)
    print('____________________________________________________________\n')
    print(name1,', ',name2,' and ',name3,' k-Neighbors:','\n____________________________________________________________\n')
    print('Prediction: \n\n',yp,'\n')
    print('Real: \n\n',yv,'\n____________________________________________________________\n')
    #mean accuracy, clf.score(xv,yv)
    comp = yp == yv
    c= Counter(comp)
    print('Accuracy: ',c[1]/(c[0]+c[1]),'\n')
    print(c,'\n____________________________________________________________\n')
    #confusion matrix
    titles_options = [("Confusion matrix", None), ("Confusion matrix, normalized", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(clf, xv, yv, cmap=plt.cm.Blues, normalize=normalize)
        disp.ax_.set_title(title)
        #print(title)
        #print(disp.confusion_matrix)
        plt.savefig(my_path+'/Figures/'+name1+'_'+name2+'_'+name3+'_k-neighrbors_'+title+'.png')
        plt.close()
    #classification report
    print('Classification Report: \n\n',classification_report(yv, yp),'\n____________________________________________________________\n')

def kneighAnalysis4(y,data1,data2,data3,data4,name1,name2,name3,name4):

    #dataset to n-array
    data = pd.concat([data1, data2, data3, data4], axis=1, sort=False)
    #print(data.head(150))
    data=data.to_numpy()
    #split rows, training and test
    clf = KNeighborsClassifier()
    yt=np.concatenate([y[:40], y[51:90], y[101:140]])
    xt = np.concatenate([data[:40,:], data[51:90,:], data[101:140,:]])
    yv=np.concatenate([y[40:50], y[90:100], y[140:150]])
    xv = np.concatenate([data[40:50,:], data[90:100,:], data[140:150,:]])
    #fit, predict, print
    clf.fit(xt, yt)
    yp=clf.predict(xv)
    print('____________________________________________________________\n')
    print(name1,', ',name2,', ',name3,' and ',name4,' k-Neighbors:','\n____________________________________________________________\n')
    print('Prediction: \n\n',yp,'\n')
    print('Real: \n\n',yv,'\n____________________________________________________________\n')
    #mean accuracy, clf.score(xv,yv)
    comp = yp == yv
    c= Counter(comp)
    print('Accuracy: ',c[1]/(c[0]+c[1]),'\n')
    print(c,'\n____________________________________________________________\n')
    #confusion matrix
    titles_options = [("Confusion matrix", None), ("Confusion matrix, normalized", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(clf, xv, yv, cmap=plt.cm.Blues, normalize=normalize)
        disp.ax_.set_title(title)
        #print(title)
        #print(disp.confusion_matrix)
        plt.savefig(my_path+'/Figures/'+name1+'_'+name2+'_'+name3+'_'+name4+'_k-neighrbors_'+title+'.png')
        plt.close()
    #classification report
    print('Classification Report: \n\n',classification_report(yv, yp),'\n____________________________________________________________\n')

#support vector machines
def svmAnalysis(y,data1,data2,name1,name2):

    #dataset to n-array
    data = pd.concat([data1, data2], axis=1, sort=False)
    #print(data.head(150))
    data=data.to_numpy()
    #split rows, training and test
    clf = svm.SVC()
    yt=np.concatenate([y[:40], y[51:90], y[101:140]])
    xt = np.concatenate([data[:40,:], data[51:90,:], data[101:140,:]])
    yv=np.concatenate([y[40:50], y[90:100], y[140:150]])
    xv = np.concatenate([data[40:50,:], data[90:100,:], data[140:150,:]])
    #fit, predict, print
    clf.fit(xt, yt)
    yp=clf.predict(xv)
    print('____________________________________________________________\n')
    print(name1,' and ',name2,' Support Vector Machines:','\n____________________________________________________________\n')
    print('Prediction: \n\n',yp,'\n')
    print('Real: \n\n',yv,'\n____________________________________________________________\n')
    #mean accuracy, clf.score(xv,yv)
    comp = yp == yv
    c= Counter(comp)
    print('Accuracy: ',c[1]/(c[0]+c[1]),'\n')
    print(c,'\n____________________________________________________________\n')
    #confusion matrix
    titles_options = [("Confusion matrix", None), ("Confusion matrix, normalized", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(clf, xv, yv, cmap=plt.cm.Blues, normalize=normalize)
        disp.ax_.set_title(title)
        #print(title)
        #print(disp.confusion_matrix)
        plt.savefig(my_path+'/Figures/'+name1+'_'+name2+'_SVM_'+title+'.png')
        plt.close()
    #classification report
    print('Classification Report: \n\n',classification_report(yv, yp),'\n____________________________________________________________\n')

def svmAnalysis3(y,data1,data2,data3,name1,name2,name3):

    #dataset to n-array
    data = pd.concat([data1, data2, data3], axis=1, sort=False)
    #print(data.head(150))
    data=data.to_numpy()
    #split rows, training and test
    clf = svm.SVC()
    yt=np.concatenate([y[:40], y[51:90], y[101:140]])
    xt = np.concatenate([data[:40,:], data[51:90,:], data[101:140,:]])
    yv=np.concatenate([y[40:50], y[90:100], y[140:150]])
    xv = np.concatenate([data[40:50,:], data[90:100,:], data[140:150,:]])
    #fit, predict, print
    clf.fit(xt, yt)
    yp=clf.predict(xv)
    print('____________________________________________________________\n')
    print(name1,', ',name2,' and ',name3,' Support Vector Machines:','\n____________________________________________________________\n')
    print('Prediction: \n\n',yp,'\n')
    print('Real: \n\n',yv,'\n____________________________________________________________\n')
    #mean accuracy, clf.score(xv,yv)
    comp = yp == yv
    c= Counter(comp)
    print('Accuracy: ',c[1]/(c[0]+c[1]),'\n')
    print(c,'\n____________________________________________________________\n')
    #confusion matrix
    titles_options = [("Confusion matrix", None), ("Confusion matrix, normalized", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(clf, xv, yv, cmap=plt.cm.Blues, normalize=normalize)
        disp.ax_.set_title(title)
        #print(title)
        #print(disp.confusion_matrix)
        plt.savefig(my_path+'/Figures/'+name1+'_'+name2+'_'+name3+'_SVM_'+title+'.png')
        plt.close()
    #classification report
    print('Classification Report: \n\n',classification_report(yv, yp),'\n____________________________________________________________\n')

def svmAnalysis4(y,data1,data2,data3,data4,name1,name2,name3,name4):

    #dataset to n-array
    data = pd.concat([data1, data2, data3, data4], axis=1, sort=False)
    #print(data.head(150))
    data=data.to_numpy()
    #split rows, training and test
    clf = svm.SVC()
    yt=np.concatenate([y[:40], y[51:90], y[101:140]])
    xt = np.concatenate([data[:40,:], data[51:90,:], data[101:140,:]])
    yv=np.concatenate([y[40:50], y[90:100], y[140:150]])
    xv = np.concatenate([data[40:50,:], data[90:100,:], data[140:150,:]])
    #fit, predict, print
    clf.fit(xt, yt)
    yp=clf.predict(xv)
    print('____________________________________________________________\n')
    print(name1,', ',name2,', ',name3,' and ',name4,' Support Vector Machines:','\n____________________________________________________________\n')
    print('Prediction: \n\n',yp,'\n')
    print('Real: \n\n',yv,'\n____________________________________________________________\n')
    #mean accuracy, clf.score(xv,yv)
    comp = yp == yv
    c= Counter(comp)
    print('Accuracy: ',c[1]/(c[0]+c[1]),'\n')
    print(c,'\n____________________________________________________________\n')
    #confusion matrix
    titles_options = [("Confusion matrix", None), ("Confusion matrix, normalized", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(clf, xv, yv, cmap=plt.cm.Blues, normalize=normalize)
        disp.ax_.set_title(title)
        #print(title)
        #print(disp.confusion_matrix)
        plt.savefig(my_path+'/Figures/'+name1+'_'+name2+'_'+name3+'_'+name4+'_SVM_'+title+'.png')
        plt.close()
    #classification report
    print('Classification Report: \n\n',classification_report(yv, yp),'\n____________________________________________________________\n')

# functions end

iris = datasets.load_iris() #loading dataset
y = iris.target #classification

table=pd.DataFrame(iris.data)

sl = pd.DataFrame(iris.data[:,0:1]) #sepal lenght

sw = pd.DataFrame(iris.data[:,1:2]) #sepal width

pl = pd.DataFrame(iris.data[:,2:3]) #petal lenght

pw = pd.DataFrame(iris.data[:,3:4]) #petal width
#sl,sw,pl,pw
#this makes all columns be named 0, but no problemo

#plots and stuff
    #write to txt instead of console(mine lacks space)
sys.stdout=open(my_path+'/output.txt','w')

#sl,sw
scatter(y,sl,sw,'Sepal Lenght','Sepal Width')
mlpAnalysis(y,sl,sw,'Sepal Lenght','Sepal Width')
kneighAnalysis(y,sl,sw,'Sepal Lenght','Sepal Width')
svmAnalysis(y,sl,sw,'Sepal Lenght','Sepal Width')
#sl,pl
scatter(y,sl,pl,'Sepal Lenght','Petal Lenght')
mlpAnalysis(y,sl,pl,'Sepal Lenght','Petal Lenght')
kneighAnalysis(y,sl,pl,'Sepal Lenght','Petal Lenght')
svmAnalysis(y,sl,pl,'Sepal Lenght','Petal Lenght')
#sl,pw
scatter(y,sl,pw,'Sepal Lenght','Petal Width')
mlpAnalysis(y,sl,pw,'Sepal Lenght','Petal Width')
kneighAnalysis(y,sl,pw,'Sepal Lenght','Petal Width')
svmAnalysis(y,sl,pw,'Sepal Lenght','Petal Width')
#sw,pl
scatter(y,sw,pl,'Sepal Width','Petal Lenght')
mlpAnalysis(y,sw,pl,'Sepal Width','Petal Lenght')
kneighAnalysis(y,sw,pl,'Sepal Width','Petal Lenght')
svmAnalysis(y,sw,pl,'Sepal Width','Petal Lenght')
#sw,pw
scatter(y,sw,pw,'Sepal Width','Petal Width')
mlpAnalysis(y,sw,pw,'Sepal Width','Petal Width')
kneighAnalysis(y,sw,pw,'Sepal Width','Petal Width')
svmAnalysis(y,sw,pw,'Sepal Width','Petal Width')
#pl,pw
scatter(y,pl,pw,'Petal Lenght','Petal Width')
mlpAnalysis(y,pl,pw,'Petal Lenght','Petal Width')
kneighAnalysis(y,pl,pw,'Petal Lenght','Petal Width')
svmAnalysis(y,pl,pw,'Petal Lenght','Petal Width')
#sl,sw,pl
mlpAnalysis3(y,sl,sw,pl,'Sepal Lenght','Sepal Width','Petal Lenght')
kneighAnalysis3(y,sl,sw,pl,'Sepal Lenght','Sepal Width','Petal Lenght')
svmAnalysis3(y,sl,sw,pl,'Sepal Lenght','Sepal Width','Petal Lenght')
#sl,sw,pw
mlpAnalysis3(y,sl,sw,pw,'Sepal Lenght','Sepal Width','Petal Width')
kneighAnalysis3(y,sl,sw,pw,'Sepal Lenght','Sepal Width','Petal Width')
svmAnalysis3(y,sl,sw,pw,'Sepal Lenght','Sepal Width','Petal Width')
#sl,pl,pw
mlpAnalysis3(y,sl,pl,pw,'Sepal Lenght','Petal Lenght','Petal Width')
kneighAnalysis3(y,sl,pl,pw,'Sepal Lenght','Petal Lenght','Petal Width')
svmAnalysis3(y,sl,pl,pw,'Sepal Lenght','Petal Lenght','Petal Width')
#sw,pl,pw
mlpAnalysis3(y,sw,pl,pw,'Sepal Width','Petal Lenght','Petal Lenght')
kneighAnalysis3(y,sw,pl,pw,'Sepal Width','Petal Lenght','Petal Lenght')
svmAnalysis3(y,sw,pl,pw,'Sepal Width','Petal Lenght','Petal Lenght')
#sl,sw,pl,pw
mlpAnalysis4(y,sl,sw,pl,pw,'Sepal Lenght','Sepal Width','Petal Lenght','Petal Width')
kneighAnalysis4(y,sl,sw,pl,pw,'Sepal Lenght','Sepal Width','Petal Lenght','Petal Width')
svmAnalysis4(y,sl,sw,pl,pw,'Sepal Lenght','Sepal Width','Petal Lenght','Petal Width')

sys.stdout.close()

#some of them return 100% accuracy, not a lot of data to work with, may be overfitted
    #and thus show low precision when new data is introduced