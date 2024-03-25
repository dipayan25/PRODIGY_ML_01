import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay
import pickle 
from sklearn import metrics
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

#import warning
import warnings
warnings.filterwarnings("ignore")

# data processing, CSV file I/O (e.g. pd.read_csv)
# Read the file "data.csv" and print the contents.
#!cat data/data.csv
data = pd.read_csv('breastcancer/data.csv', index_col=False,)
data.drop('id', axis =1, inplace=True)
data.head()
print(data)
data.info()
data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})

array = data.values
X = array[:,1:31]
y = array[:,0]

le = LabelEncoder()
y = le.fit_transform(y)


##Split data set in train 70% and test 30%
x_train, x_test, y_train, y_test = train_test_split( X, y, test_size=0.25, random_state=7)
x_train.shape, y_train.shape, x_test.shape, y_test.shape


# Normalize the  data (center around 0 and scale to remove the variance).
##Xs = scaler.fit_transform(X)

# feature extraction
pca = PCA(n_components=10)
fit = pca.fit(X)

X_pca = pca.transform(X)

PCA_df = pd.DataFrame()

PCA_df['PCA_1'] = X_pca[:,0]
PCA_df['PCA_2'] = X_pca[:,1]

plt.plot(PCA_df['PCA_1'][data.diagnosis == 'M'],PCA_df['PCA_2'][data.diagnosis == 'M'],'o', alpha = 0.7, color = 'r')
plt.plot(PCA_df['PCA_1'][data.diagnosis == 'B'],PCA_df['PCA_2'][data.diagnosis == 'B'],'o', alpha = 0.7, color = 'b')

plt.xlabel('PCA_1')
plt.ylabel('PCA_2')
plt.legend(['Malignant','Benign'])
plt.show()

clf = SVC(probability=True)
clf.fit(x_train, y_train)
y_pred = clf.fit(x_train, y_train).predict(x_test)
cm =confusion_matrix(y_test, y_pred)

outputTree=SVC()
outputTree.fit(x_train, y_train)
predTree=outputTree.predict(x_test)
train_tree=outputTree.score(x_train,y_train)
test_tree=accuracy_score(y_test, predTree)
f1_tree = f1_score(y_test, predTree, average='weighted')

print("Train set Accuracy: ", train_tree*100,"%")
print("Test set Accuracy: ", test_tree*100,"%")
print("SVM Algorithm f1 Score: ", f1_tree*100,"%")
con=confusion_matrix(y_test,predTree)
print("\n",con)

filename='bc.sav'
#pickle.dump(outputTree,open(filename,'wb'))



