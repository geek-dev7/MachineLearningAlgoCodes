import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv('/home/dev__7/work/Machine Learning Codes and Datasets/udemy/Part 3 - Classification/Section 16 - Support Vector Machine (SVM)/Python/Social_Network_Ads.csv')
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

classifier=SVC(kernel='linear',random_state=0)

classifier.fit(x_train,y_train)

print(classifier.predict(sc.transform([[30,87000]])))


y_pred=classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix,accuracy_score

cm=confusion_matrix(y_test,y_pred)

print(cm)

print(accuracy_score(y_test,y_pred))

# from matplotlib.colors import ListedColormap

# x_set,y_set=sc.inverse_transform(x_train),y_train

# x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-10,stop=x_set[:,0].max()+10,step=0.25),
#                   np.arange(start=x_set[:,1].min()-1000,stop=x_set[:,1].max()+1000,step=0.25))

# plt.contourf(x1,x2,classifier.predict(sc.transform(np.array([x1.ravel(),x2.ravel()]).T)).reshape(x1.shape),alpha=0.75,cmap=ListedColormap(('red','green')))

# plt.xlim(x1().min,x1().max)
# plt.xlim(x2().min,x2().max)

# for i,j in enumerate(np.unique(y_set)):
#     plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)

# plt.title('svm')
# plt.xlabel('age')
# plt.ylabel('estimate_salary')
# plt.legend()
# plt.show()