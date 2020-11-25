
"""
@author: Alican AKCA
"""
import pandas as pd #Kütüphaneleri ekledik!
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

veriler = pd.read_csv('Datasets.csv') #Veri setimizi yükledik. Aynı dizinde bulunduğuna dikkat edelim.
x = veriler.iloc[:,3:7].values # Kolonlarımızı görüntüleyeceğiz.
y = veriler.iloc[:,0:1].values # Kolonlarımızı görüntüleyeceğiz.

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.25, random_state=0)
# Değişkenlerimize verilerimizin %30'u nu öğrenmesi için ayırdık.
sc=StandardScaler() #Ölçekleme
X_train = sc.fit_transform(x_train) #Ölçekleme yaptık
X_test = sc.transform(x_test) #Ölçekleme yaptık

svc = SVC(kernel = "linear") #linear,sigmoid,rbf,poly
svc.fit(X_train,y_train) #Öğren/Uygula
y_pred = svc.predict(X_test) #Tahmin edecek
cm =confusion_matrix(y_test, y_pred) #Karmaşıklık matrisinde göreceğiz.
print("Linear")
print(cm)

svc = SVC(kernel = "sigmoid")
svc.fit(X_train,y_train) #Öğren/Uygula
y_pred = svc.predict(X_test) #Tahmin edecek
cm =confusion_matrix(y_test, y_pred)
print("Sigmoid")
print(cm)

svc = SVC(kernel = "rbf")
svc.fit(X_train,y_train) #Öğren/Uygula
y_pred = svc.predict(X_test) #Tahmin edecek
cm =confusion_matrix(y_test, y_pred)
print("Sigmoid")
print(cm)

svc = SVC(kernel = "poly")
svc.fit(X_train,y_train) #Öğren/Uygula
y_pred = svc.predict(X_test) #Tahmin edecek
cm =confusion_matrix(y_test, y_pred)
print("Sigmoid")
print(cm)