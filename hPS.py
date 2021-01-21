import os
import random 
import numpy as np
from scipy.optimize import *
from sklearn import decomposition
#TXT DOSYASINDAN VERİ OKUMA
#PCA ALGORITMASI 
data=np.loadtxt('data1.txt')
X=data[:,0:np.shape(data)[1]-1]
y=data[:,np.shape(data)[1]-1]

pca = decomposition.PCA(n_components=20)
pca.fit(X)
X = pca.transform(X)

data1=np.c_[X,y]
#PCA ALGORITMASI 
dim=np.shape(data1)[1]-1
print(dim)
A= data1[np.where(data1[:,dim]==0)]
B= data1[np.where(data1[:,dim]==1)]
#TXT DOSYASINDAN VERİ OKUMA

#SciPy İLE H-POLYHEDRAL AYIRMA YÖNTEMİ (SLSQP)
e=1
hpol=6
p= np.zeros((1,hpol*(dim+1)))
def f(p):
    p = np.reshape(p,(hpol,dim+1))
    ha=hb=errorA=errorB=0
    for i in range(len(A)):
        for k in range(hpol):
            if A[i][0]*p[k][0] +A[i][1]*p[k][1] - p[k][2] + e > 0:
                errorA=errorA + A[i][0]*p[k][0] +A[i][1]*p[k][1] - p[k][2] + e
        ha=ha+errorA
    for i in range(len(B)):
        tempy=np.zeros((hpol,1))
        for k in range(hpol):
            tempy[k] = -B[i][0]*p[k][0] - B[i][1]*p[k][1] + p[k][2] + e
            y = np.min(tempy)
        errorB=max(0,y)
        hb=hb+errorB
    return ha/len(A) + hb/len(B)
outcome=(minimize(f,p,method='tnc'))
#SciPy İLE H-POLYHEDRAL AYIRMA YÖNTEMİ (SLSQP)

#HATALARIN EKRANA YAZDIRILMASI
p=np.array(outcome.x)
p=np.reshape(p,(hpol,dim+1))
for i in range(len(A)):
    ha=errorA=0
    for k in range(hpol):
        if A[i][0]*p[k][0] +A[i][1]*p[k][1] - p[k][2] + e > 0:
            errorA=errorA + A[i][0]*p[k][0] +A[i][1]*p[k][1] - p[k][2] + e
    ha = ha+ errorA
    print('A kümesi',i+1,'elamanı sınıflandırma hatası =',errorA)
for i in range(len(B)):
    hb=errorB=0
    tempy=np.zeros((hpol,1))
    for k in range(hpol):
        tempy[k] = -B[i][0]*p[k][0] - B[i][1]*p[k][1] + p[k][2] + e
        y = np.min(tempy)
    errorB=(max(0,y))
    hb=hb+errorB
    print('B kümesi',i+1,'elamanı sınıflandırma hatası =',errorB)
#HATALARIN EKRANA YAZDIRILMASI

#GÖRSELLEŞTİRME İÇİN LIBRARY EKLEME
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import OrderedDict
from mpl_toolkits.mplot3d import Axes3D
#GÖRSELLEŞTİRME İÇİN LIBRARY EKLEME

#SciPy İLE ELDE EDİLEN HİPERDÜZLEMLERİN GÖRSELLEŞTİRİLMESİ
a=np.reshape(outcome.x,(hpol,dim+1))
striderz=4
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(A[:,0],A[:,1],s=10,c='r',marker='s')
ax.scatter(B[:,0],B[:,1],s=10,c='k',marker='o')
x = np.arange(-striderz, striderz+0.1, 0.1)
y = np.arange(-striderz, striderz+0.1, 0.1)
X,Y=np.meshgrid(x,y)
for i in range(hpol):
    print('z=',np.round(a[i][0],2),'x\t+',np.round(a[i][1],2),'y\t-',np.round(a[i][2],2))
    Z= a[i][0]*X+ a[i][1] *Y- a[i][2]
    ax.plot_wireframe(X,Y,Z,rstride=striderz*5, cstride=striderz*5)
plt.show()
#SciPy İLE ELDE EDİLEN HİPERDÜZLEMLERİN GÖRSELLEŞTİRİLMESİ
