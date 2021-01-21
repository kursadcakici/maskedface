#KÜTÜPHANE İMPORT
from gurobipy import *
import os
import random 
import numpy as np
from sklearn import decomposition
#KÜTÜPHANE İMPORT

#TXT DOSYASINDAN VERİ OKUMA

data=np.loadtxt('data1.txt')
X=data[:,0:np.shape(data)[1]-1]
y=data[:,np.shape(data)[1]-1]

pca = decomposition.PCA(n_components=20)
pca.fit(X)
X = pca.transform(X)

data1=np.c_[X,y]

#TXT DOSYASINDAN VERİ OKUMA 

#VERİ ÖZELLİKLERİNİ ÇIKARMA
boyut=np.shape(data1)[1]-1
instance=np.shape(data1)[0]-1
m=Model('RLP')
gamma = m.addVar(vtype=GRB.CONTINUOUS, lb=0,name='gamma')
w=[]
for i in range(boyut):
    w.append(0)
    w[i]=m.addVar(vtype=GRB.CONTINUOUS, lb=0,name='w[%s]' % i)
A= data1[np.where(data1[:,boyut]==1)]
B= data1[np.where(data1[:,boyut]==0)]
mm=len(A)
kk=len(B)
#VERİ ÖZELLİKLERİNİ ÇIKARMA

#HOLDOUT YAKLAŞIMINA GÖRE EĞİTİM VE TEST OLUŞTURMA (A0)
trainmm=round(mm*0.8)
trainkk=round(kk*0.8)
testmm=mm-trainmm
testkk=kk-trainkk
testA = []
for i in range(testmm):
    r = random.randint(0,mm-1)
    testA.append(A[r])
    A = np.delete(A,r,axis=0)
    mm = mm-1
testA = np.transpose(testA)
testA = np.transpose(testA)
testB = []
for i in range(testkk):
    r = random.randint(0,kk-1)
    testB.append(B[r])
    B = np.delete(B,r,axis=0)
    kk = kk-1
testB = np.transpose(testB)
testB = np.transpose(testB)
#HOLDOUT YAKLAŞIMINA GÖRE EĞİTİM VE TEST OLUŞTURMA (A0)

#GUROBİ İLE OPTİMİZASYON (A1)
y=m.addVars(mm,vtype=GRB.CONTINUOUS,lb=0)
z=m.addVars(kk,vtype=GRB.CONTINUOUS,lb=0)
for i in range(mm):
    m.addConstr(-quicksum(A[i][j]*w[j] for j in range(boyut))+gamma+1<=y[i])

for i in range(kk):
    m.addConstr(quicksum(B[i][j]*w[j] for j in range(boyut))-gamma+1<=z[i])

m.setObjective(quicksum(y[i]for i in range(mm))/mm+quicksum(z[j]for j in range(kk))/kk,GRB.MINIMIZE)
m.update()
print(m)
m.optimize()
W=[]
for i in range(boyut):
    W.append(0)
    W[i] = w[i].X
    print('w[%s]' %i,W[i])
print('gamma \t', gamma.X,'\n')
#GUROBİ İLE OPTİMİZASYON (A1)

#EĞİTİM KÜMESİ HSM OLUŞTURMA (A2)
Y = np.matmul(A[:,0:boyut],np.transpose(W))-gamma.X
Z = np.matmul(B[:,0:boyut],np.transpose(W))-gamma.X
traintp = 0
traintn = 0
trainfn = 0
trainfp = 0
for i in range(mm):
    if Y[i]<0:
        trainfp = trainfp +1
    else:
        traintp = traintp +1
for i in range(kk):
    if Z[i]>0:
        trainfn = trainfn +1
    else:
        traintn = traintn +1
print('HSM Eğitim')
print('Tahmin Sınıfı\t 1\t 0')
print('Gerçek\t1\t',traintp,'\t',trainfn,'\nSınıf\t0\t',trainfp,'\t',traintn,'\n')
#EĞİTİM KÜMESİ HSM OLUŞTURMA (A2)

#EĞİTİM KÜMESİ İÇİN BAŞARIM ÖLÇÜMÜ (A3)
trainduyarlilik=(traintp)/(traintp+trainfn)
trainkeskinlik=(traintp)/(traintp+trainfp)
print('Dogruluk:\t',(traintp+traintn)/(mm+kk))
print('Hata Oranı:\t',(trainfp+trainfn)/(mm+kk))
print('Keskinlik:\t',trainkeskinlik)
print('Duyarlılık:\t',trainduyarlilik)
print('F1:\t\t',2*trainduyarlilik*trainkeskinlik/(trainduyarlilik+trainkeskinlik),'\n')
#EĞİTİM KÜMESİ İÇİN BAŞARIM ÖLÇÜMÜ (A3)

#TEST KÜMESİ HSM OLUŞTURMA (A2)
Y = np.matmul(testA[:,0:boyut],np.transpose(W))-gamma.X
Z = np.matmul(testB[:,0:boyut],np.transpose(W))-gamma.X
testtp = 0
testtn = 0
testfn = 0
testfp = 0
for i in range(testmm):
    if Y[i]<0:
        testfp = testfp +1
    else:
        testtp = testtp +1
for i in range(testkk):
    if Z[i]>0:
        testfn = testfn +1
    else:
        testtn = testtn +1
print('HSM Test')
print('Tahmin Sınıfı\t 1\t 0')
print('Gerçek\t1\t',testtp,'\t',testfn,'\nSınıf\t0\t',testfp,'\t',testtn,'\n')
#TEST KÜMESİ HSM OLUŞTURMA (A2)

#TEST KÜMESİ BAŞARIM ÖLÇÜMÜ (A3)
testduyarlilik=(testtp)/(testtp+testfn)
testkeskinlik=(testtp)/(testtp+testfp)
print('Doğruluk:\t',(testtp+testtn)/(testmm+testkk))
print('Hata Oranı:\t',(testfp+testfn)/(testmm+testkk))
print('Keskinlik:\t',testkeskinlik)
print('Duyarlılık:\t',testduyarlilik)
print('F1:\t\t',2*testduyarlilik*testkeskinlik/(testduyarlilik+testkeskinlik),'\n')
#TEST KÜMESİ BAŞARIM ÖLÇÜMÜ (A3)