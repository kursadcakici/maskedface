import os
import numpy as np
from PIL import Image
#Read from folder
face=(os.listdir("covid-mask-detector\\data\\self-built-masked-face-recognition-dataset\\AFDB_face_dataset"))
faceimage=[]
img=[]
for i in range(len(face)):
    faceimage=(os.listdir("covid-mask-detector\\data\\self-built-masked-face-recognition-dataset\\AFDB_face_dataset"+"\\"+face[i]))
    for j in range(5):
        img.append("covid-mask-detector\\data\\self-built-masked-face-recognition-dataset\\AFDB_face_dataset"+"\\"+face[i]+"\\"+faceimage[j])

masked_face=(os.listdir("covid-mask-detector\\data\\self-built-masked-face-recognition-dataset\\AFDB_masked_face_dataset"))
masked_faceimage=[]
maskedimg=[]
for i in range(len(masked_face)):
    masked_faceimage=(os.listdir("covid-mask-detector\\data\\self-built-masked-face-recognition-dataset\\AFDB_masked_face_dataset"+"\\"+masked_face[i]))
    for j in range(len(masked_faceimage)):
        maskedimg.append("covid-mask-detector\\data\\self-built-masked-face-recognition-dataset\\AFDB_masked_face_dataset"+"\\"+masked_face[i]+"\\"+masked_faceimage[j])

#Add to Matrix
dim = 40
facematrix=[]
for i in range(len(img)):
    facematrix.append(np.append(((np.reshape(np.array(Image.open(img[i]).convert('L').resize((dim,dim))),(1,dim*dim)))[0]),0))
maskedfacematrix=[]
for j in range(len(maskedimg)):
    maskedfacematrix.append(np.append(((np.reshape(np.array(Image.open(maskedimg[j]).convert('L').resize((dim,dim))),(1,dim*dim)))[0]),1))
facematrix = np.transpose(np.transpose(facematrix))
maskedfacematrix = np.transpose(np.transpose(maskedfacematrix))
np.save("facedata", facematrix)
np.save("maskedfacedata", maskedfacematrix)
np.save("data", np.r_[facematrix,maskedfacematrix])
np.savetxt("data.txt", np.r_[facematrix,maskedfacematrix],delimiter=' ', fmt='%i')