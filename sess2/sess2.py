import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import scale

img = cv.imread('rose.png') 
cv.imshow('original image', img)
cv.waitKey(0)

#resize 
scale = 0.5
Height = int(img.shape[0]*scale)
Width = int(img.shape[1]*scale)
Dimension = (Width, Height)
img = cv.resize(img, Dimension, cv.INTER_AREA)
cv.imshow('Hasil Resize', img)
cv.waitKey(0)

#grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gambar di grayscaled', gray)
cv.waitKey(0)

#hitung pixel per intesitas
intensity_count = np.zeros(256, dtype=int)

#iterate pixel 
h = gray.shape[0]
w = gray.shape[1]
for i in range(h):
    for j in range(w):
        intensity_count[gray[i][j]] +=1

#plotting 
plt.figure(1)
plt.plot(intensity_count, 'g', label= 'Intesity')
plt.legend(loc='upper right')
plt.ylabel('Number of pixel')
plt.xlabel('intensity')
plt.show()

#histogram equalization
eq = cv.equalizeHist(gray)
cv.imshow('Hasil equalization', eq)
cv.waitKey(0)

#side by side
res = np.hstack((gray,eq)) #htack = horizontal, vstack = vertical
cv.imshow('side by side', res)
cv.waitKey(0)

#histo untuk compare
intensity_count2 = np.zeros(256, dtype=int)
h = eq.shape[0]
w = eq.shape[1]
for i in range(h):
    for j in range(w):
        intensity_count2[eq[i][j]] +=1

plt.figure(2,(10,5)) #10,5 itu ukuran yg background putih
plt.subplot(1,2,1) # 1 : row , 2 : collumn , 1 : penempatan histo dmna
plt.plot(intensity_count, 'g', label= 'Intesity(before)')
plt.legend(loc='upper right')
plt.ylabel('Number of pixel')
plt.xlabel('intensity')

plt.subplot(1,2,2)
plt.plot(intensity_count2, 'g', label= 'Intesity(after)')
plt.legend(loc='upper right')
plt.ylabel('Number of pixel')
plt.xlabel('intensity')
plt.show()

