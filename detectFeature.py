# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 17:13:05 2023

@author: ilngy
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
from PIL import Image
from sklearn.cluster import KMeans
from numpy.linalg import norm

# pathSave = r"C:\Users\ilngy\Downloads\pokemon-images-dataset-by-type-master\pokemon.npz"
# load = np.load(pathSave)
# x_train=load['arr_0']
# typeList=load['arr_1']
# nameList=load['arr_2']
# from sklearn import preprocessing
# le = preprocessing.LabelEncoder()
# y_train = le.fit_transform(typeList)
# print(list(le.classes_))

root = r"C:\Users\ilngy\Pictures\SSD256GBバックアップ\α6000\2023-04-09\*.JPG"
file = glob.glob(root)
imgList=[]
for i in file:
    img = Image.open(i)
    img = img.resize((256,256))
    img = np.array(img)
    imgList.append(img)
x_train = np.array(imgList)

x_train = x_train.astype("uint8")

def GetDescriptors():
    descriptors=[]
    for i in range(x_train.shape[0]):
        gray= cv2.cvtColor(x_train[i,:,:,:], cv2.COLOR_BGR2GRAY)

        detector = cv2.SIFT_create()
        kp, img_descriptors = detector.detectAndCompute(gray, None)
        descriptors.append(int(img_descriptors))
        print(len(kp), img_descriptors.shape)

        # img = cv2.drawKeypoints(gray, kp, None)
        # plt.imshow(img)
        # plt.show()
    return descriptors

descriptors = GetDescriptors()

ClusterNum=16 #visual wordsの個数(SIFT等で得た局所特徴量を何個のクラスターに集約するか)

def CreateFrequencyVectors():
    cl = KMeans(n_clusters=ClusterNum)
    cl.fit(np.concatenate(descriptors))
    # codebook = cl.cluster_centers_#code book

    # visual_words = []
    frequency_vectors=[]
    for i in range(x_train.shape[0]):
        img_visual_words = cl.predict(descriptors[i])#index of the closest code in the code book
        # distance = cl.transform(descriptors[i])#each dimension is the distance to the cluster centers
        # visual_words.append(img_visual_words)

        # create a frequency vector for each image
        img_frequency_vector = np.zeros(ClusterNum)
        for word in img_visual_words:
            img_frequency_vector[word] += 1
        frequency_vectors.append(img_frequency_vector)

        # plt.bar(range(ClusterNum), img_frequency_vector)
        # plt.show()
    frequency_vectors = np.array(frequency_vectors)
    return frequency_vectors

frequency_vectors = CreateFrequencyVectors()

def CheckSimilarImg(checkImgID, similarityAlgorithm = "cos"):
    top_k=5
    a = frequency_vectors[checkImgID]
    b = frequency_vectors

    if similarityAlgorithm == "cos":
        similarity = -np.dot(a, b.T)/(norm(a) * norm(b, axis=1))
    elif similarityAlgorithm == "euc":
        similarity = np.sum((a-b)**2, axis=1)

    # get the top k indices for most similar vecs
    idx = np.argsort(similarity)[:top_k]
    for i in idx:
        print(f"{i}: {round(similarity[i], 4)}")
        # plt.imshow(imgList[i], cmap='gray')
        # plt.show()

        fig, ax = plt.subplots(1,2,figsize=(10,5))
        ax[0].imshow(imgList[i], cmap='gray')
        ax[1].bar(range(ClusterNum), frequency_vectors[i,:])
        plt.show()

CheckSimilarImg(0, "cos")
CheckSimilarImg(0, "euc")
