# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 19:12:15 2022

@author: ilngy

"""

import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
import random
import os

root = os.getcwd()
os.chdir(r"G:\マイドライブ\python\3Dモデル")
train_points = np.load("train_points.npy")
# test_points = np.load("test_points.npy")
# train_labels = np.load("train_labels.npy")
# test_labels = np.load("test_labels.npy")
# CLASS_MAP = np.load("CLASS_MAP.npy", allow_pickle=True).tolist()
os.chdir(root)

def plot3D(x, y, z, dataName=""):
    # Figureを追加
    fig = plt.figure(figsize=(8, 8))
    # 3DAxesを追加
    ax = fig.add_subplot(111, projection='3d')
    # Axesのタイトルを設定
    ax.set_title(dataName, size=20)
    # 軸ラベルを設定
    ax.set_xlabel("x", size=14)
    ax.set_ylabel("y", size=14)
    ax.set_zlabel("z", size=14)
    # 曲線を描画
    ax.scatter(x, y, z, s=40, c="blue")

    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() * 0.5
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()

def Center(points):
    for i in range(points.shape[0]):
        centerX = points[i,:,0].mean()
        centerY = points[i,:,1].mean()
        centerZ = points[i,:,2].mean()
        points[i,:,0] = points[i,:,0] - centerX
        points[i,:,1] = points[i,:,1] - centerY
        points[i,:,2] = points[i,:,2] - centerZ
    return points

def CreateAdjacencyMatrixWithAhatH(points):
    neighborNum = 4#その節点自身を含めた数
    AhatH=[]
    for model in range(points.shape[0]):
        print("processing: model",model)
        adjacencyMatrixPlusI = np.zeros((samplingPoints,samplingPoints))
        for i in range(samplingPoints):
            squaredDistanceOfLinei=[]
            for j in range(samplingPoints):
                squaredDistance = np.sum((points[model, i, :] - points[model, j, :])**2)
                squaredDistanceOfLinei.append(squaredDistance)
            sortIndex = np.argsort(squaredDistanceOfLinei)[:neighborNum]
            adjacencyMatrixPlusI[i,sortIndex] = 1
            # plt.hist(squaredDistanceOfLinei, bins=50)
            # plt.show()
        AhatH.append(adjacencyMatrixPlusI / neighborNum)
    AhatH = np.array(AhatH)
    return AhatH

def CreateAdjacencyMatrixWithAhatH2(points):
    neighborDistance = 0.1
    AhatH=[]
    for model in range(points.shape[0]):
        print("processing: model",model)
        adjacencyMatrixPlusI = np.zeros((samplingPoints,samplingPoints))
        for i in range(samplingPoints):
            squaredDistanceOfLinei=[]
            for j in range(samplingPoints):
                squaredDistance = np.sum((points[model, i, :] - points[model, j, :])**2)
                squaredDistanceOfLinei.append(squaredDistance)
            neighborIndex = np.array(squaredDistanceOfLinei) < neighborDistance
            adjacencyMatrixPlusI[i,neighborIndex] = 1
            # plt.hist(squaredDistanceOfLinei, bins=50)
            # plt.show()
        D = np.sum(adjacencyMatrixPlusI, axis=1)
        Dii = np.eye(D.shape[0])*D
        D2 = Dii**-0.5
        D2[D2 == np.inf] = 0
        AhatHone = np.dot(D2, adjacencyMatrixPlusI)
        AhatHone = np.dot(AhatHone, D2)
        AhatH.append(AhatHone)
    AhatH = np.array(AhatH)
    return AhatH

# trainデータ数削減
random.seed(0)
train_index = random.sample(range(train_points.shape[0]), 1000)
train_points = train_points[train_index,:,:]
# train_labels = train_labels[train_index]

# ノード数削減
samplingPoints = 512
random.seed(0)
index = random.sample(range(train_points.shape[1]),samplingPoints)
train_points = train_points[:,index,:]
# test_points = test_points[:,index,:]

train_points = Center(train_points)

# 可視化
for i in range(3):
    plot3D(train_points[i,:,0], train_points[i,:,1], train_points[i,:,2])

AhatH_train = CreateAdjacencyMatrixWithAhatH(train_points)

os.chdir(r"G:\マイドライブ\python\3Dモデル\GNN\graphData")
np.save("AhatH_train_neighborDistance.npy", AhatH_train)
np.save("points_train_neighborDistance.npy", train_points)
