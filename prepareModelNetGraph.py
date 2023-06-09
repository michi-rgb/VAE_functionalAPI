# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 19:12:15 2022

@author: ilngy

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import random
import os

root = os.getcwd()
os.chdir(r"G:\マイドライブ\python\pointnet")

train_points = np.load("train_points.npy")
test_points = np.load("test_points.npy")
train_labels = np.load("train_labels.npy")
test_labels = np.load("test_labels.npy")
CLASS_MAP = np.load("CLASS_MAP.npy", allow_pickle=True).tolist()

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

def Scaler_AspectMaintain(points):
    ss_points_all=[]
    for i in range(points.shape[0]):
        ss_X = StandardScaler()
        ss_pointsX = ss_X.fit_transform(points[i, :, 0].reshape(-1,1))
        ss_pointsY = ss_X.transform(points[i, :, 1].reshape(-1,1))
        ss_pointsZ = ss_X.transform(points[i, :, 2].reshape(-1,1))
        ss_points = np.hstack([ss_pointsX,ss_pointsY,ss_pointsZ])
        ss_points_all.append(ss_points)
    return np.array(ss_points_all)

def Scaler_AspectChange(points):
    ss_points_all=[]
    for i in range(points.shape[0]):
        ss_X = StandardScaler()
        ss_Y = StandardScaler()
        ss_Z = StandardScaler()
        ss_pointsX = ss_X.fit_transform(points[i, :, 0].reshape(-1,1))
        ss_pointsY = ss_Y.fit_transform(points[i, :, 1].reshape(-1,1))
        ss_pointsZ = ss_Z.fit_transform(points[i, :, 2].reshape(-1,1))
        ss_points = np.hstack([ss_pointsX,ss_pointsY,ss_pointsZ])
        ss_points_all.append(ss_points)
    return np.array(ss_points_all)

def Scaler_AspectMaintain2(points):
    ss_points_all=[]
    ss_X = StandardScaler()
    for i in range(points.shape[0]):
        ss_points = ss_X.fit_transform(points[i, :, :].reshape(-1,1)).reshape(points[i,:,:].shape)
        ss_points_all.append(ss_points)
    return np.array(ss_points_all)

def Scaler_AspectChange2(points):
    ss_points_all=[]
    ss_X = StandardScaler()
    for i in range(points.shape[0]):
        ss_points = ss_X.fit_transform(points[i, :, :])
        ss_points_all.append(ss_points)
    return np.array(ss_points_all)

def Rotate(points, j=0):
    """
    z軸周りに回転
    points: (*,2048,3)
    deg: ラジアン
    """
    rotated_points=[]
    for i in range(points.shape[0]):
        deg = random.random() * 2 * np.pi
        affine = [[np.cos(deg), -np.sin(deg), 0],
                  [np.sin(deg), np.cos(deg), 0],
                  [0, 0, 1]]
        rotated_point = np.dot(points[i,:,:], affine)
        rotated_points.append(rotated_point)
    rotated_points = np.array(rotated_points)

    # plot3D(points[j,:,0], points[j,:,1], points[j,:,2])
    # plot3D(rotated_points[j,:,0], rotated_points[j,:,1], rotated_points[j,:,2])
    return rotated_points

def CreateAdjacencyMatrix(points):
    adjacencyMatrixList=[]
    for model in range(points.shape[0]):
        print("processing: model",model)
        adjacencyMatrix = np.zeros((samplingPoints,samplingPoints))
        for i in range(samplingPoints):
            squaredDistanceOfLinei=[]
            for j in range(samplingPoints):
                squaredDistance = np.sum((points[model, i, :] - points[model, j, :])**2)
                squaredDistanceOfLinei.append(squaredDistance)
            sortIndex = np.argsort(squaredDistanceOfLinei)[1:4]
            adjacencyMatrix[i,sortIndex] = 1
        adjacencyMatrixList.append(adjacencyMatrix)
    adjacencyMatrixList = np.array(adjacencyMatrixList)
    return adjacencyMatrixList

def CreateAhatH(A):
    AplusI = A + np.eye(A.shape[1])
    D = np.eye(A.shape[1])*0.5
    AhatH=[]
    for model in range(A.shape[0]):
        print("processing: model",model)
        AhatHone = np.dot(D, AplusI[model,:,:])
        AhatHone = np.dot(AhatHone, D)
        AhatH.append(AhatHone)
    return np.array(AhatH)

def CreateAdjacencyMatrixWithAhatH(points):
    neighborNum = 10
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
train_labels = train_labels[train_index]

# ノード数削減
samplingPoints = 512
random.seed(0)
index = random.sample(range(train_points.shape[1]),samplingPoints)
train_points = train_points[:,index,:]
test_points = test_points[:,index,:]

# ランダム回転（tnet検証）
# train_points = Rotate(train_points)
# test_points = Rotate(test_points)

# 標準化
X_train = Scaler_AspectChange2(train_points)
# X_test = Scaler_AspectChange2(test_points)

# 可視化
# for i in range(3):
#     plot3D(train_points[i,:,0], train_points[i,:,1], train_points[i,:,2])
#     plot3D(X_train[i,:,0], X_train[i,:,1], X_train[i,:,2])

# A_train = CreateAdjacencyMatrix(X_train)#(model数,512,512), edge数：各modelにつき1536個(512*3)
# A_test = CreateAdjacencyMatrix(X_test)

# AhatH_train = CreateAhatH(A_train)
# AhatH_test = CreateAhatH(A_test)

AhatH_train = CreateAdjacencyMatrixWithAhatH2(X_train)

# np.save("A_train.npy", A_train)
np.save("AhatH_train_neighborDistance.npy", AhatH_train)
np.save("points_train_neighborDistance.npy", X_train)
# np.save("labels_train.npy", train_labels)
