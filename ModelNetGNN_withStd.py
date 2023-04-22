# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 21:10:23 2023

@author: ilngy
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import random
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

root = os.getcwd()
os.chdir(r"G:\マイドライブ\python\3Dモデル\GNN\graphData")
points_train = np.load("points_train_neighborDistance.npy")
AhatH_train = np.load("AhatH_train_neighborDistance.npy")
os.chdir(root)
samplingPoints=points_train.shape[1]

ss = StandardScaler()
points_train = ss.fit_transform(points_train)

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

    # ax.set_xlim(-2,2)
    # ax.set_ylim(-2,2)
    # ax.set_zlim(-2,2)

    plt.show()

dropRate=0.0

pointsInput = layers.Input(shape=(samplingPoints, 3))
AhatHInput = layers.Input(shape=(samplingPoints, samplingPoints))
x = pointsInput

def OneCycle(x, layerNum, withAhatH):
    if withAhatH == True:
        x = tf.matmul(AhatHInput, x)
    x = layers.Dense(layerNum)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(rate=dropRate)(x)
    return x

x = OneCycle(x, 32, True)

pointsOutput = layers.Dense(3)(x)

model = Model(
    inputs=[pointsInput, AhatHInput],
    outputs=pointsOutput)

model.compile(
    loss="mse",
    metrics=["mse"],
    optimizer=Adam(learning_rate=0.001))#learning rate defaults to 0.001
model.summary(expand_nested=True)

def RunTrain():
    history = model.fit(
        [points_train, AhatH_train], points_train,
        batch_size=64,
        epochs=100,
        validation_split=0.2,
        shuffle=True,
        steps_per_epoch=None,
        validation_freq=1,
        verbose=0)

    plt.plot(history.history['mse'], label="train")
    plt.plot(history.history['val_mse'], label="test")
    plt.legend()
    plt.show()

    model.save(modelName)

def RunPredict(points_train):
    model.load_weights(modelName)
    pred = model.predict([points_train[:20], AhatH_train[:20]])

    # 標準化戻し
    points_train = ss.inverse_transform(points_train)
    pred = ss.inverse_transform(pred)

    for i in range(20):
        plot3D(points_train[i,:,0], points_train[i,:,1], points_train[i,:,2])
        plot3D(pred[i,:,0], pred[i,:,1], pred[i,:,2])

modelName = "model.h5"
RunTrain()
RunPredict(points_train)
