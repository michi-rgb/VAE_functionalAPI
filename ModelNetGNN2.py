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

train_points = np.load("train_points.npy")
test_points = np.load("test_points.npy")
train_labels = np.load("train_labels.npy")
test_labels = np.load("test_labels.npy")
CLASS_MAP = np.load("CLASS_MAP.npy", allow_pickle=True).tolist()
samplingPoints=512

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


# 隣接行列
train_A = np.load("A_train.npy")
# test_A = np.load("A_test.npy")

calculateAdjacencyMatrix = True
if calculateAdjacencyMatrix == False:
    A_hat_H_all = np.load("A_hat_H_all.npy")
else:
    A_plus_I = train_A + np.eye(samplingPoints)
    D_pre = np.sum(A_plus_I, axis=1)
    A_hat_H_all = []
    for i in range(D_pre.shape[0]):
        Di = np.diag(D_pre[i,:])
        Di = np.linalg.inv(Di)
        A_hat_i = np.dot(Di**0.5, A_plus_I[i,:,:])
        A_hat_i = np.dot(A_hat_i, Di**0.5)
        A_hat_H = np.dot(A_hat_i, train_points[i,:,:])
        A_hat_H_all.append(A_hat_H)
    A_hat_H_all = np.array(A_hat_H_all)
    np.save("A_hat_H_all.npy", A_hat_H_all)



dropRate=0.1
# def EncoderEdge():
inputsEncoderEdge = layers.Input(shape=A_hat_H_all.shape[1:])

x = layers.Conv1D(32, kernel_size=1, padding="valid")(inputsEncoderEdge)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Dropout(rate=dropRate)(x)

x = layers.Conv1D(32, kernel_size=1, padding="valid")(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Dropout(rate=dropRate)(x)

encodedEdge = layers.Conv1D(32, kernel_size=1, padding="valid")(x)



# def EncoderNode():
inputsEncoderNode = layers.Input(shape=train_points.shape[1:])

x = layers.Conv1D(32, kernel_size=1, padding="valid")(inputsEncoderNode)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Dropout(rate=dropRate)(x)

x = layers.Conv1D(32, kernel_size=1, padding="valid")(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Dropout(rate=dropRate)(x)

encodedNode = layers.Conv1D(32, kernel_size=1, padding="valid")(x)



# def ProcessorEdge():
inputsProcessorEdge = layers.Concatenate(axis=2)([encodedEdge,encodedNode])

x = layers.Conv1D(32, kernel_size=1, padding="valid")(inputsProcessorEdge)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Dropout(rate=dropRate)(x)

x = layers.Conv1D(32, kernel_size=1, padding="valid")(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Dropout(rate=dropRate)(x)

processedEdge = layers.Conv1D(32, kernel_size=1, padding="valid")(x)
encodedEdge = layers.Add()([encodedEdge, processedEdge])



# def ProcessorNode():
aggregatedEdge = tf.reduce_sum(processedEdge, axis=2, keepdims=True)

inputsProcessorEdge = layers.Concatenate(axis=2)([encodedNode, aggregatedEdge])

x = layers.Conv1D(32, kernel_size=1, padding="valid")(inputsProcessorEdge)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Dropout(rate=dropRate)(x)

x = layers.Conv1D(32, kernel_size=1, padding="valid")(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Dropout(rate=dropRate)(x)

processedNode = layers.Conv1D(32, kernel_size=1, padding="valid")(x)
encodedNode = layers.Add()([encodedNode, processedNode])



# def DecoderNode():
x = layers.Conv1D(32, kernel_size=1, padding="valid")(processedNode)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Dropout(rate=dropRate)(x)

x = layers.Conv1D(32, kernel_size=1, padding="valid")(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.Dropout(rate=dropRate)(x)

decodedNode = layers.Conv1D(3, kernel_size=1, padding="valid")(x)



model = Model(
    inputs=[inputsEncoderEdge, inputsEncoderNode],
    outputs=decodedNode)

model.compile(
    loss="mse",
    metrics=["mse"],
    optimizer=Adam(learning_rate=0.0005))#learning rate defaults to 0.001
model.summary(expand_nested=True)

history = model.fit(
    [A_hat_H_all, train_points], train_points,
    batch_size=64,
    epochs=100,
    validation_split=0.2,
    shuffle=True,
    steps_per_epoch=None,
    validation_freq=1,
    verbose=1)

plt.plot(history.history['mse'], label="train")
plt.plot(history.history['val_mse'], label="test")
plt.legend()
plt.show()

pred = model.predict([A_hat_H_all, train_points])

for i in range(3):
    plot3D(A_hat_H_all[i,:,0], A_hat_H_all[i,:,1], A_hat_H_all[i,:,2])
    plot3D(train_points[i,:,0], train_points[i,:,1], train_points[i,:,2])
    plot3D(pred[i,:,0], pred[i,:,1], pred[i,:,2])
