# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 14:41:58 2022
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import \
    Input, Flatten, Dense, Conv2D, BatchNormalization, LeakyReLU, Dropout, \
    Reshape, Conv2DTranspose, Activation, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K  # tensorflowならtensorflowと同様
import matplotlib.pyplot as plt

#cifar100
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='coarse')
x_train = x_train/255

imgSize = x_train.shape[1:]

class Sampling(Layer):#tensorflowのLayerを継承
    def call(self, inputs):
        mu, log_var = inputs
        epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
        return mu + K.exp(log_var / 2) * epsilon

def r_loss(y_true, y_pred):
    # axisを指定することで各データごとの平均が出せる
    return K.mean(K.square(y_true-y_pred), axis=[1, 2])

r_loss_factor = 10000
def vae_loss(y_true, y_pred):#損失関数はdef f(true, pred)の形、順序にする。計算にはtfを用いる。
    z_mean, z_log_var, z = encoder(y_true)#関数内で変数を導出しないで、外部の変数を読み込むとエラー。encoderは関数の引数になくてOK。
    y_pred = decoder(z)# y_pred = model(y_true)や、引数のy_predはサンプリング結果が異なるためNG!

    reconstruction_loss = tf.reduce_mean(
        tf.square(y_true - y_pred), axis=[1,2,3])#MSE。(batch,w,h,ch), batch毎に計算するので、カラーの場合は[1,2,3], 白黒なら[1, 2]
    reconstruction_loss *= r_loss_factor
    reconstruction_loss = tf.cast(reconstruction_loss, dtype=tf.float32)

    kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    kl_loss = tf.reduce_sum(kl_loss, axis=1)#(batch_num, z_dim), batch毎に計算するので、axis=1
    kl_loss *= -0.5

    total_loss = reconstruction_loss + kl_loss

    return total_loss

def Reconst_loss(y_true, y_pred):#評価関数用。zはサンプリングせず平均で評価。
    z_mean, z_log_var, z = encoder(y_true)
    y_pred = decoder(z_mean)

    reconstruction_loss = tf.reduce_mean(
        tf.square(y_true - y_pred), axis=[1,2,3])
    reconstruction_loss *= r_loss_factor
    reconstruction_loss = tf.cast(reconstruction_loss, dtype=tf.float32)

    return reconstruction_loss

def Kl_loss(y_true, y_pred):#評価関数用。
    z_mean, z_log_var, z = encoder(y_true)

    kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    kl_loss = tf.reduce_sum(kl_loss, axis=1)#(batch_num, z_dim), batch毎に計算するので、axis=1
    kl_loss *= -0.5

    return kl_loss

z_dim = 256
dropRate=0.5
# encoder
encoder_input = Input(shape=imgSize, name="encoder_input")

x = Conv2D(
    filters=32,  # フィルターの数。層が深くなる。
    kernel_size=3,  # カラーの場合は各フィルターのサイズが(s×s×3)
    strides=2,  # フィルターを移動させるステップサイズ。2以上だと出力テンソルのサイズは小さくなる
    padding="same")(encoder_input)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Dropout(rate=dropRate)(x)

x = Conv2D(
    filters=64,
    kernel_size=3,  # 各フィルターのサイズが(s×s×前の層のフィルター数)
    strides=2,
    padding="same")(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Dropout(rate=dropRate)(x)

x = Conv2D(
    filters=64,
    kernel_size=3,
    strides=2,
    padding="same")(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Dropout(rate=dropRate)(x)

x = Conv2D(
    filters=64,
    kernel_size=3,
    strides=2,
    padding="same")(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Dropout(rate=dropRate)(x)

shape_before_flattening = K.int_shape(x)[1:]
x = Flatten()(x)

z_mean = Dense(z_dim, name='mu')(x)
z_log_var = Dense(z_dim, name='log_var')(x)
z = Sampling(name='encoder_output')([z_mean, z_log_var])

encoder = Model(encoder_input, [z_mean, z_log_var, z])#複数出力はリストでOK


# decoder
decoder_input = Input(shape=(z_dim, ), name="decoder_input")

x = Dense(np.prod(shape_before_flattening))(decoder_input)
x = Reshape(shape_before_flattening)(x)

x = Conv2DTranspose(
    filters=64,
    kernel_size=3,
    strides=2,
    padding="same")(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Dropout(rate=dropRate)(x)

x = Conv2DTranspose(
    filters=64,
    kernel_size=3,
    strides=2,
    padding="same")(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Dropout(rate=dropRate)(x)

x = Conv2DTranspose(
    filters=32,
    kernel_size=3,
    strides=2,
    padding="same")(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Dropout(rate=dropRate)(x)

x = Conv2DTranspose(
    filters=3,
    kernel_size=3,
    strides=2,
    padding="same")(x)
decoder_output = Activation("sigmoid")(x)

decoder = Model(decoder_input, decoder_output)


model = Model(encoder_input, decoder(z))#decoderを一つの層として扱う。output=decoder(z)のoutputがModelのoutput。
model.summary(expand_nested=True)

optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer,
              loss=vae_loss,
              metrics=[Reconst_loss, Kl_loss])

history = model.fit(
    x_train,
    x_train,
    batch_size=256,
    shuffle=True,
    epochs=1000,
    verbose=2)

plt.plot(history.history["loss"], label="loss")
plt.plot(history.history["Reconst_loss"], label="Reconst_loss")
plt.plot(history.history["Kl_loss"], label="Kl_loss")
plt.xlabel('Epoch')
plt.ylabel('Metrics')
plt.yscale("log")
plt.legend()
plt.ylim(10,)
plt.show()

pred = model.predict(x_train)
pred_z_mean, pred_z_log_var, pred_z = encoder.predict(x_train)

plt.hist(x_train.reshape(-1), label="true", alpha=0.5)
plt.hist(pred.reshape(-1), label="pred", alpha=0.5)
plt.legend()
plt.show()

from sklearn.decomposition import PCA
pca = PCA()
pred_z_pca = pca.fit_transform(pred_z)

# reconstructing
for i in range(5):
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(x_train[i,:,:,:])
    ax[1].imshow(pred[i,:,:,:])
    plt.show()

# Newly generated
def NewGenerate(point):
    fig, ax = plt.subplots(1,2, figsize=(8,4))
    ax[1].imshow(decoder.predict(point)[0])

    ax[0].scatter(pred_z_pca[:, 0], pred_z_pca[:, 1], c=y_train, cmap='tab20', s=5)
    point_pca = pca.transform(point)
    ax[0].scatter(point_pca[:, 0], point_pca[:, 1], marker="*", s=50, c="black")
    ax[0].grid()
    plt.show()

for num in range(10):
    point = np.array([])
    for i in range(z_dim):
        point = np.append(point, np.random.randn())
    NewGenerate(point.reshape(1,-1))

# Latent space distribution
from scipy.stats import norm
x = np.linspace(-3, 3, 100)
fig = plt.figure(figsize=(20, 20))
fig.subplots_adjust(hspace=0.6, wspace=0.4)
for i in range(min(z_dim, 50)):
    ax = fig.add_subplot(5, 10, i+1)
    ax.hist(pred_z[:,i], density=True, bins = 20)
    ax.axis('off')
    ax.text(0.5, -0.35, str(i), fontsize=10, ha='center', transform=ax.transAxes)
    ax.plot(x,norm.pdf(x))
plt.show()
