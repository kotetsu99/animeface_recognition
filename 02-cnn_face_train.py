#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import sys


# 画像リサイズ定義
img_width, img_height = 64, 64

# バッチサイズ
batch_size = 4
# エポック数（1エポックの画像サンプル数 = ステップ数 * バッチサイズ）
nb_epoch = 2

# 収束判定ループ（エポック）回数
#nb_patience = 10
nb_patience = nb_epoch
# 収束判定用差分パラメータ
val_min_delta = 0.001

# 学習用画像ディレクトリパス
train_data_dir = 'dataset/02-face/'


def main():

    # 環境設定(ディスプレイの出力先をlocalhostにする)
    os.environ['DISPLAY'] = ':0'

    # 時間計測開始
    start = time.time()

    # データセットのサブディレクトリ名（クラス名）を取得
    classes = os.listdir(train_data_dir)
    nb_classes = len(classes)
    #print 'クラス名リスト = ', classes

    # 学習済ファイルの確認
    if len(sys.argv)==1:
        print('使用法: python cnn_face_train.py モデルファイル名.h5')
        sys.exit()
    savefile = sys.argv[1]

    # モデル作成(既存モデルがある場合は読み込んで再学習。なければ新規作成)
    if os.path.exists(savefile):
        print('モデル再学習')
        cnn_model = keras.models.load_model(savefile)
    else:
        print('モデル新規作成')
        cnn_model = cnn_model_maker(nb_classes)
        # 多クラス分類を指定
        cnn_model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # 画像のジェネレータ生成
    train_generator, validation_generator = image_generator(classes)

    # 収束判定設定。以下の条件を満たすエポックがpatience回続いたら打切り。
    # val_loss(観測上最小値) - min_delta  < val_loss
    es_cb = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=val_min_delta, patience=nb_patience, verbose=1, mode='min')

    # CNN学習
    history = cnn_model.fit_generator(
        train_generator,
        epochs=nb_epoch,
        validation_data=validation_generator,
        callbacks=[es_cb])

    cnn_model.save(savefile)

    # 学習所要時間の計算、表示
    process_time = (time.time() - start) / 60
    print('process_time = ', process_time, '[min]')

    # 損失関数の時系列変化をグラフ表示
    plot_loss(history)


def cnn_model_maker(nb_classes):

    # 入力画像ベクトル定義（RGB画像認識のため、チャネル数=3）
    input_shape = (img_width, img_height, 3)

    # CNNモデル定義（Keras CIFAR-10: 9層ニューラルネットワーク）
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    return model


def image_generator(classes):
    # トレーニング画像データ生成準備
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        zoom_range=0.2,
        horizontal_flip=False,
        validation_split=0.1)

    # ディレクトリ内の学習用画像を読み込み、データ作成
    train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        color_mode='rgb',
        classes=classes,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True,
        subset = "training")

    # ディレクトリ内の評価用画像を読み込み、データ作成
    validation_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        color_mode='rgb',
        classes=classes,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True,
        subset = "validation")

    return (train_generator, validation_generator)


def plot_loss(history):
    # 損失関数のグラフの軸ラベルを設定
    plt.xlabel('time step')
    plt.ylabel('loss')

    # グラフ縦軸の範囲を0以上と定める
    plt.ylim(0, max(np.r_[history.history['val_loss'], history.history['loss']]))

    # 損失関数の時間変化を描画
    val_loss, = plt.plot(history.history['val_loss'], c='#56B4E9')
    loss, = plt.plot(history.history['loss'], c='#E69F00')

    # グラフの凡例（はんれい）を追加
    plt.legend([loss, val_loss], ['loss', 'val_loss'])

    # 描画したグラフを表示
    #plt.show()

    # グラフを保存
    plt.savefig('cnn_face_train_figure.png')


if __name__ == '__main__':
    main()
