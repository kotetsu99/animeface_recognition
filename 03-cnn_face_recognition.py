#!/usr/bin/env python
# -*- coding: utf-8 -*-

import keras
import numpy as np
import cv2
import os, sys

# OpenCV検出サイズ定義
cv_width, cv_height = 64, 64
# OpenCV検出閾値
minN = 15
# 顔画像サイズ定義
img_width, img_height = 64, 64
# 学習用データセットのディレクトリパス
train_data_dir = 'dataset/02-face'
# テスト用データセットのディレクトリパス
test_data_dir = 'dataset/03-test'
# 顔検出結果データ保存先ディレクトリパス
rec_data_dir = 'dataset/04-rec'
# データセットのサブディレクトリ名（クラス名）を取得
classes = os.listdir(train_data_dir)
# 顔検出用カスケードxmlファイルパス定義
cascade_xml = "lbpcascade_animeface.xml"

def main():

    # 環境設定(ディスプレイの出力先をlocalhostにする)
    os.environ['DISPLAY'] = ':0'

    #print 'クラス名リスト = ', classes

    # 学習済ファイルの確認
    if len(sys.argv)==1:
        print('使用法: python cnn_test.py 学習済ファイル名.h5')
        sys.exit()
    savefile = sys.argv[1]
    # モデルのロード
    model = keras.models.load_model(savefile)

    # テスト用画像取得
    test_imagelist = os.listdir(test_data_dir)
    # テスト画像に対し顔検出
    for test_image in test_imagelist:
        # 画像ファイル読み込み
        file_name = os.path.join(test_data_dir, test_image)
        print(file_name)
        image = cv2.imread(file_name)
        if image is None:
            print("Not open:",file_name)
            continue

        # 顔検出実行
        rec_image = detect_face(image, model)

        # 結果をファイルに保存
        rec_file_name = os.path.join(rec_data_dir, "rec-" + test_image)
        cv2.imwrite(rec_file_name, rec_image)


def detect_face(image, model):
    # グレースケール画像に変換
    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cascade_xml)

    # 顔検出の実行
    face_list=cascade.detectMultiScale(image_gs, scaleFactor=1.01, minNeighbors=minN, minSize=(cv_width,cv_height), maxSize=(512,512))

    # 顔が1つ以上検出された場合
    if len(face_list) > 0:
        for rect in face_list:
            # 顔画像を生成
            face_img = image[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
            if face_img.shape[0] < img_width or face_img.shape[1] < img_height:
                print("too small")
                continue
            # 顔画像とサイズを定義
            face_img = cv2.resize(face_img, (img_width, img_height))

            # Keras向けにBGR->RGB変換、float型変換
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB).astype(np.float32)
            # 顔画像をAIに認識
            name = predict_who(face_img, model)

            # 顔近傍に矩形描画
            cv2.rectangle(image, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (0, 0, 255), thickness = 3)
            # AIの認識結果(キャラ名)を元画像に矩形付きで表示
            x, y, width, height = rect
            cv2.putText(image, name, (x, y + height + 60), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255) ,2)
    # 顔が検出されなかった場合
    else:
        print("no face")
    return image


def predict_who(x, model):
    # 画像データをテンソル整形
    x = np.expand_dims(x, axis=0)
    # データ正規化
    x = x / 255
    pred = model.predict(x)[0]

    # 確率が高い上位3キャラを出力
    top = 3
    top_indices = pred.argsort()[-top:][::-1]
    result = [(classes[i], pred[i]) for i in top_indices]
    print(result)
    print('=======================================')

    # 1番予測確率が高いキャラ名を返す
    return result[0][0]


if __name__ == '__main__':
    main()
