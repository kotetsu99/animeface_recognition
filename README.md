# 概要

ラブライブ！のソーシャルゲーム「[スクールアイドルフェスティバル ALL STARS](https://lovelive-as.bushimo.jp/)」、「[ラブライブ！虹ヶ咲学園スクールアイドル同好会](http://www.lovelive-anime.jp/nijigasaki/)」に登場する
メンバーの顔を学習させ、画像から彼女たちの顔を検出、さらに名前を併記するAI（Python）プログラムです。
一般的なアニメキャラの顔認識にも応用可能です。



# 使用方法・解説

こちらのブログ記事をご確認ください。

・使用方法：
https://nine-num-98.blogspot.com/2019/12/ai-lovelive-01.html
https://nine-num-98.blogspot.com/2020/12/anigasaki-ai-01.html

・解説：
https://nine-num-98.blogspot.com/2019/12/ai-lovelive-02.html



# 留意事項

・本プログラム群の中には、データセットファイル（スクリーンショット画像など）、学習済みモデルは含まれておりません。学習・認識テスト用のデータセットファイルは各自でご用意のうえ、本プログラムをご使用ください。

・OpenCVのカスケード分類器として使用したファイル（lbpcascade_animeface.xml）は以下からお借りしました。

https://github.com/nagadomi/lbpcascade_animeface
