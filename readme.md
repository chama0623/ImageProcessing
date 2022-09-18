# ImageProcessing Repository

## About
画像処理に関する実装をまとめたリポジトリです.

## 実装内容
画像処理
- グレースケール化  
- 二値化, 大津の二値化  
- HSV変換, HSV逆変換  
- 減色処理  
- Average Pooling, Max Pooling  
- フィルタ処理(ガウシアン, メディアン, 平滑化, モーション, MAX-MIN, 微分, Sobel, Prewitt, Laplacian, Emboss, LoG)  
- ヒストグラムの処理(表示, 正規化, 操作, 平坦化)  
- 補正・補間(ガンマ補正, 最近傍補間, Bi-linear補間, Bi-cubic補間)  
- アフィン変換(平行移動, 拡大縮小, 回転, スキュー)  
- フーリエ変換(ローパスフィルタ, ハイパスフィルタ)  
- Cannyエッジ検出  
- Hough変換, 直線検出   
- モルフォロジー処理(膨張, 収縮), オープニング処理, クロージング処理, モルフォロジー勾配  
- トップハット変換, ブラックハット変換  
- テンプレートマッチング(SSD, NCC,ZNCC)  
- ラベリング(4近傍, 8近傍)  
- 4連結数, 8連結数  
- 細線化, ヒルディッチの細線化, Zhang-Suenの細線化  
- HOG特徴量の計算  
- マスキング
- ピラミッド差分による高周波成分の抽出, ガウシアンピラミッド  
- 顕著性マップ  
- ガボールフィルタ(回転, エッジ検出, 特徴抽出)  
- Hessianのコーナー検出  

画像認識
- Lenet  
- AlexNet  
- GoogLeNet  
- VGGNet  
- ResNet  
- Xception  

物体検出
- HOG特徴量とSVMによる自動車検出  
- SSDの実装

## Reference
画像処理100本ノック https://github.com/minido/Gasyori100knock-1  
ディジタル画像処理[改定第2版]
画像認識 原田達也 著  
物体検出とGAN, オートエンコーダー, 画像処理入門 チーム・カルポ