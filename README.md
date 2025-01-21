# Microbiome-Age-Predictor

## Proje Hakkında
Bu proje, bağırsak mikrobiyom verilerini kullanarak kişilerin biyolojik yaşını tahmin eden bir makine öğrenmesi modeli geliştirmektedir.

## Proje İçeriği
Dışkıdaki mikropların DNA örneklerinden kişinin biyolojik yaşını tahmin etme projesidir.
Makine öğrenmesi kullanılarak gerçekleştirilmiştir. 
İntroduction to Pattern Recognition dersi final ödevidir.

## Veri Seti
- 4274 kişinin bağırsak mikrobu kompozisyon verisi
- Ages.csv: Örnek isimleri ve kişilerin yaşları
- data.csv: Her kişi için 3200 farklı mikroorganizmanın DNA parça sayıları

## Proje Yapısı
Microbiome-Age-Predictor/
│
├── Data/
│   ├── Ages.csv
│   └── data.csv
│
├── main.py
├── age_prediction_scatter.png
└── README.md

## Kullanılan Kütüphaneler
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Ekran Görüntüleri ve Performans Metrikleri

### Random Forest:

![image](https://github.com/user-attachments/assets/74c91f01-c253-4324-9d3c-eb00983a361f)
