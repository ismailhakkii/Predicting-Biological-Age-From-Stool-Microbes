README dosyasını daha profesyonel ve düzenli bir şekilde yeniden yapılandıracağım:

# Microbiome-Age-Predictor

## Proje Açıklaması
Bu proje, bağırsak mikrobiyom kompozisyon verilerini kullanarak kişilerin biyolojik yaşını tahmin eden bir makine öğrenmesi modeli geliştirmektedir. İntroduction to Pattern Recognition dersi final projesi kapsamında geliştirilmiştir.

## Veri Seti
Proje kapsamında kullanılan veri seti aşağıdaki özelliklere sahiptir:
- 4274 kişinin bağırsak mikrobiyal kompozisyon verisi
- Her birey için 3200 farklı mikroorganizmaya ait DNA parça sayıları
- İki ana dosya:
  - `Ages.csv`: Örnek isimleri ve kişilerin yaşları
  - `data.csv`: Mikroorganizma kompozisyon verileri

## Proje Yapısı
```
Microbiome-Age-Predictor/
│
├── Data/
│   ├── Ages.csv
│   └── data.csv
│
├── main.py
├── age_prediction_scatter.png
└── README.md
```

## Kullanılan Teknolojiler
- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Model Performansları

### Random Forest Modeli
- Eğitim Süresi: 105.56 saniye
- Performans Metrikleri:
  - MAE: 11.75 yıl
  - RMSE: 14.05 yıl
  - R-squared: 0.305
  - Cross-Validation MAE: 12.05 yıl

- En İyi Parametreler:
  - max_depth: None
  - max_features: sqrt
  - min_samples_leaf: 1
  - min_samples_split: 2
  - n_estimators: 300

![Random Forest Sonuçları](https://github.com/user-attachments/assets/74c91f01-c253-4324-9d3c-eb00983a361f)

### Gradient Boosting Modeli
- Eğitim Süresi: 2971.89 saniye
- Performans Metrikleri:
  - MAE: 10.53 yıl
  - RMSE: 13.14 yıl
  - R-squared: 0.392
  - Cross-Validation MAE: 11.42 yıl

- En İyi Parametreler:
  - learning_rate: 0.1
  - max_depth: 5
  - min_samples_leaf: 1
  - min_samples_split: 2
  - n_estimators: 300

![Gradient Boosting Sonuçları](https://github.com/user-attachments/assets/a85e1727-1b42-4cfc-8621-b0cdc4e03bb6)

## Sonuçlar
İki farklı model karşılaştırıldığında:
- Gradient Boosting modeli daha iyi performans göstermiştir (R-squared: 0.392)
- Ancak eğitim süresi Random Forest modeline göre yaklaşık 28 kat daha uzundur
- Her iki model de yaş tahmininde ortalama 10-12 yıl hata payına sahiptir
