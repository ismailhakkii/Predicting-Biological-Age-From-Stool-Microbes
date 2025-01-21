import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel
import threading
import os


class MicrobiomeAgePredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("Mikrobiom Yaş Tahmini")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')

        # Ana Frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Başlık
        ttk.Label(main_frame,
                  text="Mikrobiom Yaş Tahmini Modeli",
                  font=('Helvetica', 16, 'bold')).pack(pady=10)

        # Progress Bar Frame
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=5)

        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(progress_frame,
                                        variable=self.progress_var,
                                        length=400)
        self.progress.pack()

        self.status_var = tk.StringVar(value="Hazır")
        ttk.Label(progress_frame,
                  textvariable=self.status_var).pack()

        # Buton Frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)

        self.train_button = ttk.Button(button_frame,
                                       text="Modeli Eğit",
                                       command=self.start_training,
                                       width=20)
        self.train_button.pack()

        # Sonuçlar Frame
        results_frame = ttk.LabelFrame(main_frame, text="Sonuçlar")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.result_text = scrolledtext.ScrolledText(results_frame,
                                                     height=10,
                                                     width=70)
        self.result_text.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        # Grafikler Frame
        plot_frame = ttk.LabelFrame(main_frame, text="Grafikler")
        plot_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.fig = Figure(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_status(self, message, progress):
        self.status_var.set(message)
        self.progress_var.set(progress)
        self.root.update()

    def start_training(self):
        self.train_button.config(state='disabled')
        threading.Thread(target=self.train_model, daemon=True).start()

    def train_model(self):
        try:
            self.result_text.delete(1.0, tk.END)
            self.update_status("Veriler yükleniyor...", 5)

            # Veri yükleme
            ages_df = pd.read_csv('Data/Ages.csv')
            data_df = pd.read_csv('Data/data.csv')

            self.update_status("Veriler ön işleniyor...", 10)
            if 'Sample Accession.1' in data_df.columns:
                data_df = data_df.drop('Sample Accession.1', axis=1)

            merged_df = pd.merge(data_df, ages_df, on='Sample Accession')
            X = merged_df.drop(['Sample Accession', 'Age'], axis=1)
            y = merged_df['Age']

            self.update_status("Veri seti bölünüyor...", 20)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            self.update_status("Özellik seçimi yapılıyor...", 30)
            # İlk aşama: Özellik seçimi için basit bir RandomForest
            pre_selector = RandomForestRegressor(n_estimators=100, random_state=42)
            pre_selector.fit(X_train, y_train)

            # Önemli özellikleri seç
            selector = SelectFromModel(pre_selector, prefit=True, threshold='mean')
            X_train_selected = selector.transform(X_train)
            X_test_selected = selector.transform(X_test)

            self.update_status("Grid Search hazırlanıyor...", 40)
            # Random Forest için parametre grid'i
            rf_params = {
                'n_estimators': [200, 300],
                'max_depth': [20, 30, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2']
            }

            # Gradient Boosting için parametre grid'i
            gb_params = {
                'n_estimators': [200, 300],
                'learning_rate': [0.05, 0.1],
                'max_depth': [5, 7],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }

            self.update_status("Model seçimi yapılıyor...", 50)
            # Random Forest ve Gradient Boosting modellerini karşılaştır
            rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
            gb_model = GradientBoostingRegressor(random_state=42)

            rf_grid = GridSearchCV(rf_model, rf_params, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
            gb_grid = GridSearchCV(gb_model, gb_params, cv=5, scoring='neg_mean_absolute_error')

            self.update_status("Random Forest eğitiliyor...", 60)
            rf_grid.fit(X_train_selected, y_train)
            rf_score = -rf_grid.best_score_

            self.update_status("Gradient Boosting eğitiliyor...", 70)
            gb_grid.fit(X_train_selected, y_train)
            gb_score = -gb_grid.best_score_

            # En iyi modeli seç
            if rf_score < gb_score:
                best_model = rf_grid.best_estimator_
                model_name = "Random Forest"
            else:
                best_model = gb_grid.best_estimator_
                model_name = "Gradient Boosting"

            self.update_status("Final model eğitiliyor...", 80)
            best_model.fit(X_train_selected, y_train)
            y_pred = best_model.predict(X_test_selected)

            # Performans metrikleri
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Cross-validation sonuçları
            cv_scores = cross_val_score(best_model, X_train_selected, y_train,
                                        cv=5, scoring='neg_mean_absolute_error')
            cv_mae = -cv_scores.mean()

            # Sonuçları göster
            self.result_text.insert(tk.END, "=== Model Sonuçları ===\n\n")
            self.result_text.insert(tk.END, f"Seçilen Model: {model_name}\n")
            self.result_text.insert(tk.END, f"MAE: {mae:.2f} yıl\n")
            self.result_text.insert(tk.END, f"R-squared: {r2:.3f}\n")
            self.result_text.insert(tk.END, f"Cross-Validation MAE: {cv_mae:.2f} yıl\n\n")

            # En önemli özellikleri göster
            feature_names = X.columns[selector.get_support()]
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': best_model.feature_importances_
            })
            top_features = feature_importance.nlargest(10, 'importance')

            self.result_text.insert(tk.END, "En Önemli 10 Mikroorganizma:\n")
            for idx, row in top_features.iterrows():
                self.result_text.insert(tk.END, f"{row['feature']}: {row['importance']:.4f}\n")

            # Grafik
            self.update_status("Grafikler oluşturuluyor...", 90)
            self.fig.clear()

            # Ana grafik
            ax = self.fig.add_subplot(111)
            scatter = ax.scatter(y_test, y_pred, alpha=0.5, c='blue', label='Tahminler')
            ax.plot([y_test.min(), y_test.max()],
                    [y_test.min(), y_test.max()],
                    'r--', lw=2,
                    label='Perfect Prediction')

            ax.set_xlabel('Gerçek Yaş')
            ax.set_ylabel('Tahmin Edilen Yaş')
            ax.set_title('Gerçek vs Tahmin Edilen Yaşlar')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()

            # İstatistikleri grafik üzerine ekle
            stats_text = (f'Model: {model_name}\n'
                          f'MAE: {mae:.2f} yıl\n'
                          f'R²: {r2:.3f}\n'
                          f'CV MAE: {cv_mae:.2f} yıl')
            ax.text(0.05, 0.95, stats_text,
                    transform=ax.transAxes,
                    bbox=dict(facecolor='white', alpha=0.8),
                    verticalalignment='top')

            self.fig.tight_layout()
            self.canvas.draw()

            self.update_status("Eğitim tamamlandı!", 100)

        except Exception as e:
            self.result_text.insert(tk.END, f"\nHata oluştu: {str(e)}")
            self.update_status("Hata oluştu!", 0)
            messagebox.showerror("Hata", str(e))
        finally:
            self.train_button.config(state='normal')


if __name__ == "__main__":
    root = tk.Tk()
    app = MicrobiomeAgePredictor(root)
    root.mainloop()