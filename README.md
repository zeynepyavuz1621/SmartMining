# Smart Mining 

Mayın İşletmesi Emtialarında (Commodity ETF) Ortalama Dönüş Stratejisini Kullanan Gelişmiş Portföy Optimizasyon ve Backtesting Platformu

**YAP 471 Hesaplamalı Finans Dönem Projesi**  
**Takım: MetalMinds**

---

##  Proje Özeti

Smart Mining, bakır, lityum, nikel, alüminyum, gümüş, platin, paladyum ve altın gibi kritik emtiaların ETF'lerine dayalı otomatik ticari sinyaller üreten ve portföy performansını optimize etmeye yardımcı olan bir web tabanlı finansal analiz platformudur.

### Ana Özellikler

- **Veri İndirme & Temizleme**: Yahoo Finance'den gerçek zamanlı emtia ETF verilerini otomatik indirme
- **Sinyal Üretimi**: Z-score temelli ortalama dönüş stratejisi (%90 güven aralığında)
- **Portföy Yönetimi**: Risk-uygun varlık tahsisi ve yeniden dengeleme
- **Risk Yönetimi**: Stop-loss emirleri ve portföy düşüş takibi
- **Backtesting**: Tarihsel verilerde strateji performansını test etme ve karşılaştırma
- **İnteraktif Web Arayüzü**: Streamlit ile modern, kullanıcı dostu dashboard

---

##  Veri Kaynağı

Proje aşağıdaki emtia ETF'lerini kullanır:

| Ticker | İsim | Açıklama |
|--------|------|----------|
| CPER   | Bakır | Copper ETF |
| LIT    | Lityum | Lithium ETF |
| JJN    | Nikel | Nickel ETF |
| JJU    | Alüminyum | Aluminum ETF |
| SLV    | Gümüş | Silver ETF |
| PPLT   | Platin | Platinum ETF |
| PALL   | Paladyum | Palladium ETF |
| GLD    | Altın (Kıyaslama) | Gold ETF - Benchmark |

---

##  Kurulum

### Gereksinimler

- Python 3.8+
- PIP paket yöneticisi
- İnternet bağlantısı (veri indirme için)

### Adımlar

1. **Repoyu klonlayın veya projeyi indirin**
   ```bash
   cd SmartMining
   ```

2. **Virtual ortam oluşturun (önerilir)**
   ```bash
   python -m venv .venv
   # Windows için:
   .venv\Scripts\activate
   # Linux/Mac için:
   source .venv/bin/activate
   ```

3. **Gerekli paketleri yükleyin**
   ```bash
   pip install -r requirements.txt
   ```

---

##  Kullanım

### Uygulamayı Başlatma

```bash
streamlit run app.py
```

Tarayıcınız otomatik olarak `http://localhost:8501` adresinde açılacaktır.

### Web Arayüzü Özellikleri

**Sol Sidebar'da:**
- Başlangıç ve bitiş tarihleri seçimi
- Lookback penceresi (sinyal hesaplaması için)
- Tahmin penceresi (volatilite tahminlemesi)
- Yeniden dengeleme sıklığı
- Risk iştahı ayarı
- Stop-loss seviyesi
- İşlem soğutma dönem ayarı

**Ana Sekmelerde:**
1. **Sinyal Analizi**: Gerçek zamanlı alım/satım sinyalleri
2. **Portföy Analizi**: Varlık tahsisi ve ağırlıklandırması
3. **Risk Özeti**: Portföy riskini ve sınırlı kayıpları analiz etme
4. **Backtest Sonuçları**: Strateji performansı ve metrikler

---

##  Proje Yapısı

```
SmartMining/
├── app.py                  # Ana Streamlit uygulaması
├── data_module.py          # Veri indirme ve işleme
├── signal_module.py        # Alım/satım sinyali üretimi (z-score)
├── portfolio_module.py     # Portföy optimizasyonu ve tahsisi
├── risk_module.py          # Risk yönetimi ve stop-loss
├── backtest_module.py      # Tarihsel backtesting motoru
├── requirements.txt        # Python bağımlılıkları
└── README.md              # Bu dosya
```

### Modül Açıklamaları

#### `data_module.py`
- `download_prices()`: Yahoo Finance'den veri indir
- `compute_returns()`: Günlük getirileri hesapla
- `get_data_summary()`: Veri istatistiklerini al

#### `signal_module.py`
- `generate_signals()`: Z-score temelli sinyaller üret
- `compute_zscore()`: Fiyat z-score'unu hesapla
- **Eşik**: Z > 1.645 (SATIŞ), Z < -1.645 (ALIŞ), Else (BEKLE)

#### `portfolio_module.py`
- `build_portfolio()`: Optimum varlık ağırlıklandırması
- Ortalama-varyans optimizasyonu kullanır

#### `risk_module.py`
- `apply_stop_loss()`: Stop-loss emirleri uygula
- `compute_portfolio_drawdown()`: Maksimum düşüş hesapla

#### `backtest_module.py`
- `run_backtest()`: Tarihsel performansı simüle et
- `compute_metrics()`: Sharpe oranı, toplam getiri vb.
- `equal_weight_benchmark()`: Eşit ağırlık stratejisiyle karşılaştır

---

##  Varsayılan Parametreler

```python
Başlangıç Tarihi:     2021-01-04
Bitiş Tarihi:         2025-12-31
Lookback Penceresi:   20 gün
Tahmin Penceresi:     60 gün
Yeniden Dengeleme:    Her 5 günde bir
Risk İştahı:          1.0
Stop-Loss:            %10
Soğutma Dönem:        5 gün
Başlangıç Nakidi:     $100,000
```

---

##  Strateji Açıklaması

### Ortalama Dönüş (Mean Reversion) Stratejisi

1. **Z-Score Hesaplaması**
   - Her emtianın fiyatının, 20 günlük hareketli ortalamadan kaç standart sapma uzak olduğunu ölçer

2. **Sinyal Üretimi**
   - **SATIŞ**: Z > 1.645 (Fiyat istatistiksel olarak çok yüksek → düşüş beklenir)
   - **ALIŞ**: Z < -1.645 (Fiyat istatistiksel olarak çok düşük → yükseliş beklenir)
   - **BEKLE**: Diğer durumlarda

3. **Portföy Optimizasyonu**
   - Alış sinyallerine sahip varlıklara daha fazla ağırlık
   - Risk iştahı parametresine dayalı pozisyon boyutu

4. **Risk Yönetimi**
   - Stop-loss seviyeleri ({10}_başlangıçta)
   - Portföy düşüş takibi
   - Yeniden dengeleme tüm riskini kontrol etmek için

---

## 📊 Temel Metrikler

- **Sharpe Oranı**: Riskin uyarlanmış getiri performansı
- **Maksimum Düşüş (Max Drawdown)**: En kötü kaybeden başlangıç durumundan
- **Toplam Getiri**: Başlangıçtan bitiş dönemi getirisi
- **Yıllık Volatilite**: Getiri dalgalanmaları
- **Win Rate**: Karlı işlemlerin yüzdesi

---

## 🔧 Gereksinimler

| Teknoloji | Versiyon | Kullanım |
|-----------|----------|----------|
| Python | 3.8+ | Programlama dili |
| Streamlit | ≥1.32.0 | Web framework |
| Pandas | ≥2.0.0 | Veri işleme |
| NumPy | ≥1.24.0 | Sayısal hesaplamalar |
| SciPy | ≥1.11.0 | İstatistik/optimizasyon |
| yfinance | ≥0.2.38 | Veri indirme |
| Plotly | ≥5.19.0 | Interaktif grafikler |

---

## 📝 Notlar

- Tüm tarihsel veriler Yahoo Finance'den alınmaktadır
- Backtesting gerçek işlem maliyetleri (komisyon, slippage) içermez
- Z-score eşiği (%90 güven aralığı) sabitlenmiştir
- Strateji eğitim amaçlıdır, gerçek para yatırımı için danışman konsultasyonu önerilir


