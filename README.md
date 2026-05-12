# AI MLOps - Vehicle Detection with Active Learning

Canlı sehir kamerasından arac tespiti yapan, dusuk guvenilirlikli tahminleri otomatik kaydeden ve yeterli veri birikince modeli fine-tune eden bir Active Learning pipeline'i. Tum surec MLOps araclariyla entegre.

## Proje Yapisi


ai-mlops/
├── src/
│   ├── stream.py         # Kameradan frame cekme
│   ├── detect.py         # YOLOv8 ile arac detection
│   ├── collector.py      # Dusuk confidence frame kaydetme
│   ├── labeller.py       # Frame etiketleme arayuzu
│   ├── finetune.py       # Model fine-tune pipeline
│   ├── monitor.py        # Model performans takibi
│   └── pipeline.py       # Tum akisi birlestiren ana script
├── api/
│   └── app.py            # FastAPI - detection API
├── configs/
│   └── config.yaml       # Tum parametreler
├── tests/                # pytest testleri
├── .github/workflows/    # CI/CD pipeline
├── Dockerfile
└── requirements.txt


## Kurulum

```bash
git clone https://github.com/Daryl61/ai-mlops-vehicle-detection.git
cd ai-mlops-vehicle-detection
python -m venv venv
venv\Scripts\activate        
pip install -r requirements.txt
```

GPU kullanmak icin (CUDA):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Kullanim

### Pipeline (Canli Detection + Active Learning)
```bash
python src/pipeline.py
```
- Canli kameradan arac tespiti yapar
- Dusuk confidence frame'leri otomatik kaydeder
- `q` ile cikis

### Etiketleme
```bash
python src/labeller.py
```
- Toplanan frame'leri gosterir
- `y` = onayla, `n` = atla, `q` = cik

### Fine-tune
```bash
python src/finetune.py
```
- Yeterli etiketli veri varsa modeli fine-tune eder
- MLflow'a metrikleri loglar

### API
```bash
uvicorn api.app:app --reload
```
- `GET /health` - Saglik kontrolu
- `POST /detect` - Resim yukle, detection sonucu al
- `GET /detect-live` - Canli kameradan anlik detection
- `GET /stats` - Istatistikler

### Monitoring
```bash
python src/monitor.py
```

### MLflow UI
```bash
mlflow ui
```
Tarayicide `localhost:5000` ac.

## Teknoloji Stack

| Arac | Amac |
|------|------|
| YOLOv8 | Arac detection + fine-tune |
| OpenCV | Kamera stream |
| MLflow | Experiment tracking |
| FastAPI | Model serving API |
| GitHub Actions | CI/CD |
| pytest | Testing |
| Docker | Containerization |

## Akis

```
Kamera --> Frame al --> YOLO detect --> Confidence kontrol
                                           |
                              Yuksek (>0.7): logla, devam
                              Dusuk (<0.7): frame kaydet
                                           |
                              Yeterli veri --> Etiketle --> Fine-tune --> Yeni model
```
