# Ultralytics MCP Server

[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://docker.com)
[![N8N](https://img.shields.io/badge/N8N-Compatible-orange)](https://n8n.io)
[![MCP](https://img.shields.io/badge/MCP-Protocol-green)](https://modelcontextprotocol.io)
[![YOLO](https://img.shields.io/badge/YOLO-v11-red)](https://ultralytics.com)

Ultralytics YOLO modelleri için N8N entegrasyonu sağlayan **Model Context Protocol (MCP)** server implementasyonu. Docker container'ları üzerinden gerçek YOLO object detection, segmentation, classification ve pose estimation işlemlerini N8N workflow'larında kullanabilirsiniz.

## 🎯 Özellikler

### 🚀 AI & Machine Learning
- ✅ **Gerçek YOLO İşlemleri**: Mock değil, gerçek Ultralytics 8.3.171 container'da çalışan YOLO modelleri
- ✅ **Multi-Model Support**: YOLO v8, v9, v10, v11 + custom trained models desteği
- ✅ **Real-time Processing**: Base64 encoding ile güvenli Python execution
- ✅ **Custom Model Integration**: Kendi eğittiğiniz modelleri kullanabilme

### 🌐 Web Interfaces & Monitoring
- ✅ **Enhanced Streamlit UI**: Custom model scanning ve modern web arayüzü
- ✅ **Native TensorBoard**: Training süreçlerini görselleştirme
- ✅ **Jupyter Lab**: Interactive development environment
- ✅ **Auto-startup Services**: Container başladığında tüm servisler otomatik ayağa kalkar

### 🔗 Integration & DevOps
- ✅ **N8N Entegrasyonu**: SSE transport ile direct N8N workflow entegrasyonu
- ✅ **Docker Tabanlı**: İzole ve güvenli container environment
- ✅ **7 MCP Tools**: Comprehensive AI workflow automation
- ✅ **Error Handling**: Kapsamlı hata yönetimi ve logging
- ✅ **Health Monitoring**: Container durumu ve connection monitoring

### 📊 Solutions Framework
- ✅ **15+ Ready AI Apps**: Ultralytics Solutions ile hazır AI uygulamaları
- ✅ **Production Ready**: GPU acceleration ile yüksek performans

## 🚀 Hızlı Başlangıç

### Gereksinimler

- Docker & Docker Compose
- Node.js 18+
- Windows/Linux/macOS
- En az 4GB RAM
- NVIDIA GPU (opsiyonel, CPU'da da çalışır)

### 1. Repository'yi Clone Edin

```bash
git clone https://github.com/MetehanYasar11/ultralytics_mcp_server.git
cd ultralytics_mcp_server
```

### 2. Container'ları Başlatın

```bash
# Tüm servisleri başlat
docker-compose up -d

# Logları takip et
docker-compose logs -f
```

### 3. Web Arayüzlerine Erişim

Container başlatıldıktan sonra aşağıdaki arayüzlere erişebilirsiniz:

- **🌐 Streamlit UI**: http://localhost:8501 - Enhanced custom model interface
- **📊 TensorBoard**: http://localhost:6006 - Training visualization 
- **📓 Jupyter Lab**: http://localhost:8888 - Interactive development
- **🔧 MCP Health**: http://localhost:8092/health - Service status

### 4. Bağlantı Kontrolü

```bash
# Health check
curl http://localhost:8092/health

# SSE endpoint test
curl http://localhost:8092/sse
```

### 5. N8N'de Kullanım

N8N'de **MCP** node ekleyin ve aşağıdaki URL'yi kullanın:

```
http://localhost:8092/sse
```

## 📚 Kullanım Kılavuzu

### 7 MCP Tools ile Kapsamlı AI Workflow

#### 1. `execute_python`
**Açıklama**: Ultralytics container'da Python kodu çalıştırır  
**Parametreler**: `code` (string) - Çalıştırılacak Python kodu  

```python
from ultralytics import YOLO
model = YOLO('yolo11n.pt')
print("Model loaded successfully!")
```

#### 2. `yolo_detect`
**Açıklama**: Object detection yapar  
**Parametreler**: 
- `image_url` (string) - Görüntü URL'si
- `model` (string, default: "yolo11n.pt") - Model dosyası

```json
{
  "image_url": "https://ultralytics.com/images/bus.jpg",
  "model": "yolo11n.pt"
}
```

#### 3. `yolo_segment`
**Açıklama**: Instance segmentation yapar  
**Parametreler**: 
- `image_url` (string) - Görüntü URL'si
- `model` (string, default: "yolo11n-seg.pt") - Segmentation model

#### 4. `yolo_classify`
**Açıklama**: Image classification yapar  
**Parametreler**: 
- `image_url` (string) - Görüntü URL'si
- `model` (string, default: "yolo11n-cls.pt") - Classification model

#### 5. `yolo_pose`
**Açıklama**: Pose estimation yapar  
**Parametreler**: 
- `image_url` (string) - Görüntü URL'si
- `model` (string, default: "yolo11n-pose.pt") - Pose model

#### 6. `launch_streamlit_interface`
**Açıklama**: Enhanced Streamlit web arayüzünü başlatır  
**Parametreler**: Yok - Otomatik custom model scanning ile

#### 7. `get_system_info`
**Açıklama**: Container sistem bilgilerini getirir  
**Parametreler**: Yok - GPU, memory, disk bilgileri

## 🌐 Web Arayüzleri

### Streamlit Enhanced Interface
- **URL**: http://localhost:8501
- **Özellikler**: Custom model scanning, modern UI, file upload
- **Desteklenen Modeller**: YOLO11, custom trained models

### TensorBoard Visualization  
- **URL**: http://localhost:6006
- **Özellikler**: Training metrics, loss curves, model graphs
- **Otomatik Başlatma**: Container startup ile birlikte

### Jupyter Lab Development
- **URL**: http://localhost:8888
- **Özellikler**: Interactive Python environment, notebook support
- **Pre-installed**: Ultralytics, all dependencies

### N8N Workflow Örnekleri

#### Basit Object Detection

1. **Trigger Node**: Manual trigger veya webhook
2. **Set Node**: Image URL'sini ayarla
3. **MCP Node**: `yolo_detect` tool'unu çağır
4. **Code Node**: Sonuçları işle

```javascript
// N8N Code Node örneği
const result = $input.all()[0].json;
const detections = JSON.parse(result.content[0].text);

return detections.map(detection => ({
  class: detection.class_name,
  confidence: detection.confidence,
  bbox: detection.bbox
}));
```

## 🎯 Ultralytics Solutions Framework

Bu proje **15+ hazır AI uygulaması** ile birlikte gelir. Streamlit arayüzünden erişebileceğiniz Solutions:

### 🔍 Detection & Tracking
- **Object Counting**: Gerçek zamanlı nesne sayma
- **Object Tracking**: Multi-object tracking (ByteTrack, BoT-SORT)
- **Speed Estimation**: Nesne hız ölçümü
- **Distance Calculation**: Mesafe hesaplaması

### 🛡️ Security & Monitoring  
- **Security Alarm**: Güvenlik alarm sistemi
- **Queue Management**: Kuyruk yönetimi
- **Region Counting**: Bölgesel nesne sayma
- **Analytics Dashboard**: Analitik kontrol paneli

### 🏭 Industrial Applications
- **Workzone Monitoring**: İş alanı takibi
- **PPE Detection**: Kişisel koruyucu ekipman tespiti
- **Pose Estimation**: İnsan duruş analizi
- **Instance Segmentation**: Pixel-level segmentasyon

### 📱 Interactive Features
- **Live Inference**: Gerçek zamanlı çıkarım
- **Custom Model Upload**: Kendi modelinizi yükleyin
- **Batch Processing**: Toplu görüntü işleme
- **Export Results**: Sonuçları dışa aktarma

## 🔧 Konfigürasyon

### Otomatik Servis Başlatma
Container başlatıldığında otomatik olarak şu servisler çalışır:
- **TensorBoard**: Port 6006
- **Streamlit**: Port 8501  
- **Jupyter Lab**: Port 8888

### Docker Compose Ayarları

```yaml
# docker-compose.yml
version: '3.8'
services:
  ultralytics:
    command: bash -c "/usr/local/bin/startup.sh || sleep infinity"
    ports:
      - "8888:8888"  # Jupyter Lab
      - "6006:6006"  # TensorBoard  
      - "8501:8501"  # Streamlit
    # GPU desteği
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
  
  mcp-connector:
    build: .
    ports:
      - "8092:8092"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
```

### Environment Variables

```bash
# .env dosyası oluşturun
MCP_PORT=8092
ULTRALYTICS_CONTAINER=ultralytics-container
NODE_ENV=production
LOG_LEVEL=info
```

## 🐛 Sorun Giderme

### Yaygın Sorunlar

#### 1. "Container not found" Hatası

**Problem**: Ultralytics container bulunamıyor  
**Çözüm**: 
```bash
# Container'ların durumunu kontrol et
docker ps -a

# Eğer yoksa, yeniden başlat
docker-compose up -d ultralytics-container
```

#### 2. "Permission denied" - Docker Socket

**Problem**: Docker socket'e erişim yok  
**Çözüm**: 
```bash
# Linux/macOS
sudo chmod 666 /var/run/docker.sock

# Windows - Docker Desktop'ı admin olarak çalıştır
```

#### 3. "SSE Connection Failed"

**Problem**: N8N'den MCP server'a bağlanamıyor  
**Çözüm**: 
```bash
# Port kontrolü
netstat -an | grep 8092

# Container logları
docker logs mcp-connector-container

# Network connectivity
curl http://localhost:8092/health
```

#### 4. "Python execution failed: SyntaxError"

**Problem**: Python kodu çalıştırılamıyor  
**Çözüm**: 
- Base64 encoding kullanılıyor, özel karakterlere dikkat
- Tek satırda kod yazın veya `\n` ile satır ayırın
- String içinde tırnak işaretlerine dikkat

#### 5. "Model download failed"

**Problem**: YOLO model indirilemiyor  
**Çözüm**: 
```bash
# Container'a bağlan ve manuel indir
docker exec -it ultralytics-container bash
yolo predict model=yolo11n.pt source=https://ultralytics.com/images/bus.jpg
```

#### 6. Memory/Performance Issues

**Problem**: Yavaş çalışma veya memory hataları  
**Çözüm**: 
```yaml
# docker-compose.yml - resource limits
services:
  mcp-connector:
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
```

### Debug Modu

```bash
# Debug logları aktifleştir
docker-compose logs -f mcp-connector

# Detaylı container durumu
docker inspect mcp-connector-container

# Network debug
docker network ls
docker network inspect ultralytics_mcp_server_default
```

## 📊 Monitoring ve Logging

### Health Check Endpoints

```bash
# Genel durum
curl http://localhost:8092/health

# SSE connections
curl http://localhost:8092/sse

# Active sessions (internal)
docker logs mcp-connector-container | grep "SSE transport connected"
```

### Log Seviyeleri

- `ERROR`: Kritik hatalar
- `WARN`: Uyarılar
- `INFO`: Genel bilgiler
- `DEBUG`: Detaylı debug bilgileri

### Performans Metrikleri

```bash
# Container resource kullanımı
docker stats mcp-connector-container ultralytics-container

# Disk kullanımı
docker system df

# Network trafiği
docker exec mcp-connector-container netstat -i
```

## 🔒 Güvenlik

### Güvenlik Önlemleri

1. **Container İzolasyonu**: Her servis kendi container'ında
2. **Network Segmentation**: Internal Docker network
3. **No Root Access**: Non-root user ile çalışma
4. **Resource Limits**: Memory ve CPU limitleri
5. **Input Validation**: Python code validation

### Production Deployment

```yaml
# production docker-compose.yml
version: '3.8'
services:
  mcp-connector:
    restart: unless-stopped
    environment:
      - NODE_ENV=production
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8092/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## 🚀 Gelişmiş Kullanım

### Custom Model Training

```python
# Custom model training örneği
from ultralytics import YOLO

# Yeni model oluştur
model = YOLO('yolo11n.yaml')

# Eğit
model.train(data='custom_dataset.yaml', epochs=100)

# Kaydet
model.save('custom_model.pt')
```

### Batch Processing

```python
# Toplu işlem örneği
import glob
from ultralytics import YOLO

model = YOLO('yolo11n.pt')

# Klasördeki tüm resimleri işle
images = glob.glob('/path/to/images/*.jpg')
results = model.predict(images, save=True)

for r in results:
    print(f"Detected {len(r.boxes)} objects")
```

### Real-time Video Processing

```python
# Video stream processing
from ultralytics import YOLO
import cv2

model = YOLO('yolo11n.pt')

# Webcam'den stream
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model.predict(frame, stream=True)
    
    for r in results:
        # Sonuçları işle
        annotated = r.plot()
        cv2.imshow('YOLO', annotated)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 🚀 Yenilikler (v2.0)

### ✨ Son Güncellemeler
- **🔄 Otomatik Servis Başlatma**: Container boot'ta tüm servisler otomatik ayağa kalkar
- **🎯 Enhanced Streamlit**: Custom model scanning ve modern UI
- **📊 Native TensorBoard**: Ultralytics 8.3.171 ile built-in TensorBoard desteği  
- **🛠️ 7 MCP Tools**: Comprehensive AI workflow automation
- **🏭 Solutions Framework**: 15+ hazır AI uygulaması
- **🔧 System Info Tool**: Container sistem bilgileri ve monitoring

### 🎯 Performance Optimizations
- CUDA 12.4.1 ile GPU acceleration
- RTX 5070 Ti optimize edilmiş configuration
- Memory efficient processing
- Background service management

## 🤝 Katkıda Bulunma

1. Fork'layın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit'leyin (`git commit -m 'Add amazing feature'`)
4. Push'layın (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 🙏 Teşekkürler

- [Ultralytics](https://ultralytics.com) - YOLO implementasyonu
- [N8N](https://n8n.io) - Workflow automation  
- [Model Context Protocol](https://modelcontextprotocol.io) - MCP specification

## 📞 Destek

- **Issues**: [GitHub Issues](https://github.com/MetehanYasar11/ultralytics_mcp_server/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MetehanYasar11/ultralytics_mcp_server/discussions)
- **Documentation**: [Wiki](https://github.com/MetehanYasar11/ultralytics_mcp_server/wiki)

---

**⭐ Bu projeyi beğendiyseniz star vermeyi unutmayın!**
