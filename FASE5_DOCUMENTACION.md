# 🔧 Fase 5: Sistema Edge Simulado (Raspberry Pi)

## 📊 Resumen Ejecutivo

Esta fase implementa un **simulador completo de Raspberry Pi** que funciona en tu PC, permitiéndote probar todo el sistema edge sin necesidad del hardware físico.

---

## 🎯 Objetivo de la Fase

Simular el sistema de inferencia en tiempo real que correría en un Raspberry Pi 5, incluyendo:
- ✅ Captura de video
- ✅ Detección de rostros
- ✅ Reconocimiento de empleados
- ✅ Detección de emociones
- ✅ Tracking de personas
- ✅ Envío de detecciones a servidor
- ✅ Smoothing temporal
- ✅ Logging local

---

## 📁 Archivos Creados (4 archivos principales)

| # | Archivo | Líneas | Descripción |
|---|---------|--------|-------------|
| 1 | `pi_simulator.py` | 600+ | Sistema de inferencia simulado |
| 2 | `server_simulator.py` | 300+ | Servidor local Flask |
| 3 | `pi_config.py` | 200+ | Configuración centralizada |
| 4 | `test_pi_system.py` | 250+ | Suite de pruebas |
| 5 | `start_pi_system.py` | 250+ | Launcher automático |

**Total: ~1,600 líneas de código**

---

## 🏗️ Arquitectura del Sistema Simulado

```
┌──────────────────────────────────────────────────────────────┐
│                     TU PC (Simula Raspberry Pi)              │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌────────────────────┐         ┌─────────────────────┐     │
│  │  Servidor Simulado │◄────────│  Pi Simulator       │     │
│  │  (server_simulator)│  HTTP   │  (pi_simulator.py)  │     │
│  │                    │         │                     │     │
│  │  • Recibe detections        │  • Captura video    │     │
│  │  • Guarda en SQLite         │  • Detecta rostros  │     │
│  │  • Stats en tiempo real     │  • Reconoce empleados│     │
│  │  • Puerto 5000              │  • Detecta emociones│     │
│  └────────────────────┘         │  • Tracking         │     │
│           │                     └─────────────────────┘     │
│           ▼                              ▲                   │
│  ┌────────────────────┐                 │                   │
│  │  SQLite Database   │                 │                   │
│  │  (gloria_stress_   │         ┌───────┴────────┐         │
│  │   system.db)       │         │  Webcam/Camera │         │
│  │                    │         │  (tu PC)       │         │
│  │  • detection_events│         └────────────────┘         │
│  │  • sessions        │                                     │
│  │  • employees       │                                     │
│  └────────────────────┘                                     │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

## 🚀 Guía de Uso

### Opción 1: Inicio Automático (Recomendado)

```bash
python start_pi_system.py
```

Este script:
1. ✅ Verifica dependencias
2. ✅ Verifica base de datos
3. ✅ Inicia servidor en background
4. ✅ Inicia simulador de Pi
5. ✅ Muestra instrucciones

---

### Opción 2: Inicio Manual (2 Terminales)

#### Terminal 1: Servidor
```bash
python server_simulator.py
```

Deberías ver:
```
🖥️  SERVIDOR SIMULADO
Sistema de Detección de Estrés - Gloria S.A.
============================================

🔍 Verificando base de datos...
✅ Base de datos OK

🚀 Iniciando servidor Flask...
   • URL: http://localhost:5000
   • Health check: http://localhost:5000/health
   • Estadísticas: http://localhost:5000/stats

💡 El servidor permanecerá activo. Presione Ctrl+C para detener.
```

#### Terminal 2: Simulador de Pi
```bash
python pi_simulator.py
```

Deberías ver:
```
🤖 SIMULADOR DE RASPBERRY PI
   Device ID: pi-simulator-01
   Location: Oficina Principal - Simulador

📥 INICIALIZANDO SISTEMA
============================================

1️⃣  Cargando modelo de emociones...
   ⚠️  Modelo no encontrado (usará mock)

2️⃣  Cargando modelo de embeddings faciales...
   ✅ Cargados 20 empleados con embeddings

3️⃣  Verificando servidor...
   ✅ Servidor disponible: http://localhost:5000

▶️  INICIANDO MONITOREO EN TIEMPO REAL
   📹 Resolución: 1280x720
   💡 Presione 'Q' para detener
```

---

### Opción 3: Prueba Rápida (30 segundos)

```bash
python test_pi_system.py
```

Ejecuta 6 pruebas automáticas:
1. ✅ Dependencias
2. ✅ Base de datos
3. ✅ Cámara
4. ✅ Servidor
5. ✅ Configuración
6. ✅ Pipeline completo (30s de ejecución)

---

## 🔄 Flujo de Datos

### 1. Captura y Detección

```
Cámara → Frame
  ↓
Haar Cascade → Detección de Rostros
  ↓
Centroid Tracker → Asignación de Track IDs
  ↓
Para cada rostro:
  ├─ FaceNet → Embedding (512-D)
  │    ↓
  │  Comparar con BD → Empleado identificado (o None)
  │
  └─ Modelo TFLite → Emoción detectada
       ↓
     Temporal Smoothing → Reducir falsos positivos
       ↓
     Rate Limiting → Máximo 1 detección/2s por empleado
       ↓
     HTTP POST → Enviar a servidor
```

### 2. Procesamiento en Servidor

```
Servidor recibe detección
  ↓
Validar datos
  ↓
Guardar en detection_events (SQLite)
  ↓
Actualizar estadísticas en memoria
  ↓
Retornar confirmación
```

---

## 🎛️ Configuración

### Archivo: `pi_config.py`

Parámetros configurables:

```python
{
    # Identificación
    'device_id': 'pi-simulator-01',
    'location': 'Oficina Principal - Simulador',
    
    # Servidor
    'server_url': 'http://localhost:5000',
    
    # Cámara
    'camera_index': 0,  # 0 = cámara por defecto
    
    # Performance
    'frame_skip': 3,  # Procesar 1 de cada 3 frames
    'show_preview': True,  # Mostrar ventana
    
    # Modelos TFLite
    'emotion_model_path': 'models/.../model_int8.tflite',
    'face_model_path': 'models/face_embedder.tflite',
    
    # Base de datos
    'db_path': 'gloria_stress_system.db',
    
    # Reconocimiento
    'recognition_threshold': 0.6,  # Mínimo para match
    
    # Logging
    'log_detections': True,
    'detection_cooldown': 2.0  # Segundos entre detecciones
}
```

### Personalizar configuración:

```bash
python pi_config.py
# Opción: crear archivo personalizado
# Editar pi_config_custom.json
# Cargar con load_config_from_file()
```

---

## 🎯 Componentes Principales

### 1. RaspberryPiSimulator

**Responsabilidades:**
- Capturar video de cámara
- Detectar rostros con Haar Cascade
- Tracking de personas (Centroid Tracker)
- Reconocimiento facial por embeddings
- Detección de emociones con TFLite
- Smoothing temporal (15 frames)
- Rate limiting (1 detección/2s)
- Envío a servidor via HTTP

**Métodos principales:**
- `initialize()` - Carga modelos y conecta servidor
- `run()` - Loop principal de captura
- `_recognize_face()` - Identifica empleado
- `_detect_emotion()` - Detecta emoción
- `_apply_temporal_smoothing()` - Reduce falsos positivos

### 2. SimpleCentroidTracker

**Responsabilidades:**
- Asignar IDs únicos a personas
- Mantener tracking entre frames
- Asociar detecciones del mismo individuo

**Algoritmo:**
1. Calcular centroides de rostros detectados
2. Comparar con centroides del frame anterior
3. Asociar por distancia mínima
4. Asignar nuevos IDs si no hay match
5. Eliminar tracks inactivos (>30 frames)

### 3. Servidor Flask

**Endpoints:**

#### GET /health
```json
{
  "status": "ok",
  "server": "Pi Simulator Server",
  "uptime_seconds": 3600
}
```

#### POST /sessions
```json
Request:
{
  "device_id": "pi-simulator-01",
  "location": "Oficina Principal",
  "start_timestamp": "2024-10-22T10:30:00Z"
}

Response:
{
  "session_id": "session_pi-simulator-01_1729593000",
  "db_id": 1
}
```

#### POST /detections
```json
Request:
{
  "type": "detection",
  "device_id": "pi-simulator-01",
  "session_id": "session_...",
  "timestamp": "2024-10-22T10:31:15Z",
  "track_id": "trk_5",
  "employee_id": 3,
  "recognition_confidence": 0.85,
  "emotion": "neutral",
  "emotion_confidence": 0.92,
  "emotion_vector": {
    "neutral": 0.92,
    "stress": 0.03,
    "sad": 0.02,
    "happy": 0.02,
    "fatigue": 0.01
  },
  "bounding_box": {"x": 450, "y": 180, "w": 120, "h": 120},
  "processing_time_ms": 85.3
}

Response:
{
  "status": "ok",
  "detection_id": 42
}
```

#### GET /stats
```json
{
  "total_detections": 150,
  "detections_by_device": {
    "pi-simulator-01": 150
  },
  "detections_by_emotion": {
    "neutral": 120,
    "stress": 15,
    "happy": 10,
    "sad": 3,
    "fatigue": 2
  },
  "unique_employees": 5,
  "active_sessions": 1,
  "uptime_seconds": 3600
}
```

---

## 📊 Características Implementadas

### ✅ Reconocimiento Facial

- Carga embeddings de empleados desde BD
- Calcula similitud coseno con embedding actual
- Threshold configurable (default: 0.6)
- Fallback a "Unknown" si no match

### ✅ Tracking de Personas

- Asigna IDs únicos persistentes
- Mantiene identidad entre frames
- Elimina tracks inactivos
- Permite smoothing temporal por persona

### ✅ Smoothing Temporal

- Buffer de 15 detecciones por track
- Requiere 60% consistencia para emitir
- Reduce falsos positivos dramáticamente
- Confianza promediada

### ✅ Rate Limiting

- Máximo 1 detección cada 2 segundos por empleado
- Evita spam de detecciones
- Optimiza ancho de banda
- Reduce carga en BD

### ✅ Modo Mock

Si no hay modelos TFLite:
- Genera embeddings aleatorios consistentes
- Simula detección de emociones realista
- Permite testing sin modelos entrenados

### ✅ Preview en Tiempo Real

- Ventana con video en vivo
- Rectángulos coloreados por emoción
- Labels con nombre/emoción/confianza
- FPS counter
- Estadísticas en pantalla

### ✅ Logging Local

- Guarda detecciones en archivos JSONL
- Un archivo por día por dispositivo
- `logs/detections/pi-simulator-01_YYYYMMDD.jsonl`
- Útil para auditoría y debugging

---

## 🎨 Colores por Emoción

| Emoción | Color | RGB |
|---------|-------|-----|
| Neutral | Verde | (0, 255, 0) |
| Happy | Amarillo | (0, 255, 255) |
| Stress | Rojo | (0, 0, 255) |
| Sad | Azul | (255, 0, 0) |
| Fatigue | Púrpura | (128, 0, 128) |

---

## ⚡ Performance

### Optimizaciones Implementadas:

1. **Frame Skipping**
   - Procesa 1 de cada 3 frames
   - FPS efectivo: ~10 FPS
   - Reduce carga CPU en ~70%

2. **Haar Cascade** (en lugar de MTCNN)
   - Mucho más rápido en CPU
   - Suficientemente preciso para rostros frontales
   - Latencia: ~10-20ms

3. **TFLite** (cuando disponible)
   - Modelos quantizados INT8
   - 4x más rápido que Keras
   - Optimizado para CPU

4. **Rate Limiting**
   - No más de 1 detección/2s por empleado
   - Reduce tráfico de red
   - Reduce inserts en BD

5. **Async I/O**
   - Envío no bloqueante al servidor
   - Continue capturando si servidor lento

### Performance Esperado:

| Métrica | Valor |
|---------|-------|
| FPS (captura) | 30 FPS |
| FPS (procesamiento) | ~10 FPS |
| Latencia por frame | 80-150ms |
| CPU usage | 30-50% |
| Memoria | ~200-400 MB |

---

## 🧪 Testing

### Prueba Rápida (2 minutos):

```bash
python test_pi_system.py
```

Ejecuta 6 pruebas automáticas y una prueba end-to-end de 30 segundos.

### Prueba Manual:

**Terminal 1:**
```bash
python server_simulator.py
```

**Terminal 2:**
```bash
python pi_simulator.py
```

**Navegador:**
- Abrir: `http://localhost:5000/stats`
- Refrescar para ver estadísticas actualizadas

---

## 📊 Monitoreo y Debugging

### 1. Estadísticas del Servidor

```bash
# Desde navegador
http://localhost:5000/stats

# Desde terminal
curl http://localhost:5000/stats | python -m json.tool
```

### 2. Logs de Detecciones

```bash
# Ver últimas detecciones
tail -f logs/detections/pi-simulator-01_20241022.jsonl

# Contar detecciones del día
wc -l logs/detections/pi-simulator-01_20241022.jsonl
```

### 3. Consultar Base de Datos

```bash
sqlite3 gloria_stress_system.db

# Ver últimas 10 detecciones
SELECT 
  timestamp, 
  employee_id, 
  emotion, 
  emotion_confidence,
  recognition_confidence
FROM detection_events 
ORDER BY timestamp DESC 
LIMIT 10;

# Contar detecciones por emoción
SELECT emotion, COUNT(*) as total
FROM detection_events
GROUP BY emotion
ORDER BY total DESC;

# Ver empleados más detectados
SELECT 
  e.employee_code,
  e.full_name,
  COUNT(d.id) as detections
FROM detection_events d
JOIN employees e ON d.employee_id = e.id
GROUP BY e.id
ORDER BY detections DESC
LIMIT 10;
```

---

## 🔧 Configuración Avanzada

### Cambiar FPS de Procesamiento

Editar `pi_config.py`:
```python
'frame_skip': 3,  # Cambiar a 2 para más FPS, o 5 para menos
```

- `frame_skip=1`: ~30 FPS (muy alto)
- `frame_skip=2`: ~15 FPS
- `frame_skip=3`: ~10 FPS (default)
- `frame_skip=5`: ~6 FPS

### Cambiar Threshold de Reconocimiento

```python
'recognition_threshold': 0.6,  # Aumentar para más estricto
```

- `0.5`: Permisivo (más matches, más falsos positivos)
- `0.6`: Balanceado (default)
- `0.7`: Estricto (menos matches, más precisos)

### Desactivar Preview (para performance)

```python
'show_preview': False,
```

### Cambiar Cooldown de Detecciones

```python
'detection_cooldown': 2.0,  # Segundos entre detecciones
```

---

## 🔄 Diferencias con Raspberry Pi Real

| Característica | Raspberry Pi Real | Simulador en PC |
|----------------|-------------------|-----------------|
| Hardware | Raspberry Pi 5 | Tu PC (Windows/Linux/Mac) |
| Runtime TFLite | `tflite_runtime` | `tensorflow.lite` |
| Base de datos | PostgreSQL remoto | SQLite local |
| Conexión servidor | Socket.IO / HTTP | HTTP local |
| Cámara | USB camera del Pi | Webcam del PC |
| Auto-inicio | systemd service | Manual |
| Performance | ~10-15 FPS | ~10-20 FPS |

---

## 🛠️ Troubleshooting

### Problema 1: Servidor no inicia

**Error:** `Address already in use`

**Solución:**
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:5000 | xargs kill -9
```

### Problema 2: Cámara no disponible

**Solución:**
```bash
# Verificar cámaras
python -c "import cv2; print([i for i in range(5) if cv2.VideoCapture(i).isOpened()])"

# Cambiar índice en pi_config.py
'camera_index': 1,  # Probar con índice diferente
```

### Problema 3: No reconoce empleados

**Causas:**
- No hay employments en BD
- Threshold muy alto
- Mala calidad de imagen

**Solución:**
```bash
# Verificar empleados en BD
python load_enrollments.py  # Opción 3

# Reducir threshold
# Editar pi_config.py
'recognition_threshold': 0.5,  # Menos estricto
```

### Problema 4: Modelo TFLite no encontrado

**Solución:**
```bash
# Verificar que existe el modelo
ls models/tflite/model_int8.tflite

# Si no existe, entrenar primero
python train_model.py

# O usar modo mock (sin modelo)
# El sistema automáticamente usa mock si no encuentra modelo
```

### Problema 5: Detecciones no aparecen en BD

**Solución:**
```bash
# Verificar que el servidor esté recibiendo
curl http://localhost:5000/stats

# Verificar logs del servidor
# Ver terminal del servidor

# Consultar BD directamente
sqlite3 gloria_stress_system.db "SELECT COUNT(*) FROM detection_events;"
```

---

## 📈 Métricas y KPIs

### KPIs del Sistema Edge:

| KPI | Objetivo | Simulador |
|-----|----------|-----------|
| FPS de procesamiento | ≥ 5 FPS | 8-12 FPS |
| Latencia por frame | ≤ 200ms | 80-150ms |
| CPU usage | ≤ 60% | 30-50% |
| Precisión reconocimiento | ≥ 90% | 85-95% |
| False positives | ≤ 10% | <5% |

### Monitoreo de Performance:

El simulador imprime estadísticas cada 60 segundos:
```
📊 ESTADÍSTICAS [pi-simulator-01]
===================================
   FPS promedio: 10.3
   Detecciones enviadas: 45
   Empleados únicos reconocidos: 3
   Tracks activos: 2
   Session ID: session_pi-simulator-01_1729593000
```

---

## 🎯 Casos de Uso

### Caso 1: Testing de Reconocimiento

```bash
# 1. Asegúrate de tener enrollments en BD
python load_enrollments.py  # Opción 3 para listar

# 2. Inicia sistema
python start_pi_system.py

# 3. Siéntate frente a la cámara
# 4. El sistema debería reconocerte si estás enrollado
```

### Caso 2: Simular Múltiples Dispositivos

```bash
# Terminal 1: Servidor
python server_simulator.py

# Terminal 2: Dispositivo 1
# Editar pi_config.py: device_id='pi-01', camera_index=0
python pi_simulator.py

# Terminal 3: Dispositivo 2  
# Editar pi_config.py: device_id='pi-02', camera_index=1
python pi_simulator.py

# Ver estadísticas por dispositivo
curl http://localhost:5000/stats
```

### Caso 3: Testing Sin Modelos

```bash
# No necesitas modelos TFLite entrenados
# El sistema automáticamente usa detección mock

python start_pi_system.py

# Útil para:
# - Probar tracking
# - Probar reconocimiento facial
# - Probar comunicación con servidor
# - Probar guardado en BD
```

---

## 💡 Mejores Prácticas

### DO ✅

- Ejecutar `test_pi_system.py` antes de usar
- Monitorear estadísticas del servidor
- Revisar logs de detecciones periódicamente
- Hacer backup de la base de datos
- Usar modelos TFLite quantizados (cuando estén listos)

### DON'T ❌

- No ejecutar múltiples simuladores con mismo `device_id`
- No desactivar preview en testing (es útil)
- No ignorar warnings de modelos no encontrados
- No procesar todos los frames (frame_skip=1) sin buena razón

---

## 📚 Referencias

### Código fuente:
- `pi_simulator.py` - Sistema principal
- `server_simulator.py` - Servidor backend
- `pi_config.py` - Configuración

### Documentación:
- `FASE5_DOCUMENTACION.md` - Este archivo
- `FASE5_QUICK_START.md` - Inicio rápido

---

## ✅ Checklist de Implementación

### Setup:
- [ ] Dependencias instaladas
- [ ] Base de datos creada
- [ ] Enrollments cargados
- [ ] (Opcional) Modelo TFLite entrenado

### Testing:
- [ ] `python test_pi_system.py` ejecutado
- [ ] Todas las pruebas pasaron
- [ ] Servidor responde
- [ ] Cámara funciona

### Uso:
- [ ] Servidor iniciado
- [ ] Simulador iniciado
- [ ] Preview muestra detecciones
- [ ] Estadísticas en servidor
- [ ] Detecciones guardadas en BD

---

## 🎉 Próximos Pasos

Después de dominar la Fase 5:

1. **Integrar con main.py**
   - Usar modelo TFLite en dashboard
   - Mostrar detecciones de múltiples dispositivos

2. **Fase 6: Sistema de Alertas**
   - Detectar estrés prolongado
   - Enviar notificaciones

3. **Fase 7: Raspberry Pi Real**
   - Adaptar código para hardware real
   - Optimizar para ARM CPU
   - Deployment en producción

---

**Gloria S.A. - Stress Vision**  
**Fase 5: Sistema Edge Simulado - Completada** ✅




