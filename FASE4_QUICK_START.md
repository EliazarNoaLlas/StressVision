# ⚡ Quick Start - Fase 4: Entrenamiento del Modelo

## 🚀 Inicio Rápido (5 Pasos)

### Paso 1: Descargar FER-2013 (5 min)

```bash
# Instalar Kaggle CLI
pip install kaggle

# Configurar credenciales de Kaggle
# 1. Ir a kaggle.com → Account → Create New API Token
# 2. Copiar kaggle.json a ~/.kaggle/

# Descargar dataset
kaggle datasets download -d msambare/fer2013
unzip fer2013.zip -d data/fer2013/
```

---

### Paso 2: Preparar Datos (10-15 min)

```bash
python data_preparation.py
```

**Responder:**
- ¿Usar FER-2013? → `s`
- Path a fer2013.csv → `Enter` (usa default)
- ¿Usar dataset custom? → `n` (por ahora)
- ¿Aplicar data augmentation? → `s`
- Número objetivo por clase → `5000`
- Directorio de salida → `Enter`

**Output:** `data/processed/` con train, val, test data

---

### Paso 3: Entrenar Modelo (2-4 horas)

```bash
python train_model.py
```

**Configuración recomendada para empezar:**
- Arquitectura → `1` (MobileNetV3)
- Épocas → `50`
- Batch size → `32`
- Learning rate → `0.001`
- Dropout → `0.3`
- Data augmentation → `s`
- ¿Convertir a TFLite? → `s`

**Monitorear en tiempo real:**
```bash
# En otra terminal
tensorboard --logdir logs/
# Abrir http://localhost:6006
```

---

### Paso 4: Evaluar Resultados (2 min)

```bash
python evaluate_model.py
```

- Seleccionar experimento (el más reciente)
- Revisar métricas y confusion matrix

**Verificar:**
- ✅ Test accuracy ≥ 84%
- ✅ Latencia ≤ 200ms (si convirtió a TFLite)

---

### Paso 5: Desplegar en Raspberry Pi

```bash
# 1. Copiar modelo a Raspberry Pi
scp models/experiments/[EXPERIMENT]/tflite/model_int8.tflite pi@raspberry:~/

# 2. En Raspberry Pi, probar latencia
python3 benchmark_tflite.py model_int8.tflite
```

---

## 📊 Resultados Esperados

### Con MobileNetV3-Small + FER-2013:

| Métrica | Esperado | Mínimo |
|---------|----------|--------|
| Test Accuracy | 85-90% | 84% |
| Precision | 83-88% | 80% |
| Recall | 83-88% | 80% |
| Modelo (.keras) | 5-7 MB | - |
| TFLite (INT8) | 1.5-2.5 MB | <10 MB |
| Latencia (RPi 5) | 150-200ms | ≤200ms |

---

## 🔧 Si algo sale mal...

### Accuracy < 84%

```bash
# Opción 1: Más datos
python data_preparation.py
# Aumentar target_per_class a 8000-10000

# Opción 2: Más epochs
python train_model.py
# Aumentar epochs a 100

# Opción 3: Fine-tuning más agresivo
# Editar model_architecture.py línea 113
for layer in base_model.layers[:-50]:  # Entrenar más capas
```

### Latencia > 200ms

```bash
# Opción 1: Usar Custom Light
python train_model.py
# Seleccionar arquitectura 2

# Opción 2: Reducir input size
# Editar train_model.py línea 205
input_shape=(120, 120, 3)  # Menos que 160x160

# Opción 3: Coral USB TPU
# Comprar: ~$60
# Latencia esperada: 10-20ms
```

### Out of Memory

```bash
# Reducir batch size
python train_model.py
# batch_size = 16  # O incluso 8
```

---

## 💡 Tips Pro

### 1. Usar GPU para entrenar más rápido

```bash
# Verificar GPU disponible
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Si tienes GPU CUDA, el training será 5-10x más rápido
# 2-4 horas → 15-30 minutos
```

### 2. Guardar múltiples experimentos

```python
# Los experimentos se guardan automáticamente con timestamp
# Puedes entrenar múltiples configuraciones y comparar:

# Experimento 1: MobileNetV3 + 50 epochs
python train_model.py

# Experimento 2: Custom Light + 100 epochs  
python train_model.py

# Comparar:
python evaluate_model.py
```

### 3. Transfer learning incremental

```python
# 1. Entrenar solo el head (primero)
for layer in base_model.layers:
    layer.trainable = False

# 2. Fine-tune capas superiores (después)
for layer in base_model.layers[-30:]:
    layer.trainable = True
```

---

## 📁 Estructura Final

Después de completar todo:

```
StressVision/
├── data/
│   ├── fer2013/
│   │   └── fer2013.csv
│   └── processed/
│       ├── train_data.npz
│       ├── val_data.npz
│       ├── test_data.npz
│       └── metadata.json
│
├── models/
│   └── experiments/
│       └── YYYYMMDD_HHMMSS/
│           ├── final_model.keras
│           ├── experiment_config.json
│           ├── training_history.png
│           └── tflite/
│               ├── model_int8.tflite
│               └── latency_stats.json
│
├── logs/
│   └── YYYYMMDD_HHMMSS/
│       └── (TensorBoard logs)
│
├── data_preparation.py
├── model_architecture.py
├── model_trainer.py
├── convert_to_tflite.py
├── train_model.py
└── evaluate_model.py
```

---

## ⏱️ Timeline Estimado

| Actividad | Tiempo |
|-----------|--------|
| Descargar FER-2013 | 5 min |
| Preparar datos | 10-15 min |
| Entrenar modelo | 2-4 horas |
| Evaluar | 2 min |
| Convertir TFLite | 5-10 min |
| **TOTAL** | **~3-5 horas** |

*Con GPU: ~1-2 horas total*

---

## 🎯 Próximos Pasos

Después de completar Fase 4:

1. ✅ **Fase 5: Integración en Sistema de Monitoreo**
   - Integrar modelo TFLite en `main.py`
   - Reconocimiento en tiempo real
   - Asociar detecciones con empleados

2. ✅ **Fase 6: Dashboard Avanzado**
   - Visualizar estrés por empleado
   - Reportes automáticos
   - Alertas personalizadas

3. ✅ **Fase 7: Despliegue en Raspberry Pi**
   - Configurar Raspberry Pi 5
   - Instalar dependencias
   - Pruebas en producción

---

**¡Listo para entrenar tu modelo de detección de estrés! 🚀**

Gloria S.A. - Stress Vision


