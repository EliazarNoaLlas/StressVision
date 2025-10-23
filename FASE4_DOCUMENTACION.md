# 🤖 Fase 4: Entrenamiento y Optimización del Modelo

## 📊 Resumen Ejecutivo

Esta fase implementa el entrenamiento completo del modelo de detección de estrés, desde la preparación de datos hasta la optimización para Raspberry Pi 5.

---

## 🎯 Objetivos de la Fase

### KPIs Objetivo:
- ✅ **Accuracy ≥ 84%** en test set
- ✅ **Latencia ≤ 200ms** por frame en Raspberry Pi 5
- ✅ **Tamaño del modelo ≤ 10 MB** (TFLite quantized)
- ✅ **5 clases de emoción**: neutral, stress, sad, happy, fatigue

---

## 📁 Archivos Creados

### Módulos Python (5 archivos)

| Archivo | Líneas | Descripción |
|---------|--------|-------------|
| `data_preparation.py` | 500+ | Carga y procesamiento de datasets |
| `model_architecture.py` | 400+ | Arquitecturas MobileNetV3 y Custom Light |
| `model_trainer.py` | 450+ | Sistema de entrenamiento con callbacks |
| `convert_to_tflite.py` | 400+ | Conversión y quantization a TFLite |
| `train_model.py` | 350+ | Script principal de entrenamiento |
| `evaluate_model.py` | 250+ | Evaluación y análisis de resultados |

**Total: ~2,350 líneas de código**

---

## 🔄 Pipeline Completo

```
┌──────────────────────────────────────────────────────────────┐
│ 1. PREPARACIÓN DE DATOS (data_preparation.py)                │
├──────────────────────────────────────────────────────────────┤
│ • Cargar FER-2013 (28,709 imágenes)                          │
│ • Cargar dataset custom de Gloria                            │
│ • Mapeo de emociones → [neutral, stress, sad, happy, fatigue]│
│ • Data augmentation para balanceo                            │
│ • División: 70% train / 15% val / 15% test                   │
│ • Guardar en NPZ: data/processed/                            │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ 2. CONSTRUCCIÓN DEL MODELO (model_architecture.py)           │
├──────────────────────────────────────────────────────────────┤
│ Opción A: MobileNetV3-Small                                  │
│   • Transfer learning de ImageNet                            │
│   • Fine-tuning de últimas 30 capas                          │
│   • Custom head: Dense(256) → Dense(128) → Softmax(5)        │
│   • ~4-6 MB                                                   │
│                                                               │
│ Opción B: Custom Light                                       │
│   • Arquitectura custom con Separable Convolutions           │
│   • Optimizado para CPU                                      │
│   • ~1-2 MB                                                   │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ 3. ENTRENAMIENTO (model_trainer.py + train_model.py)         │
├──────────────────────────────────────────────────────────────┤
│ • Optimizer: Adam (lr=0.001)                                 │
│ • Loss: Categorical Crossentropy                             │
│ • Callbacks:                                                  │
│   - EarlyStopping (patience=10)                              │
│   - ModelCheckpoint (guardar mejor modelo)                   │
│   - ReduceLROnPlateau (factor=0.5)                           │
│   - TensorBoard logging                                      │
│   - CSV logger                                                │
│ • Data augmentation en training (opcional)                   │
│ • Métricas: accuracy, precision, recall, AUC                 │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ 4. EVALUACIÓN (evaluate_model.py)                            │
├──────────────────────────────────────────────────────────────┤
│ • Métricas globales en test set                              │
│ • Classification report por clase                            │
│ • Matriz de confusión                                        │
│ • Análisis de errores más comunes                            │
│ • Visualizaciones: confusion matrix, historial training      │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ 5. OPTIMIZACIÓN PARA RASPBERRY PI (convert_to_tflite.py)     │
├──────────────────────────────────────────────────────────────┤
│ • Conversión a TensorFlow Lite                               │
│ • Quantization INT8 (reducción 4x de tamaño)                 │
│ • Benchmark de latencia (100 runs)                           │
│ • Verificación de KPI ≤ 200ms                                │
│ • Output: model_int8.tflite (~1-3 MB)                        │
└──────────────────────────────────────────────────────────────┘
```

---

## 🚀 Guía de Uso

### Paso 1: Preparar Datos

```bash
python data_preparation.py
```

**Requisitos previos:**
- Descargar FER-2013 de [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
- (Opcional) Preparar dataset custom en `data/custom_dataset/`

**Estructura esperada:**
```
data/
├── fer2013/
│   └── fer2013.csv
└── custom_dataset/
    ├── neutral/
    ├── stress/
    ├── sad/
    ├── happy/
    └── fatigue/
```

**Output:**
```
data/processed/
├── train_data.npz
├── val_data.npz
├── test_data.npz
└── metadata.json
```

---

### Paso 2: Entrenar Modelo

```bash
python train_model.py
```

**Parámetros configurables:**
- Arquitectura: MobileNetV3 o Custom Light
- Épocas máximas: 50 (default)
- Batch size: 32 (default)
- Learning rate: 0.001 (default)
- Dropout rate: 0.3 (default)
- Data augmentation: Sí (default)

**Durante el entrenamiento:**
- Se guardan checkpoints cada época
- EarlyStopping detiene si no mejora en 10 épocas
- TensorBoard registra todas las métricas
- CSV log guarda historial

**Output:**
```
models/experiments/YYYYMMDD_HHMMSS/
├── final_model.keras
├── model_config.json
├── experiment_config.json
├── training_history.json
├── training_history.png
└── checkpoints/
    └── best_model.keras
```

---

### Paso 3: Evaluar Modelo

```bash
python evaluate_model.py
```

Selecciona el experimento a evaluar.

**Output:**
- Métricas globales
- Classification report
- Matriz de confusión (PNG)
- Análisis de errores
- JSON con resultados

---

### Paso 4: Convertir a TFLite

Opción A: Durante training (automático si acepta)

Opción B: Manual
```bash
python convert_to_tflite.py
```

**Tipos de conversión disponibles:**
1. Float32: Sin optimización (máxima precisión)
2. Float16: ~50% reducción
3. INT8 Quantized: ~75% reducción (RECOMENDADO)

**Benchmark de latencia:**
- 100 ejecuciones
- Calcula: mean, std, p50, p95, p99, FPS
- Verifica KPI ≤ 200ms

---

## 📊 Datasets Soportados

### 1. FER-2013 (Principal)

- **Origen:** Kaggle
- **Imágenes:** 28,709
- **Resolución:** 48x48 pixels (grayscale)
- **Clases originales:** 7 (angry, disgust, fear, happy, sad, surprise, neutral)
- **Mapeo a nuestras clases:**
  - angry, fear, disgust → **stress**
  - sad → **sad**
  - neutral, surprise → **neutral**
  - happy → **happy**

**Descarga:**
```bash
# Instalar Kaggle CLI
pip install kaggle

# Descargar FER-2013
kaggle datasets download -d msambare/fer2013
unzip fer2013.zip -d data/fer2013/
```

### 2. Dataset Custom de Gloria

Estructura:
```
data/custom_dataset/
├── neutral/
├── stress/
├── sad/
├── happy/
└── fatigue/
```

**Cómo crear:**
1. Capturar fotos de empleados en diferentes estados
2. Organizar en carpetas por emoción
3. Formato: JPG, PNG (160x160 recomendado)
4. Mínimo: 100 imágenes por clase

---

## 🏗️ Arquitecturas del Modelo

### Opción 1: MobileNetV3-Small (Recomendado)

**Ventajas:**
- Pre-entrenado en ImageNet (transfer learning)
- Alta precisión (esperada: 85-90%)
- Optimizado para móviles/edge devices
- Squeeze-and-Excitation para mejor feature extraction

**Desventajas:**
- ~4-6 MB (más pesado que Custom Light)
- Latencia ligeramente mayor

**Arquitectura:**
```python
Input (160x160x3)
  ↓
Preprocessing (normalización [-1, 1])
  ↓
MobileNetV3Small backbone (ImageNet)
  ↓ (fine-tuning últimas 30 capas)
GlobalAveragePooling2D
  ↓
Dropout(0.3)
  ↓
Dense(256, relu) + L2(0.001) + BatchNorm
  ↓
Dropout(0.2)
  ↓
Dense(128, relu) + L2(0.001) + BatchNorm
  ↓
Dense(5, softmax)
```

### Opción 2: Custom Light

**Ventajas:**
- Ultra-ligero (~1-2 MB)
- Latencia mínima
- Optimizado específicamente para CPU Raspberry Pi

**Desventajas:**
- Precisión ligeramente menor (esperada: 80-85%)
- Sin transfer learning

**Arquitectura:**
```python
Input (160x160x3)
  ↓
Rescaling (1/255)
  ↓
Conv2D(32, 3x3, stride=2) + BN + ReLU + MaxPool
  ↓
SeparableConv2D(64, 3x3) + BN + ReLU + MaxPool
  ↓
SeparableConv2D(128, 3x3) + BN + ReLU + MaxPool
  ↓
SeparableConv2D(256, 3x3) + BN + ReLU
  ↓
GlobalAveragePooling2D
  ↓
Dropout(0.3)
  ↓
Dense(128, relu)
  ↓
Dense(5, softmax)
```

---

## ⚙️ Hiperparámetros Recomendados

### Training

| Parámetro | Valor Default | Rango Recomendado |
|-----------|---------------|-------------------|
| Épocas | 50 | 30-100 |
| Batch Size | 32 | 16-64 |
| Learning Rate | 0.001 | 0.0001-0.01 |
| Dropout | 0.3 | 0.2-0.5 |
| L2 Regularization | 0.001 | 0.0001-0.01 |
| Optimizer | Adam | Adam, SGD+momentum |

### Data Augmentation

| Técnica | Valor |
|---------|-------|
| Rotation | ±10-15° |
| Width/Height Shift | ±10% |
| Horizontal Flip | Sí |
| Brightness | [0.9, 1.1] |
| Zoom | ±5% |

---

## 📈 Métricas de Evaluación

### Métricas Globales

- **Accuracy**: % de predicciones correctas
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: Media armónica de precision y recall
- **AUC**: Área bajo curva ROC

### Por Clase

Classification Report muestra precision, recall, f1-score para cada emoción:
- neutral
- stress
- sad
- happy
- fatigue

### Matriz de Confusión

Visualiza confusiones entre clases:
- Diagonal: Predicciones correctas
- Off-diagonal: Errores

---

## 🔧 Troubleshooting

### Problema 1: Accuracy muy baja (<70%)

**Causas posibles:**
- Dataset desbalanceado
- Modelo muy simple
- Hiperparámetros inadecuados
- Overfitting/Underfitting

**Soluciones:**
```bash
# 1. Aplicar data augmentation más agresivo
use_augmentation = True
target_per_class = 5000  # en data_preparation.py

# 2. Usar MobileNetV3 en lugar de Custom Light
backbone = 'mobilenetv3'

# 3. Ajustar learning rate
learning_rate = 0.0001  # Más bajo

# 4. Aumentar epochs
epochs = 100
```

### Problema 2: Overfitting (val_loss >> train_loss)

**Soluciones:**
```python
# 1. Aumentar dropout
dropout_rate = 0.5

# 2. Aumentar data augmentation
# 3. Aplicar L2 regularization más fuerte
# 4. Early stopping más agresivo
patience = 5
```

### Problema 3: Latencia > 200ms

**Soluciones:**
```bash
# 1. Usar Custom Light en lugar de MobileNetV3
backbone = 'custom_light'

# 2. Reducir input size
input_shape = (120, 120, 3)  # En lugar de 160x160

# 3. Usar INT8 quantization
python convert_to_tflite.py  # Opción 3

# 4. Considerar Coral USB TPU (~$60)
# Latencia: ~10-20ms
```

### Problema 4: Modelo muy grande

**Soluciones:**
```bash
# 1. Conversión a TFLite INT8
# Original: 10-20 MB → Quantized: 2-5 MB

# 2. Usar Custom Light
backbone = 'custom_light'  # ~1-2 MB

# 3. Reducir número de filtros
# Modificar arquitectura en model_architecture.py
```

---

## 🎯 Casos de Uso

### Caso 1: Máxima Precisión (Investigación)

```python
backbone = 'mobilenetv3'
epochs = 100
batch_size = 16
learning_rate = 0.0001
dropout_rate = 0.3
use_augmentation = True
target_per_class = 10000  # Dataset muy grande
```

### Caso 2: Balance Precisión-Velocidad (Producción)

```python
backbone = 'mobilenetv3'
epochs = 50
batch_size = 32
learning_rate = 0.001
dropout_rate = 0.3
use_augmentation = True
# + Conversión INT8
```

### Caso 3: Máxima Velocidad (Tiempo Real)

```python
backbone = 'custom_light'
input_shape = (120, 120, 3)
epochs = 50
batch_size = 64
learning_rate = 0.001
dropout_rate = 0.2
# + Conversión INT8
```

---

## 📊 Resultados Esperados

### MobileNetV3-Small

| Métrica | Esperado |
|---------|----------|
| Test Accuracy | 85-90% |
| Precision (avg) | 83-88% |
| Recall (avg) | 83-88% |
| Tamaño (Keras) | ~5-7 MB |
| Tamaño (TFLite INT8) | ~1.5-2.5 MB |
| Latencia (Raspberry Pi 5) | 150-200ms |

### Custom Light

| Métrica | Esperado |
|---------|----------|
| Test Accuracy | 80-85% |
| Precision (avg) | 78-83% |
| Recall (avg) | 78-83% |
| Tamaño (Keras) | ~2-3 MB |
| Tamaño (TFLite INT8) | ~0.7-1.0 MB |
| Latencia (Raspberry Pi 5) | 80-120ms |

---

## 💡 Mejores Prácticas

### 1. Preparación de Datos

✅ **DO:**
- Balancear clases con augmentation
- Usar validation set separado
- Normalizar inputs ([-1, 1] o [0, 1])
- Mezclar (shuffle) datasets combinados

❌ **DON'T:**
- Usar datos de test para entrenamiento
- Ignorar desbalance de clases
- Olvidar normalizar

### 2. Entrenamiento

✅ **DO:**
- Usar EarlyStopping para evitar overfitting
- Guardar checkpoints frecuentemente
- Monitorear val_loss y val_accuracy
- Usar TensorBoard para visualizar métricas

❌ **DON'T:**
- Entrenar sin validation set
- Ignorar overfitting (val_loss aumenta)
- Usar learning rate muy alto
- Olvidar data augmentation

### 3. Evaluación

✅ **DO:**
- Evaluar en test set nunca visto
- Analizar confusion matrix
- Revisar errores más comunes
- Calcular métricas por clase

❌ **DON'T:**
- Evaluar solo accuracy global
- Ignorar clases desbalanceadas
- Olvidar analizar errores

### 4. Optimización

✅ **DO:**
- Usar INT8 quantization para producción
- Hacer benchmark de latencia
- Verificar precisión post-quantization
- Considerar Coral TPU si latencia crítica

❌ **DON'T:**
- Desplegar modelo sin optimizar
- Ignorar latencia en dispositivo real
- Sacrificar mucha precisión por velocidad

---

## 📚 Referencias

### Papers

1. **MobileNetV3**: "Searching for MobileNetV3" (Howard et al., 2019)
2. **FER-2013**: "Challenges in Representation Learning" (Goodfellow et al., 2013)
3. **Transfer Learning**: "A Survey on Transfer Learning" (Pan & Yang, 2010)
4. **Quantization**: "Quantization and Training of Neural Networks" (Jacob et al., 2018)

### Datasets

- FER-2013: https://www.kaggle.com/datasets/msambare/fer2013
- AffectNet: http://mohammadmahoor.com/affectnet/
- RAF-DB: http://www.whdeng.cn/raf/model1.html

### Tools

- TensorFlow: https://www.tensorflow.org/
- TFLite: https://www.tensorflow.org/lite
- TensorBoard: https://www.tensorflow.org/tensorboard

---

## ✅ Checklist de Implementación

### Preparación
- [ ] Descargar FER-2013
- [ ] (Opcional) Crear dataset custom
- [ ] Ejecutar `data_preparation.py`
- [ ] Verificar `data/processed/` creado

### Entrenamiento
- [ ] Seleccionar arquitectura (MobileNetV3 o Custom Light)
- [ ] Configurar hiperparámetros
- [ ] Ejecutar `train_model.py`
- [ ] Monitorear TensorBoard
- [ ] Verificar convergencia

### Evaluación
- [ ] Ejecutar `evaluate_model.py`
- [ ] Revisar classification report
- [ ] Analizar confusion matrix
- [ ] Identificar errores comunes
- [ ] Verificar KPI accuracy ≥ 84%

### Optimización
- [ ] Convertir a TFLite INT8
- [ ] Ejecutar benchmark de latencia
- [ ] Verificar KPI latencia ≤ 200ms
- [ ] Probar en Raspberry Pi 5
- [ ] Ajustar si no cumple KPIs

### Despliegue
- [ ] Copiar `.tflite` a Raspberry Pi
- [ ] Integrar en sistema de monitoreo
- [ ] Pruebas end-to-end
- [ ] Monitoreo en producción

---

**Gloria S.A. - Stress Vision**  
**Fase 4: Entrenamiento y Optimización - Completada** ✅


