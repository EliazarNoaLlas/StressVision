# 📊 Diagrama de Flujo - Stress Vision

## 🔄 Flujo Completo del Sistema

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FASE 1: CONFIGURACIÓN INICIAL                     │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │  Instalar Dependencias   │
                    │  pip install -r          │
                    │  requirements.txt        │
                    └──────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │  Crear Base de Datos     │
                    │  python init_database.py │
                    └──────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FASE 2: ENROLLMENT DE EMPLEADOS                   │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │  Ejecutar Enrollment     │
                    │  python enrollment.py    │
                    └──────────────────────────┘
                                   │
                    ┌──────────────┴─────────────┐
                    │                            │
                    ▼                            ▼
        ┌──────────────────┐        ┌──────────────────┐
        │   Individual     │        │   Batch (20)     │
        │   1 empleado     │        │   empleados      │
        └──────────────────┘        └──────────────────┘
                    │                            │
                    └──────────────┬─────────────┘
                                   ▼
                    ┌──────────────────────────┐
                    │  Por cada empleado:      │
                    │  1. Consentimiento       │
                    │  2. Captura 10 fotos     │
                    │  3. Generar embeddings   │
                    │  4. Calcular calidad     │
                    │  5. Guardar JSON + JPGs  │
                    └──────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │  Archivos generados:     │
                    │  • EMP###_embedding.json │
                    │  • EMP###_sample_N.jpg   │
                    └──────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FASE 3: CARGA A BASE DE DATOS                     │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │  Cargar Enrollments      │
                    │  python                  │
                    │  load_enrollments.py     │
                    └──────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │  Leer archivos JSON      │
                    │  desde enrollments/      │
                    └──────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │  Por cada archivo:       │
                    │  1. Parsear JSON         │
                    │  2. Validar embedding    │
                    │  3. INSERT/UPDATE en BD  │
                    │  4. Verificar calidad    │
                    └──────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │  Base de Datos           │
                    │  gloria_stress_system.db │
                    │  - 20 empleados activos  │
                    │  - Embeddings cargados   │
                    └──────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FASE 4: MONITOREO EN TIEMPO REAL                  │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │  Iniciar Streamlit       │
                    │  streamlit run main.py   │
                    └──────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │  Capturar frame          │
                    │  desde cámara            │
                    └──────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │  Detectar rostros        │
                    │  (MTCNN)                 │
                    └──────────────────────────┘
                                   │
                    ┌──────────────┴─────────────┐
                    │                            │
                    ▼                            ▼
        ┌──────────────────┐        ┌──────────────────┐
        │  Rostro          │        │  No hay rostro   │
        │  detectado       │        │  → Siguiente     │
        └──────────────────┘        │    frame         │
                    │                └──────────────────┘
                    ▼
        ┌──────────────────────────┐
        │  Generar embedding       │
        │  (FaceNet)               │
        └──────────────────────────┘
                    │
                    ▼
        ┌──────────────────────────┐
        │  Buscar en BD            │
        │  (similitud coseno)      │
        └──────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
┌──────────────┐      ┌──────────────┐
│ Empleado     │      │ Desconocido  │
│ identificado │      │ (invitado)   │
└──────────────┘      └──────────────┘
        │                       │
        └───────────┬───────────┘
                    ▼
        ┌──────────────────────────┐
        │  Detectar emoción        │
        │  (DeepFace)              │
        └──────────────────────────┘
                    │
                    ▼
        ┌──────────────────────────┐
        │  Guardar detección       │
        │  en detection_events     │
        └──────────────────────────┘
                    │
                    ▼
        ┌──────────────────────────┐
        │  Calcular índice         │
        │  de estrés               │
        └──────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
┌──────────────┐      ┌──────────────┐
│ Estrés ALTO  │      │ Estrés       │
│ → Generar    │      │ Normal       │
│   Alerta     │      └──────────────┘
└──────────────┘
        │
        ▼
┌──────────────────────────┐
│  Insertar en alerts      │
│  Enviar notificación     │
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│  Actualizar dashboard    │
│  Mostrar métricas        │
└──────────────────────────┘
```

---

## 🔄 Flujo Detallado de Enrollment

```
INICIO
  │
  ▼
┌─────────────────────┐
│ Iniciar sistema     │
│ enrollment.py       │
└─────────────────────┘
  │
  ▼
┌─────────────────────┐
│ Cargar modelos:     │
│ • MTCNN             │
│ • FaceNet           │
└─────────────────────┘
  │
  ▼
┌─────────────────────┐
│ Solicitar datos:    │
│ • Código            │
│ • Nombre            │
│ • Departamento      │
│ • Turno             │
└─────────────────────┘
  │
  ▼
┌─────────────────────┐
│ Abrir cámara        │
└─────────────────────┘
  │
  ▼
┌─────────────────────┐
│ Capturar frame      │
└─────────────────────┘
  │
  ▼
┌─────────────────────┐
│ Detectar rostro     │
│ con MTCNN           │
└─────────────────────┘
  │
  ├─ No detectado ─→ Volver a capturar
  │
  ▼ Detectado
┌─────────────────────┐
│ Esperar ESPACIO     │
└─────────────────────┘
  │
  ▼ ESPACIO presionado
┌─────────────────────┐
│ Extraer ROI facial  │
└─────────────────────┘
  │
  ▼
┌─────────────────────┐
│ Redimensionar       │
│ 160x160 pixels      │
└─────────────────────┘
  │
  ▼
┌─────────────────────┐
│ Normalizar imagen   │
│ (mean=127.5, std=128)│
└─────────────────────┘
  │
  ▼
┌─────────────────────┐
│ Inferencia FaceNet  │
│ → embedding (512-D) │
└─────────────────────┘
  │
  ▼
┌─────────────────────┐
│ Guardar embedding   │
│ y foto de muestra   │
└─────────────────────┘
  │
  ▼
┌─────────────────────┐
│ Contador++          │
│ (10 muestras?)      │
└─────────────────────┘
  │
  ├─ No (< 10) ─→ Volver a capturar
  │
  ▼ Sí (= 10)
┌─────────────────────┐
│ Calcular embedding  │
│ promedio (mean)     │
└─────────────────────┘
  │
  ▼
┌─────────────────────┐
│ Calcular desviación │
│ estándar (std)      │
└─────────────────────┘
  │
  ▼
┌─────────────────────┐
│ Calcular calidad    │
│ (similitud coseno)  │
└─────────────────────┘
  │
  ▼
┌─────────────────────┐
│ Generar thumbnail   │
│ en base64           │
└─────────────────────┘
  │
  ▼
┌─────────────────────┐
│ Crear JSON:         │
│ • employee_code     │
│ • employee_name     │
│ • mean_embedding    │
│ • std_embedding     │
│ • quality_score     │
│ • thumbnail_base64  │
│ • timestamp         │
└─────────────────────┘
  │
  ▼
┌─────────────────────┐
│ Guardar archivo     │
│ EMP###_embedding    │
│ .json               │
└─────────────────────┘
  │
  ▼
┌─────────────────────┐
│ Mostrar resumen     │
│ • Calidad           │
│ • Muestras          │
│ • Recomendación     │
└─────────────────────┘
  │
  ▼
FIN
```

---

## 🔄 Flujo de Reconocimiento Facial

```
INICIO (frame de cámara)
  │
  ▼
┌─────────────────────┐
│ Detectar rostros    │
│ con MTCNN           │
└─────────────────────┘
  │
  ├─ No hay rostros ─→ Retornar None
  │
  ▼ Hay rostros
┌─────────────────────┐
│ Extraer ROI facial  │
└─────────────────────┘
  │
  ▼
┌─────────────────────┐
│ Generar embedding   │
│ del rostro actual   │
└─────────────────────┘
  │
  ▼
┌─────────────────────┐
│ Cargar embeddings   │
│ desde BD (tabla     │
│ employees)          │
└─────────────────────┘
  │
  ▼
┌─────────────────────────────────────┐
│ Para cada empleado en BD:           │
│                                     │
│   1. Cargar embedding guardado      │
│   2. Calcular similitud coseno      │
│      con embedding actual           │
│   3. Si similitud > umbral (0.7):   │
│      → Match encontrado             │
└─────────────────────────────────────┘
  │
  ├─ No hay match ─→ Retornar "Desconocido"
  │
  ▼ Hay match
┌─────────────────────┐
│ Retornar:           │
│ • employee_id       │
│ • employee_name     │
│ • confidence        │
└─────────────────────┘
  │
  ▼
┌─────────────────────┐
│ Detectar emoción    │
│ con DeepFace        │
└─────────────────────┘
  │
  ▼
┌─────────────────────┐
│ Guardar en BD:      │
│ detection_events    │
│                     │
│ • employee_id       │
│ • emotion           │
│ • timestamp         │
│ • confidence        │
└─────────────────────┘
  │
  ▼
┌─────────────────────┐
│ Actualizar:         │
│ • last_seen         │
│ • Métricas          │
└─────────────────────┘
  │
  ▼
FIN
```

---

## 📊 Flujo de Datos

```
┌──────────────┐
│   Cámara     │
└──────┬───────┘
       │ Frame (BGR)
       ▼
┌──────────────┐
│    MTCNN     │
│  (Detector)  │
└──────┬───────┘
       │ Bounding Box + Confidence
       ▼
┌──────────────┐
│   FaceNet    │
│ (Embedder)   │
└──────┬───────┘
       │ Embedding (512-D)
       ▼
┌──────────────────────────────┐
│      Base de Datos           │
│  gloria_stress_system.db     │
│                              │
│  ┌──────────────────────┐   │
│  │  employees           │   │
│  │  • face_embedding    │   │
│  └──────────────────────┘   │
│           │                  │
│           ▼ (match)          │
│  ┌──────────────────────┐   │
│  │  detection_events    │   │
│  │  • employee_id       │   │
│  │  • timestamp         │   │
│  │  • emotion           │   │
│  └──────────────────────┘   │
│           │                  │
│           ▼ (aggregate)      │
│  ┌──────────────────────┐   │
│  │  employee_stress_    │   │
│  │  summary             │   │
│  └──────────────────────┘   │
│           │                  │
│           ▼ (alert?)         │
│  ┌──────────────────────┐   │
│  │  alerts              │   │
│  └──────────────────────┘   │
└──────────────────────────────┘
       │
       ▼
┌──────────────┐
│  Dashboard   │
│  (Streamlit) │
└──────────────┘
```

---

## 🎯 Estados del Sistema

### Estado 1: Sin Configurar
```
[START] → Instalar deps → Crear BD → [Estado 2]
```

### Estado 2: BD Creada, Sin Enrollments
```
[BD Lista] → Enrollment → [Estado 3]
```

### Estado 3: Enrollments Pendientes de Carga
```
[JSON Files] → Cargar a BD → [Estado 4]
```

### Estado 4: Sistema Listo
```
[Todo OK] → Streamlit → [Monitoreo Activo]
```

### Estado 5: Monitoreo Activo
```
[Cámara ON] → Detección → Reconocimiento → Análisis
              ↑______________|
```

---

## 📝 Decisiones de Diseño

### 1. ¿Por qué FaceNet (512-D)?
```
Alternativas evaluadas:
├─ FaceNet (512-D)     ✅ Elegido
│  • Precisión: Alta
│  • Velocidad: Media
│  • Tamaño: Moderado
│
├─ VGGFace (2622-D)    ❌
│  • Precisión: Alta
│  • Velocidad: Lenta
│  • Tamaño: Grande
│
└─ OpenFace (128-D)    ❌
   • Precisión: Media
   • Velocidad: Rápida
   • Tamaño: Pequeño
```

### 2. ¿Por qué SQLite vs PostgreSQL?
```
SQLite:                PostgreSQL:
✅ Sin servidor        ❌ Requiere servidor
✅ Archivo único       ❌ Configuración compleja
✅ Portable            ❌ Dependencias externas
✅ Ideal para piloto   ✅ Mejor para producción
```

### 3. ¿Por qué MTCNN vs Haar Cascade?
```
MTCNN:                 Haar Cascade:
✅ Más preciso         ❌ Menos preciso
✅ Mejor con ángulos   ❌ Solo frontal
⚠️ Más lento           ✅ Muy rápido
✅ Deep Learning       ❌ Heurístico
```

---

Gloria S.A. - Stress Vision v2.0




