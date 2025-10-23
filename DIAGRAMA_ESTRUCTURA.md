# Diagrama de Estructura del Proyecto StressVision

## 📊 Estructura Visual Completa

```
StressVision/
│
├── 📄 README.md                         # Documentación principal del proyecto
├── 📄 LICENSE                           # Licencia del proyecto
├── 📄 .gitignore                        # Archivos a ignorar en git
├── 📄 .env.example                      # Ejemplo de variables de entorno
├── 📄 requirements.txt                  # Dependencias Python globales
├── 📄 docker-compose.yml                # Orquestación de contenedores
│
├── 📁 docs/                             # 📚 DOCUMENTACIÓN
│   ├── 📄 README.md
│   ├── 📄 PROYECTO_COMPLETO_FINAL.md
│   ├── 📄 README_PROYECTO_ACTUALIZADO.md
│   │
│   ├── 📁 fases/                        # Documentación por fases
│   │   ├── 📄 FASE4_COMPLETADA.md
│   │   ├── 📄 FASE4_DOCUMENTACION.md
│   │   ├── 📄 FASE5_COMPLETADA.md
│   │   └── 📄 FASE6_COMPLETADA.md
│   │
│   ├── 📁 guias/                        # Guías de usuario
│   │   ├── 📄 INSTRUCCIONES_ENROLLMENT.md
│   │   ├── 📄 COMANDOS_RAPIDOS.md
│   │   └── 📄 quick_start_guide.md
│   │
│   ├── 📁 arquitectura/                 # Documentación técnica
│   │   ├── 📄 DIAGRAMA_FLUJO.md
│   │   ├── 📄 ESTADO_PROYECTO_COMPLETO.md
│   │   └── 📄 database_schema.md
│   │
│   ├── 📁 implementacion/               # Detalles de implementación
│   │   ├── 📄 IMPLEMENTACION_COMPLETADA.md
│   │   └── 📄 RESUMEN_IMPLEMENTACION.md
│   │
│   └── 📁 assets/                       # Imágenes y diagramas
│       ├── 🖼️ img.png
│       ├── 🖼️ img_1.png
│       └── 🖼️ system_architecture.png
│
├── 📁 config/                           # ⚙️ CONFIGURACIÓN
│   ├── 📄 config.yaml
│   ├── 📄 database.yaml
│   ├── 📄 model_config.yaml
│   ├── 📄 pi_config.yaml
│   └── 📄 logging.yaml
│
├── 📁 database/                         # 🗄️ BASE DE DATOS
│   ├── 📄 README.md
│   ├── 📄 schema.sql
│   ├── 📄 init_database.py
│   ├── 🗄️ gloria_stress_system.db
│   │
│   ├── 📁 migrations/
│   │   └── 📄 001_initial_schema.sql
│   │
│   └── 📁 scripts/
│       ├── 📄 backup.sh
│       └── 📄 cleanup.sql
│
├── 📁 models/                           # 🧠 MACHINE LEARNING
│   ├── 📄 README.md
│   │
│   ├── 📁 training/
│   │   ├── 📁 scripts/
│   │   │   ├── 🐍 data_preparation.py
│   │   │   ├── 🐍 train_model.py
│   │   │   ├── 🐍 model_trainer.py
│   │   │   ├── 🐍 model_architecture.py
│   │   │   ├── 🐍 evaluate_model.py
│   │   │   └── 🐍 convert_to_tflite.py
│   │   │
│   │   ├── 📁 configs/
│   │   │   └── 📄 training_config.yaml
│   │   │
│   │   └── 📁 utils/
│   │       ├── 📄 __init__.py
│   │       └── 🐍 data_loader.py
│   │
│   ├── 📁 trained/                      # Modelos entrenados
│   │   ├── 🤖 emotion_detector.h5
│   │   ├── 🤖 emotion_detector.tflite
│   │   ├── 🤖 face_embedder.h5
│   │   └── 📄 model_metadata.json
│   │
│   └── 📁 evaluation/                   # Resultados de evaluación
│       ├── 📊 confusion_matrix.png
│       └── 📄 validation_report.pdf
│
├── 📁 edge/                             # 🔌 RASPBERRY PI / EDGE
│   ├── 📄 README.md
│   ├── 📄 requirements.txt
│   ├── 🐍 main.py
│   │
│   ├── 📁 src/
│   │   ├── 📄 __init__.py
│   │   ├── 🐍 pi_simulator.py
│   │   ├── 🐍 camera_manager.py
│   │   ├── 🐍 face_detector.py
│   │   ├── 🐍 emotion_detector.py
│   │   └── 🐍 websocket_client.py
│   │
│   ├── 📁 config/
│   │   └── 🐍 pi_config.py
│   │
│   ├── 📁 tests/
│   │   ├── 🐍 test_pi_system.py
│   │   └── 🐍 test_system.py
│   │
│   ├── 📁 scripts/
│   │   └── 🐍 start_pi_system.py
│   │
│   └── 📁 logs/
│       └── 📄 .gitkeep
│
├── 📁 backend/                          # 🖥️ SERVIDOR API
│   ├── 📄 README.md
│   ├── 📄 requirements.txt
│   ├── 🐍 main.py
│   │
│   ├── 📁 app/
│   │   ├── 📄 __init__.py
│   │   ├── 🐍 backend_api.py
│   │   ├── 🐍 database.py
│   │   ├── 🐍 config.py
│   │   │
│   │   ├── 📁 models/                   # SQLAlchemy models
│   │   │   ├── 📄 __init__.py
│   │   │   ├── 🐍 employee.py
│   │   │   └── 🐍 detection.py
│   │   │
│   │   ├── 📁 api/
│   │   │   ├── 📄 __init__.py
│   │   │   └── 📁 v1/
│   │   │       ├── 📄 __init__.py
│   │   │       └── 📁 endpoints/
│   │   │           ├── 🐍 employees.py
│   │   │           ├── 🐍 detections.py
│   │   │           └── 🐍 reports.py
│   │   │
│   │   ├── 📁 services/
│   │   │   ├── 📄 __init__.py
│   │   │   ├── 🐍 detection_service.py
│   │   │   └── 🐍 alert_service.py
│   │   │
│   │   └── 📁 utils/
│   │       ├── 📄 __init__.py
│   │       └── 🐍 helpers.py
│   │
│   ├── 📁 scripts/
│   │   ├── 🐍 server_simulator.py
│   │   └── 🐍 start_complete_system.py
│   │
│   └── 📁 logs/
│       └── 📄 .gitkeep
│
├── 📁 enrollment/                       # 👤 SISTEMA DE REGISTRO
│   ├── 📄 README.md
│   ├── 📄 requirements.txt
│   ├── 🐍 enrollment.py
│   ├── 🐍 load_enrollments.py
│   │
│   ├── 📁 data/
│   │   ├── 📄 employees.csv
│   │   └── 📁 enrollments/              # ⚠️ PRIVADO - No en git
│   │       ├── 📄 EMP001_embedding.json
│   │       ├── 📷 EMP001_sample_1.jpg
│   │       ├── 📄 EMP002_embedding.json
│   │       └── 📷 EMP002_sample_1.jpg
│   │
│   ├── 📁 scripts/
│   │   ├── 🐍 capture_photos.py
│   │   └── 🐍 generate_embeddings.py
│   │
│   └── 📁 utils/
│       ├── 📄 __init__.py
│       └── 🐍 face_utils.py
│
├── 📁 reporting/                        # 📊 GENERACIÓN DE REPORTES
│   ├── 📄 README.md
│   ├── 📄 requirements.txt
│   ├── 🐍 report_generator.py
│   │
│   ├── 📁 templates/
│   │   ├── 📄 executive_template.html
│   │   └── 📄 email_template.html
│   │
│   ├── 📁 outputs/
│   │   ├── 📁 daily/
│   │   ├── 📁 weekly/
│   │   └── 📁 monthly/
│   │
│   └── 📁 utils/
│       ├── 📄 __init__.py
│       ├── 🐍 pdf_generator.py
│       └── 🐍 chart_generator.py
│
├── 📁 scripts/                          # 🛠️ SCRIPTS DE UTILIDAD
│   ├── 📁 setup/
│   │   ├── 📄 install_dependencies.sh
│   │   └── 📄 setup_database.sh
│   │
│   ├── 📁 deployment/
│   │   ├── 📄 deploy.sh
│   │   └── 📄 health_check.sh
│   │
│   ├── 📁 maintenance/
│   │   ├── 📄 backup_database.sh
│   │   └── 📄 cleanup_logs.sh
│   │
│   └── 📁 testing/
│       └── 🐍 quick_start.py
│
├── 📁 tests/                            # 🧪 TESTS
│   ├── 📄 __init__.py
│   ├── 📁 integration/
│   │   └── 🐍 test_end_to_end.py
│   │
│   ├── 📁 unit/
│   │   └── 🐍 test_models.py
│   │
│   └── 📁 fixtures/
│       └── 📄 test_data.json
│
├── 📁 data/                             # 💾 DATOS
│   ├── 📁 raw/                          # ⚠️ No en git
│   ├── 📁 processed/                    # ⚠️ No en git
│   ├── 📁 embeddings/                   # ⚠️ No en git
│   └── 📁 exports/                      # ⚠️ No en git
│
├── 📁 logs/                             # 📝 LOGS
│   ├── 📁 backend/
│   ├── 📁 edge/
│   └── 📁 app/
│
├── 📁 notebooks/                        # 📓 JUPYTER NOTEBOOKS
│   ├── 📁 exploratory/
│   ├── 📁 analysis/
│   └── 📁 visualization/
│
└── 📁 deployment/                       # 🚀 DEPLOYMENT
    ├── 📁 docker/
    │   ├── 🐳 backend.Dockerfile
    │   ├── 🐳 edge.Dockerfile
    │   └── 🐳 nginx.Dockerfile
    │
    └── 📁 scripts/
        └── 📄 deploy_production.sh
```

## 🎯 Flujo de Datos

```
┌─────────────────────────────────────────────────────────────────┐
│                    STRESSVISION - FLUJO DE DATOS                │
└─────────────────────────────────────────────────────────────────┘

   📷 Cámara (Raspberry Pi)
        │
        ▼
   ┌─────────────────┐
   │  edge/          │  ← Captura y procesa video
   │  • Camera       │  ← Detección facial
   │  • Face Detect  │  ← Reconocimiento
   │  • Emotion AI   │  ← Inferencia de emociones
   └────────┬────────┘
            │ WebSocket
            ▼
   ┌─────────────────┐
   │  backend/       │  ← Recibe detecciones
   │  • API REST     │  ← Almacena en BD
   │  • WebSocket    │  ← Procesa alertas
   │  • Database     │  ← Genera eventos
   └────────┬────────┘
            │
            ├──────────────────┬──────────────────┐
            ▼                  ▼                  ▼
   ┌────────────────┐  ┌──────────────┐  ┌────────────────┐
   │  database/     │  │ reporting/   │  │ Dashboard      │
   │  • SQLite      │  │ • PDF Gen    │  │ (Futuro)       │
   │  • Histórico   │  │ • Charts     │  │ • React        │
   │  • Analytics   │  │ • Reports    │  │ • Real-time    │
   └────────────────┘  └──────────────┘  └────────────────┘
```

## 🔄 Ciclo de Vida del Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                        CICLO DE VIDA                            │
└─────────────────────────────────────────────────────────────────┘

1. ENTRENAMIENTO (Offline)
   ┌─────────────────────────────────────────┐
   │ models/training/                        │
   │ ├── Preparar datos                      │
   │ ├── Entrenar modelo de emociones        │
   │ ├── Evaluar y optimizar                 │
   │ └── Convertir a TFLite                  │
   └─────────────────────────────────────────┘
                    │
                    ▼
   ┌─────────────────────────────────────────┐
   │ models/trained/                         │
   │ └── emotion_detector.tflite             │
   └─────────────────────────────────────────┘

2. ENROLLMENT (Una vez por empleado)
   ┌─────────────────────────────────────────┐
   │ enrollment/                             │
   │ ├── Capturar fotos del empleado        │
   │ ├── Generar embeddings faciales        │
   │ ├── Guardar en BD                       │
   │ └── Validar calidad                     │
   └─────────────────────────────────────────┘
                    │
                    ▼
   ┌─────────────────────────────────────────┐
   │ enrollment/data/enrollments/            │
   │ ├── EMP001_embedding.json               │
   │ └── EMP001_sample_*.jpg                 │
   └─────────────────────────────────────────┘

3. DETECCIÓN (Tiempo real)
   ┌─────────────────────────────────────────┐
   │ edge/ (Raspberry Pi)                    │
   │ ├── Captura frame de cámara            │
   │ ├── Detecta cara                        │
   │ ├── Reconoce empleado                   │
   │ ├── Detecta emoción                     │
   │ └── Envía al backend                    │
   └─────────────────────────────────────────┘
                    │ WebSocket
                    ▼
   ┌─────────────────────────────────────────┐
   │ backend/                                │
   │ ├── Recibe detección                    │
   │ ├── Valida y almacena                   │
   │ ├── Procesa alertas                     │
   │ └── Notifica dashboard                  │
   └─────────────────────────────────────────┘

4. REPORTES (Bajo demanda)
   ┌─────────────────────────────────────────┐
   │ reporting/                              │
   │ ├── Consulta histórico                  │
   │ ├── Genera estadísticas                 │
   │ ├── Crea gráficos                       │
   │ └── Exporta PDF                         │
   └─────────────────────────────────────────┘
```

## 📦 Dependencias entre Módulos

```
backend/
  ↓ usa
  ├── database/ (schema, conexión)
  ├── models/trained/ (para validación)
  └── config/ (configuración)

edge/
  ↓ usa
  ├── models/trained/ (inferencia)
  ├── config/ (parámetros)
  └── backend/ (envía datos via WebSocket)

enrollment/
  ↓ usa
  ├── database/ (guarda embeddings)
  ├── models/trained/ (genera embeddings)
  └── config/ (configuración)

reporting/
  ↓ usa
  ├── database/ (consulta datos)
  └── config/ (templates, configuración)

models/training/
  ↓ genera
  └── models/trained/ (modelos .h5, .tflite)
```

## 🔐 Datos Sensibles (No en Git)

```
⚠️ PRIVACIDAD - Estos datos NO deben subirse a git:

enrollment/data/enrollments/
  ├── *.jpg                    # Fotos de empleados
  └── *_embedding.json         # Embeddings faciales

database/
  └── gloria_stress_system.db  # Base de datos con datos personales

data/
  ├── raw/                     # Datos crudos
  ├── processed/               # Datos procesados
  └── embeddings/              # Embeddings

models/trained/
  └── *.h5                     # Modelos grandes (usar Git LFS si es necesario)

logs/
  └── **/*.log                 # Logs pueden contener info sensible
```

## 🚀 Puntos de Entrada Principales

```
# Entrenar modelo
python models/training/scripts/train_model.py

# Registrar empleado
python enrollment/enrollment.py --employee-id EMP001

# Iniciar backend
python backend/main.py

# Iniciar edge device
python edge/scripts/start_pi_system.py

# Iniciar sistema completo
python backend/scripts/start_complete_system.py

# Generar reporte
python reporting/report_generator.py --type weekly

# Tests
python -m pytest tests/

# Quick start (demo)
python scripts/testing/quick_start.py
```

## 📊 Métricas del Proyecto

```
Total de Módulos:         7
  ├── backend             (API REST + WebSocket)
  ├── edge                (Raspberry Pi)
  ├── models              (ML Training)
  ├── enrollment          (Registro)
  ├── reporting           (Reportes)
  ├── database            (Persistencia)
  └── docs                (Documentación)

Lenguajes:
  ├── Python 3.8+         (Backend, ML, Edge)
  ├── SQL                 (Database)
  └── Markdown            (Docs)

Frameworks Principales:
  ├── FastAPI/Flask       (Backend API)
  ├── TensorFlow/Keras    (ML)
  ├── OpenCV              (Visión)
  ├── SQLite              (Database)
  └── Socket.IO           (WebSocket)
```

## 🎓 Referencias Rápidas

| Necesito...                    | Ir a...                              |
|--------------------------------|--------------------------------------|
| Entrenar un modelo             | `models/training/scripts/`           |
| Registrar empleado             | `enrollment/enrollment.py`           |
| Configurar API                 | `backend/app/backend_api.py`         |
| Configurar Raspberry Pi        | `edge/config/pi_config.py`           |
| Ver documentación de fase      | `docs/fases/`                        |
| Ver guías de uso               | `docs/guias/`                        |
| Generar reportes               | `reporting/report_generator.py`      |
| Consultar esquema de BD        | `database/schema.sql`                |
| Scripts de deployment          | `scripts/deployment/`                |
| Tests                          | `tests/`                             |

---

**Leyenda:**
- 📁 Carpeta
- 📄 Archivo de documentación
- 🐍 Archivo Python
- 🤖 Modelo ML
- 📊 Gráfico/Reporte
- 🗄️ Base de datos
- 🖼️ Imagen
- 🐳 Dockerfile
- ⚠️ Sensible/Privado


