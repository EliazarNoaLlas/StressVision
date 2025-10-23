# Estructura de Carpetas Propuesta - StressVision

## Estructura Completa

```
StressVision/
│
├── README.md
├── LICENSE
├── .gitignore
├── .env.example
├── requirements.txt
├── docker-compose.yml
│
├── docs/                                    # Documentación del proyecto
│   ├── README.md
│   ├── PROYECTO_COMPLETO_FINAL.md
│   ├── README_PROYECTO_ACTUALIZADO.md
│   │
│   ├── fases/                              # Documentación por fases
│   │   ├── FASE4_COMPLETADA.md
│   │   ├── FASE4_DOCUMENTACION.md
│   │   ├── FASE4_QUICK_START.md
│   │   ├── FASE5_COMPLETADA.md
│   │   ├── FASE5_DOCUMENTACION.md
│   │   ├── FASE5_QUICK_START.md
│   │   ├── FASE6_COMPLETADA.md
│   │   ├── FASE6_DOCUMENTACION.md
│   │   └── FASE6_QUICK_START.md
│   │
│   ├── guias/                              # Guías de uso
│   │   ├── INSTRUCCIONES_ENROLLMENT.md
│   │   ├── COMANDOS_RAPIDOS.md
│   │   └── quick_start_guide.md
│   │
│   ├── arquitectura/                       # Documentación técnica
│   │   ├── DIAGRAMA_FLUJO.md
│   │   ├── ESTADO_PROYECTO_COMPLETO.md
│   │   ├── architecture.md
│   │   └── database_schema.md
│   │
│   ├── implementacion/                     # Documentación de implementación
│   │   ├── IMPLEMENTACION_COMPLETADA.md
│   │   └── RESUMEN_IMPLEMENTACION.md
│   │
│   └── assets/                             # Imágenes y diagramas
│       ├── img.png
│       ├── img_1.png
│       ├── img_2.png
│       ├── img_3.png
│       ├── img_4.png
│       └── img_5.png
│
├── config/                                  # Configuración centralizada
│   ├── config.yaml
│   ├── database.yaml
│   ├── model_config.yaml
│   ├── pi_config.yaml                      # Ya existe (pi_config.py)
│   ├── alert_thresholds.yaml
│   └── logging.yaml
│
├── database/                                # Base de datos y migraciones
│   ├── README.md
│   ├── schema.sql
│   ├── gloria_stress_system.db             # Base de datos actual
│   ├── init_database.py                    # Script de inicialización
│   │
│   ├── migrations/
│   │   └── README.md
│   │
│   └── scripts/
│       ├── backup.sh
│       └── cleanup.sql
│
├── models/                                  # Modelos ML y entrenamiento
│   ├── README.md
│   │
│   ├── training/                           # Scripts de entrenamiento
│   │   ├── scripts/
│   │   │   ├── data_preparation.py         # Ya existe
│   │   │   ├── train_model.py              # Ya existe
│   │   │   ├── model_trainer.py            # Ya existe
│   │   │   ├── model_architecture.py       # Ya existe
│   │   │   ├── evaluate_model.py           # Ya existe
│   │   │   └── convert_to_tflite.py        # Ya existe
│   │   │
│   │   ├── configs/
│   │   │   └── training_config.yaml
│   │   │
│   │   └── utils/
│   │       └── __init__.py
│   │
│   ├── trained/                            # Modelos entrenados
│   │   ├── emotion_detector.h5
│   │   ├── emotion_detector.tflite
│   │   ├── face_embedder.h5
│   │   └── model_metadata.json
│   │
│   └── evaluation/                         # Resultados de evaluación
│       ├── confusion_matrix.png
│       ├── metrics.json
│       └── validation_report.pdf
│
├── edge/                                   # Código Raspberry Pi
│   ├── README.md
│   ├── requirements.txt
│   ├── main.py
│   │
│   ├── src/
│   │   ├── __init__.py
│   │   ├── pi_simulator.py                # Ya existe
│   │   ├── camera_manager.py
│   │   ├── face_detector.py
│   │   ├── emotion_detector.py
│   │   └── websocket_client.py
│   │
│   ├── config/
│   │   └── pi_config.yaml
│   │
│   ├── tests/
│   │   ├── test_pi_system.py              # Ya existe
│   │   └── test_system.py                 # Ya existe
│   │
│   ├── scripts/
│   │   ├── start_pi_system.py             # Ya existe
│   │   └── test_connection.py
│   │
│   └── logs/
│       └── .gitkeep
│
├── backend/                                # Backend API y servicios
│   ├── README.md
│   ├── requirements.txt
│   ├── main.py                            # Ya existe
│   │
│   ├── app/
│   │   ├── __init__.py
│   │   ├── backend_api.py                 # Ya existe
│   │   ├── database.py
│   │   ├── config.py
│   │   │
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── employee.py
│   │   │   ├── detection.py
│   │   │   └── alert.py
│   │   │
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── v1/
│   │   │       ├── __init__.py
│   │   │       └── endpoints/
│   │   │           ├── employees.py
│   │   │           ├── detections.py
│   │   │           ├── alerts.py
│   │   │           └── reports.py
│   │   │
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── detection_service.py
│   │   │   ├── alert_service.py
│   │   │   └── report_service.py
│   │   │
│   │   └── utils/
│   │       ├── __init__.py
│   │       └── helpers.py
│   │
│   ├── tests/
│   │   └── __init__.py
│   │
│   ├── scripts/
│   │   ├── server_simulator.py            # Ya existe
│   │   └── start_complete_system.py       # Ya existe
│   │
│   └── logs/
│       └── .gitkeep
│
├── enrollment/                             # Sistema de registro
│   ├── README.md
│   ├── requirements.txt
│   ├── enrollment.py                      # Ya existe
│   ├── load_enrollments.py                # Ya existe
│   │
│   ├── data/
│   │   ├── enrollments/                   # Carpeta actual
│   │   │   ├── EMP001_embedding.json
│   │   │   ├── EMP001_sample_*.jpg
│   │   │   ├── EMP002_embedding.json
│   │   │   └── EMP002_sample_*.jpg
│   │   │
│   │   └── employees.csv
│   │
│   ├── scripts/
│   │   ├── capture_photos.py
│   │   ├── generate_embeddings.py
│   │   └── validate_enrollment.py
│   │
│   └── utils/
│       ├── __init__.py
│       └── face_utils.py
│
├── reporting/                              # Generación de reportes
│   ├── README.md
│   ├── requirements.txt
│   ├── report_generator.py                # Ya existe
│   │
│   ├── templates/
│   │   ├── executive_template.html
│   │   └── email_template.html
│   │
│   ├── outputs/
│   │   ├── daily/
│   │   ├── weekly/
│   │   └── monthly/
│   │
│   └── utils/
│       ├── __init__.py
│       ├── pdf_generator.py
│       └── chart_generator.py
│
├── scripts/                                # Scripts de utilidad
│   ├── setup/
│   │   ├── install_dependencies.sh
│   │   ├── setup_database.sh
│   │   └── setup_environment.sh
│   │
│   ├── deployment/
│   │   ├── deploy.sh
│   │   └── health_check.sh
│   │
│   ├── maintenance/
│   │   ├── backup_database.sh
│   │   ├── cleanup_logs.sh
│   │   └── restart_services.sh
│   │
│   └── testing/
│       ├── run_all_tests.sh
│       └── quick_start.py                 # Ya existe
│
├── tests/                                  # Tests integrados
│   ├── integration/
│   │   ├── test_end_to_end.py
│   │   └── test_api_integration.py
│   │
│   ├── unit/
│   │   ├── test_models.py
│   │   └── test_services.py
│   │
│   └── fixtures/
│       ├── sample_images/
│       └── test_data.json
│
├── data/                                   # Datos del proyecto
│   ├── raw/
│   │   └── .gitkeep
│   │
│   ├── processed/
│   │   └── .gitkeep
│   │
│   ├── embeddings/
│   │   └── .gitkeep
│   │
│   └── exports/
│       └── .gitkeep
│
├── logs/                                   # Logs del sistema
│   ├── backend/
│   │   └── .gitkeep
│   ├── edge/
│   │   └── .gitkeep
│   └── app/
│       └── .gitkeep
│
├── notebooks/                              # Jupyter notebooks
│   ├── exploratory/
│   │   └── 01_dataset_exploration.ipynb
│   │
│   ├── analysis/
│   │   └── stress_patterns.ipynb
│   │
│   └── visualization/
│       └── dashboard_prototypes.ipynb
│
├── deployment/                             # Configuración de despliegue
│   ├── docker/
│   │   ├── backend.Dockerfile
│   │   ├── edge.Dockerfile
│   │   └── nginx.Dockerfile
│   │
│   └── scripts/
│       └── deploy_production.sh
│
└── __pycache__/                           # Cache de Python (en .gitignore)
```

## Mapeo de Archivos Actuales a Nueva Estructura

### Archivos Raíz → Nueva Ubicación

```
ACTUAL                              →    NUEVA UBICACIÓN
─────────────────────────────────────────────────────────────────────
README.md                           →    README.md (actualizado)
requirements.txt                    →    requirements.txt

# Documentación
PROYECTO_COMPLETO_FINAL.md          →    docs/PROYECTO_COMPLETO_FINAL.md
README_PROYECTO_ACTUALIZADO.md      →    docs/README_PROYECTO_ACTUALIZADO.md
FASE4_*.md                          →    docs/fases/FASE4_*.md
FASE5_*.md                          →    docs/fases/FASE5_*.md
FASE6_*.md                          →    docs/fases/FASE6_*.md
INSTRUCCIONES_ENROLLMENT.md         →    docs/guias/INSTRUCCIONES_ENROLLMENT.md
COMANDOS_RAPIDOS.md                 →    docs/guias/COMANDOS_RAPIDOS.md
DIAGRAMA_FLUJO.md                   →    docs/arquitectura/DIAGRAMA_FLUJO.md
ESTADO_PROYECTO_COMPLETO.md         →    docs/arquitectura/ESTADO_PROYECTO_COMPLETO.md
IMPLEMENTACION_COMPLETADA.md        →    docs/implementacion/IMPLEMENTACION_COMPLETADA.md
RESUMEN_IMPLEMENTACION.md           →    docs/implementacion/RESUMEN_IMPLEMENTACION.md
img*.png                            →    docs/assets/img*.png

# Backend
backend_api.py                      →    backend/app/backend_api.py
main.py                             →    backend/main.py
server_simulator.py                 →    backend/scripts/server_simulator.py
start_complete_system.py            →    backend/scripts/start_complete_system.py

# Modelos y Entrenamiento
model_architecture.py               →    models/training/scripts/model_architecture.py
model_trainer.py                    →    models/training/scripts/model_trainer.py
train_model.py                      →    models/training/scripts/train_model.py
data_preparation.py                 →    models/training/scripts/data_preparation.py
evaluate_model.py                   →    models/training/scripts/evaluate_model.py
convert_to_tflite.py                →    models/training/scripts/convert_to_tflite.py

# Edge/Raspberry Pi
pi_config.py                        →    edge/config/pi_config.py
pi_simulator.py                     →    edge/src/pi_simulator.py
start_pi_system.py                  →    edge/scripts/start_pi_system.py
test_pi_system.py                   →    edge/tests/test_pi_system.py
test_system.py                      →    edge/tests/test_system.py

# Enrollment
enrollment.py                       →    enrollment/enrollment.py
load_enrollments.py                 →    enrollment/load_enrollments.py
enrollments/*                       →    enrollment/data/enrollments/*

# Database
gloria_stress_system.db             →    database/gloria_stress_system.db
init_database.py                    →    database/init_database.py

# Reporting
report_generator.py                 →    reporting/report_generator.py

# Scripts
quick_start.py                      →    scripts/testing/quick_start.py

# Data
data/*                              →    data/raw/*

# Archivos temporales/testing
proyectopruba.py                    →    [ELIMINAR o mover a scripts/testing/]
```

## Cambios Recomendados en .gitignore

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# IDEs
.idea/
.vscode/
*.swp
*.swo
*~

# Environments
.env
.venv
env/
venv/

# Database
*.db
*.sqlite
*.sqlite3
database/*.db
!database/schema.sql

# Logs
logs/**/*
!logs/**/.gitkeep
*.log

# Models (archivos grandes)
models/trained/*.h5
models/trained/*.tflite
models/pretrained/*.h5

# Data
data/raw/**/*
data/processed/**/*
data/embeddings/**/*
data/exports/**/*
!data/**/.gitkeep

# Enrollment data (privacidad)
enrollment/data/enrollments/*.jpg
enrollment/data/enrollments/*.json
!enrollment/data/enrollments/.gitkeep

# Reports outputs
reporting/outputs/**/*
!reporting/outputs/**/.gitkeep

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# OS
.DS_Store
Thumbs.db
```

## Ventajas de Esta Estructura

### 1. **Separación Clara de Responsabilidades**
- `backend/`: Todo el código del servidor
- `edge/`: Todo lo relacionado con Raspberry Pi
- `models/`: Entrenamiento y modelos ML
- `enrollment/`: Sistema de registro separado
- `reporting/`: Generación de reportes

### 2. **Documentación Organizada**
- Por fases del proyecto
- Por tipo (guías, arquitectura, implementación)
- Assets separados

### 3. **Configuración Centralizada**
- Todos los archivos de configuración en `config/`
- Fácil de encontrar y modificar

### 4. **Escalabilidad**
- Estructura preparada para crecer
- Fácil agregar nuevos módulos
- Cada componente independiente

### 5. **Mejor para Equipos**
- Estructura profesional estándar
- Fácil onboarding de nuevos desarrolladores
- Separación clara de componentes

### 6. **CI/CD Ready**
- Estructura preparada para Docker
- Separación backend/frontend/edge clara
- Scripts de deployment organizados

## Próximos Pasos Sugeridos

1. **Crear estructura base de carpetas**
2. **Mover archivos según mapeo**
3. **Actualizar imports en código Python**
4. **Crear archivos `__init__.py` necesarios**
5. **Actualizar `requirements.txt` por módulo**
6. **Crear README.md en cada carpeta principal**
7. **Actualizar .gitignore**
8. **Crear archivos .gitkeep en carpetas vacías**
9. **Probar que todo funcione después de la reorganización**
10. **Commit de la nueva estructura**

## Comandos de Migración

Ver archivo `MIGRACION_ESTRUCTURA.sh` para script automatizado de migración.
```



