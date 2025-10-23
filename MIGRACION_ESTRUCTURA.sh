#!/bin/bash

# Script de Migración de Estructura - StressVision
# Este script reorganiza los archivos del proyecto a la nueva estructura

set -e  # Salir si hay algún error

echo "=========================================="
echo "  MIGRACIÓN DE ESTRUCTURA - STRESSVISION"
echo "=========================================="
echo ""
echo "Este script reorganizará tu proyecto a la nueva estructura."
echo "Se creará un backup antes de comenzar."
echo ""
read -p "¿Deseas continuar? (s/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Ss]$ ]]
then
    echo "Migración cancelada."
    exit 1
fi

# Crear backup
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
echo "Creando backup en $BACKUP_DIR..."
mkdir -p "$BACKUP_DIR"
cp -r . "$BACKUP_DIR/" 2>/dev/null || true
echo "✓ Backup creado"

# Función para crear directorio si no existe
create_dir() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
        echo "  Creado: $1"
    fi
}

# Función para mover archivo
move_file() {
    if [ -f "$1" ]; then
        mkdir -p "$(dirname "$2")"
        mv "$1" "$2"
        echo "  ✓ Movido: $1 → $2"
    fi
}

echo ""
echo "Creando estructura de carpetas..."

# Crear todas las carpetas necesarias
create_dir "docs"
create_dir "docs/fases"
create_dir "docs/guias"
create_dir "docs/arquitectura"
create_dir "docs/implementacion"
create_dir "docs/assets"

create_dir "config"

create_dir "database"
create_dir "database/migrations"
create_dir "database/scripts"

create_dir "models"
create_dir "models/training"
create_dir "models/training/scripts"
create_dir "models/training/configs"
create_dir "models/training/utils"
create_dir "models/trained"
create_dir "models/evaluation"

create_dir "edge"
create_dir "edge/src"
create_dir "edge/config"
create_dir "edge/tests"
create_dir "edge/scripts"
create_dir "edge/utils"
create_dir "edge/logs"

create_dir "backend"
create_dir "backend/app"
create_dir "backend/app/models"
create_dir "backend/app/api"
create_dir "backend/app/api/v1"
create_dir "backend/app/api/v1/endpoints"
create_dir "backend/app/services"
create_dir "backend/app/utils"
create_dir "backend/tests"
create_dir "backend/scripts"
create_dir "backend/logs"

create_dir "enrollment"
create_dir "enrollment/data"
create_dir "enrollment/data/enrollments"
create_dir "enrollment/scripts"
create_dir "enrollment/utils"

create_dir "reporting"
create_dir "reporting/templates"
create_dir "reporting/outputs"
create_dir "reporting/outputs/daily"
create_dir "reporting/outputs/weekly"
create_dir "reporting/outputs/monthly"
create_dir "reporting/utils"

create_dir "scripts"
create_dir "scripts/setup"
create_dir "scripts/deployment"
create_dir "scripts/maintenance"
create_dir "scripts/testing"

create_dir "tests"
create_dir "tests/integration"
create_dir "tests/unit"
create_dir "tests/fixtures"

create_dir "data"
create_dir "data/raw"
create_dir "data/processed"
create_dir "data/embeddings"
create_dir "data/exports"

create_dir "logs"
create_dir "logs/backend"
create_dir "logs/edge"
create_dir "logs/app"

create_dir "notebooks"
create_dir "notebooks/exploratory"
create_dir "notebooks/analysis"
create_dir "notebooks/visualization"

create_dir "deployment"
create_dir "deployment/docker"
create_dir "deployment/scripts"

echo "✓ Estructura de carpetas creada"

echo ""
echo "Moviendo archivos de documentación..."
move_file "PROYECTO_COMPLETO_FINAL.md" "docs/PROYECTO_COMPLETO_FINAL.md"
move_file "README_PROYECTO_ACTUALIZADO.md" "docs/README_PROYECTO_ACTUALIZADO.md"
move_file "FASE4_COMPLETADA.md" "docs/fases/FASE4_COMPLETADA.md"
move_file "FASE4_DOCUMENTACION.md" "docs/fases/FASE4_DOCUMENTACION.md"
move_file "FASE4_QUICK_START.md" "docs/fases/FASE4_QUICK_START.md"
move_file "FASE5_COMPLETADA.md" "docs/fases/FASE5_COMPLETADA.md"
move_file "FASE5_DOCUMENTACION.md" "docs/fases/FASE5_DOCUMENTACION.md"
move_file "FASE5_QUICK_START.md" "docs/fases/FASE5_QUICK_START.md"
move_file "FASE6_COMPLETADA.md" "docs/fases/FASE6_COMPLETADA.md"
move_file "FASE6_DOCUMENTACION.md" "docs/fases/FASE6_DOCUMENTACION.md"
move_file "FASE6_QUICK_START.md" "docs/fases/FASE6_QUICK_START.md"
move_file "INSTRUCCIONES_ENROLLMENT.md" "docs/guias/INSTRUCCIONES_ENROLLMENT.md"
move_file "COMANDOS_RAPIDOS.md" "docs/guias/COMANDOS_RAPIDOS.md"
move_file "DIAGRAMA_FLUJO.md" "docs/arquitectura/DIAGRAMA_FLUJO.md"
move_file "ESTADO_PROYECTO_COMPLETO.md" "docs/arquitectura/ESTADO_PROYECTO_COMPLETO.md"
move_file "IMPLEMENTACION_COMPLETADA.md" "docs/implementacion/IMPLEMENTACION_COMPLETADA.md"
move_file "RESUMEN_IMPLEMENTACION.md" "docs/implementacion/RESUMEN_IMPLEMENTACION.md"
move_file "img.png" "docs/assets/img.png"
move_file "img_1.png" "docs/assets/img_1.png"
move_file "img_2.png" "docs/assets/img_2.png"
move_file "img_3.png" "docs/assets/img_3.png"
move_file "img_4.png" "docs/assets/img_4.png"
move_file "img_5.png" "docs/assets/img_5.png"

echo ""
echo "Moviendo archivos de backend..."
move_file "backend_api.py" "backend/app/backend_api.py"
move_file "main.py" "backend/main.py"
move_file "server_simulator.py" "backend/scripts/server_simulator.py"
move_file "start_complete_system.py" "backend/scripts/start_complete_system.py"

echo ""
echo "Moviendo archivos de modelos..."
move_file "model_architecture.py" "models/training/scripts/model_architecture.py"
move_file "model_trainer.py" "models/training/scripts/model_trainer.py"
move_file "train_model.py" "models/training/scripts/train_model.py"
move_file "data_preparation.py" "models/training/scripts/data_preparation.py"
move_file "evaluate_model.py" "models/training/scripts/evaluate_model.py"
move_file "convert_to_tflite.py" "models/training/scripts/convert_to_tflite.py"

echo ""
echo "Moviendo archivos de edge/Pi..."
move_file "pi_config.py" "edge/config/pi_config.py"
move_file "pi_simulator.py" "edge/src/pi_simulator.py"
move_file "start_pi_system.py" "edge/scripts/start_pi_system.py"
move_file "test_pi_system.py" "edge/tests/test_pi_system.py"
move_file "test_system.py" "edge/tests/test_system.py"

echo ""
echo "Moviendo archivos de enrollment..."
move_file "enrollment.py" "enrollment/enrollment.py"
move_file "load_enrollments.py" "enrollment/load_enrollments.py"
if [ -d "enrollments" ]; then
    mv enrollments/* enrollment/data/enrollments/ 2>/dev/null || true
    rmdir enrollments 2>/dev/null || true
    echo "  ✓ Movido: enrollments/* → enrollment/data/enrollments/"
fi

echo ""
echo "Moviendo archivos de database..."
move_file "gloria_stress_system.db" "database/gloria_stress_system.db"
move_file "init_database.py" "database/init_database.py"

echo ""
echo "Moviendo archivos de reporting..."
move_file "report_generator.py" "reporting/report_generator.py"

echo ""
echo "Moviendo scripts..."
move_file "quick_start.py" "scripts/testing/quick_start.py"

echo ""
echo "Creando archivos .gitkeep en carpetas vacías..."
touch edge/logs/.gitkeep
touch backend/logs/.gitkeep
touch logs/backend/.gitkeep
touch logs/edge/.gitkeep
touch logs/app/.gitkeep
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch data/embeddings/.gitkeep
touch data/exports/.gitkeep
touch models/trained/.gitkeep
touch models/evaluation/.gitkeep
touch reporting/outputs/daily/.gitkeep
touch reporting/outputs/weekly/.gitkeep
touch reporting/outputs/monthly/.gitkeep
touch tests/fixtures/.gitkeep

echo ""
echo "Creando archivos __init__.py..."
touch edge/src/__init__.py
touch edge/utils/__init__.py
touch backend/app/__init__.py
touch backend/app/models/__init__.py
touch backend/app/api/__init__.py
touch backend/app/api/v1/__init__.py
touch backend/app/api/v1/endpoints/__init__.py
touch backend/app/services/__init__.py
touch backend/app/utils/__init__.py
touch backend/tests/__init__.py
touch enrollment/utils/__init__.py
touch reporting/utils/__init__.py
touch models/training/utils/__init__.py
touch tests/__init__.py

echo ""
echo "Creando archivo .env.example..."
cat > .env.example << 'EOF'
# StressVision - Configuración de Entorno

# Database
DATABASE_URL=sqlite:///database/gloria_stress_system.db

# API
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=False

# Edge Device
EDGE_DEVICE_ID=PI001
SERVER_URL=http://localhost:8000

# Model Paths
EMOTION_MODEL_PATH=models/trained/emotion_detector.tflite
FACE_EMBEDDER_PATH=models/trained/face_embedder.tflite

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app/stressvision.log

# Alert Thresholds
STRESS_THRESHOLD=0.7
ALERT_COOLDOWN_MINUTES=30

# Security
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=localhost,127.0.0.1

# Camera
CAMERA_INDEX=0
CAMERA_WIDTH=640
CAMERA_HEIGHT=480
CAMERA_FPS=30
EOF

echo ""
echo "Creando README.md en carpetas principales..."

cat > docs/README.md << 'EOF'
# Documentación del Proyecto StressVision

Esta carpeta contiene toda la documentación del proyecto organizada por categorías.

## Estructura

- `fases/`: Documentación de cada fase del desarrollo
- `guias/`: Guías de uso y quick starts
- `arquitectura/`: Documentación técnica y diagramas
- `implementacion/`: Detalles de implementación
- `assets/`: Imágenes y diagramas
EOF

cat > models/README.md << 'EOF'
# Modelos de Machine Learning

Esta carpeta contiene todo lo relacionado con los modelos de ML del proyecto.

## Estructura

- `training/`: Scripts y configuraciones para entrenar modelos
- `trained/`: Modelos entrenados listos para usar
- `evaluation/`: Resultados de evaluación y métricas
EOF

cat > edge/README.md << 'EOF'
# Edge Device (Raspberry Pi)

Código y configuración para dispositivos edge (Raspberry Pi).

## Estructura

- `src/`: Código fuente principal
- `config/`: Archivos de configuración
- `tests/`: Tests específicos del edge
- `scripts/`: Scripts de utilidad
EOF

cat > backend/README.md << 'EOF'
# Backend API

Servidor backend y API REST del sistema.

## Estructura

- `app/`: Aplicación principal
- `tests/`: Tests del backend
- `scripts/`: Scripts de utilidad
EOF

cat > enrollment/README.md << 'EOF'
# Sistema de Enrollment

Sistema de registro de empleados y generación de embeddings faciales.

## Estructura

- `data/`: Datos de enrollment (fotos y embeddings)
- `scripts/`: Scripts de utilidad
- `utils/`: Utilidades compartidas
EOF

cat > reporting/README.md << 'EOF'
# Sistema de Reportes

Generación de reportes automáticos y bajo demanda.

## Estructura

- `templates/`: Plantillas HTML/PDF
- `outputs/`: Reportes generados
- `utils/`: Utilidades para generación de reportes
EOF

echo ""
echo "=========================================="
echo "  ✓ MIGRACIÓN COMPLETADA"
echo "=========================================="
echo ""
echo "Próximos pasos:"
echo "1. Revisar que todos los archivos se movieron correctamente"
echo "2. Actualizar imports en archivos Python"
echo "3. Actualizar rutas en archivos de configuración"
echo "4. Probar que el sistema funciona"
echo "5. Revisar y actualizar .gitignore"
echo "6. Commit de la nueva estructura"
echo ""
echo "Backup guardado en: $BACKUP_DIR"
echo ""
echo "Para deshacer la migración:"
echo "  rm -rf docs models edge backend enrollment reporting scripts tests deployment"
echo "  cp -r $BACKUP_DIR/* ."
echo ""


