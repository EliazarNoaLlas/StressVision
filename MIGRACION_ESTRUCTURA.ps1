# Script de Migración de Estructura - StressVision (PowerShell)
# Para ejecutar: .\MIGRACION_ESTRUCTURA.ps1

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  MIGRACIÓN DE ESTRUCTURA - STRESSVISION" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Este script reorganizará tu proyecto a la nueva estructura." -ForegroundColor Yellow
Write-Host "Se creará un backup antes de comenzar." -ForegroundColor Yellow
Write-Host ""

$confirmation = Read-Host "¿Deseas continuar? (s/n)"
if ($confirmation -ne 's' -and $confirmation -ne 'S') {
    Write-Host "Migración cancelada." -ForegroundColor Red
    exit
}

# Crear backup
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backupDir = "backup_$timestamp"
Write-Host "`nCreando backup en $backupDir..." -ForegroundColor Green
New-Item -ItemType Directory -Path $backupDir -Force | Out-Null

# Copiar todo excepto __pycache__ y venv
Get-ChildItem -Path "." -Exclude $backupDir,"__pycache__","venv","env",".git" | 
    Copy-Item -Destination $backupDir -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "✓ Backup creado" -ForegroundColor Green

# Función para crear directorio si no existe
function Create-Directory {
    param([string]$Path)
    if (!(Test-Path -Path $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
        Write-Host "  Creado: $Path" -ForegroundColor Gray
    }
}

# Función para mover archivo
function Move-FileIfExists {
    param(
        [string]$Source,
        [string]$Destination
    )
    if (Test-Path -Path $Source -PathType Leaf) {
        $destDir = Split-Path -Path $Destination -Parent
        if ($destDir -and !(Test-Path -Path $destDir)) {
            New-Item -ItemType Directory -Path $destDir -Force | Out-Null
        }
        Move-Item -Path $Source -Destination $Destination -Force
        Write-Host "  ✓ Movido: $Source → $Destination" -ForegroundColor Green
    }
}

Write-Host "`nCreando estructura de carpetas..." -ForegroundColor Cyan

# Crear todas las carpetas necesarias
$directories = @(
    "docs",
    "docs/fases",
    "docs/guias",
    "docs/arquitectura",
    "docs/implementacion",
    "docs/assets",
    "config",
    "database",
    "database/migrations",
    "database/scripts",
    "models",
    "models/training",
    "models/training/scripts",
    "models/training/configs",
    "models/training/utils",
    "models/trained",
    "models/evaluation",
    "edge",
    "edge/src",
    "edge/config",
    "edge/tests",
    "edge/scripts",
    "edge/utils",
    "edge/logs",
    "backend",
    "backend/app",
    "backend/app/models",
    "backend/app/api",
    "backend/app/api/v1",
    "backend/app/api/v1/endpoints",
    "backend/app/services",
    "backend/app/utils",
    "backend/tests",
    "backend/scripts",
    "backend/logs",
    "enrollment",
    "enrollment/data",
    "enrollment/data/enrollments",
    "enrollment/scripts",
    "enrollment/utils",
    "reporting",
    "reporting/templates",
    "reporting/outputs",
    "reporting/outputs/daily",
    "reporting/outputs/weekly",
    "reporting/outputs/monthly",
    "reporting/utils",
    "scripts",
    "scripts/setup",
    "scripts/deployment",
    "scripts/maintenance",
    "scripts/testing",
    "tests",
    "tests/integration",
    "tests/unit",
    "tests/fixtures",
    "data",
    "data/raw",
    "data/processed",
    "data/embeddings",
    "data/exports",
    "logs",
    "logs/backend",
    "logs/edge",
    "logs/app",
    "notebooks",
    "notebooks/exploratory",
    "notebooks/analysis",
    "notebooks/visualization",
    "deployment",
    "deployment/docker",
    "deployment/scripts"
)

foreach ($dir in $directories) {
    Create-Directory -Path $dir
}

Write-Host "✓ Estructura de carpetas creada" -ForegroundColor Green

# Mover archivos de documentación
Write-Host "`nMoviendo archivos de documentación..." -ForegroundColor Cyan
Move-FileIfExists "PROYECTO_COMPLETO_FINAL.md" "docs/PROYECTO_COMPLETO_FINAL.md"
Move-FileIfExists "README_PROYECTO_ACTUALIZADO.md" "docs/README_PROYECTO_ACTUALIZADO.md"
Move-FileIfExists "FASE4_COMPLETADA.md" "docs/fases/FASE4_COMPLETADA.md"
Move-FileIfExists "FASE4_DOCUMENTACION.md" "docs/fases/FASE4_DOCUMENTACION.md"
Move-FileIfExists "FASE4_QUICK_START.md" "docs/fases/FASE4_QUICK_START.md"
Move-FileIfExists "FASE5_COMPLETADA.md" "docs/fases/FASE5_COMPLETADA.md"
Move-FileIfExists "FASE5_DOCUMENTACION.md" "docs/fases/FASE5_DOCUMENTACION.md"
Move-FileIfExists "FASE5_QUICK_START.md" "docs/fases/FASE5_QUICK_START.md"
Move-FileIfExists "FASE6_COMPLETADA.md" "docs/fases/FASE6_COMPLETADA.md"
Move-FileIfExists "FASE6_DOCUMENTACION.md" "docs/fases/FASE6_DOCUMENTACION.md"
Move-FileIfExists "FASE6_QUICK_START.md" "docs/fases/FASE6_QUICK_START.md"
Move-FileIfExists "INSTRUCCIONES_ENROLLMENT.md" "docs/guias/INSTRUCCIONES_ENROLLMENT.md"
Move-FileIfExists "COMANDOS_RAPIDOS.md" "docs/guias/COMANDOS_RAPIDOS.md"
Move-FileIfExists "DIAGRAMA_FLUJO.md" "docs/arquitectura/DIAGRAMA_FLUJO.md"
Move-FileIfExists "ESTADO_PROYECTO_COMPLETO.md" "docs/arquitectura/ESTADO_PROYECTO_COMPLETO.md"
Move-FileIfExists "IMPLEMENTACION_COMPLETADA.md" "docs/implementacion/IMPLEMENTACION_COMPLETADA.md"
Move-FileIfExists "RESUMEN_IMPLEMENTACION.md" "docs/implementacion/RESUMEN_IMPLEMENTACION.md"
Move-FileIfExists "img.png" "docs/assets/img.png"
Move-FileIfExists "img_1.png" "docs/assets/img_1.png"
Move-FileIfExists "img_2.png" "docs/assets/img_2.png"
Move-FileIfExists "img_3.png" "docs/assets/img_3.png"
Move-FileIfExists "img_4.png" "docs/assets/img_4.png"
Move-FileIfExists "img_5.png" "docs/assets/img_5.png"

# Mover archivos de backend
Write-Host "`nMoviendo archivos de backend..." -ForegroundColor Cyan
Move-FileIfExists "backend_api.py" "backend/app/backend_api.py"
Move-FileIfExists "main.py" "backend/main.py"
Move-FileIfExists "server_simulator.py" "backend/scripts/server_simulator.py"
Move-FileIfExists "start_complete_system.py" "backend/scripts/start_complete_system.py"

# Mover archivos de modelos
Write-Host "`nMoviendo archivos de modelos..." -ForegroundColor Cyan
Move-FileIfExists "model_architecture.py" "models/training/scripts/model_architecture.py"
Move-FileIfExists "model_trainer.py" "models/training/scripts/model_trainer.py"
Move-FileIfExists "train_model.py" "models/training/scripts/train_model.py"
Move-FileIfExists "data_preparation.py" "models/training/scripts/data_preparation.py"
Move-FileIfExists "evaluate_model.py" "models/training/scripts/evaluate_model.py"
Move-FileIfExists "convert_to_tflite.py" "models/training/scripts/convert_to_tflite.py"

# Mover archivos de edge/Pi
Write-Host "`nMoviendo archivos de edge/Pi..." -ForegroundColor Cyan
Move-FileIfExists "pi_config.py" "edge/config/pi_config.py"
Move-FileIfExists "pi_simulator.py" "edge/src/pi_simulator.py"
Move-FileIfExists "start_pi_system.py" "edge/scripts/start_pi_system.py"
Move-FileIfExists "test_pi_system.py" "edge/tests/test_pi_system.py"
Move-FileIfExists "test_system.py" "edge/tests/test_system.py"

# Mover archivos de enrollment
Write-Host "`nMoviendo archivos de enrollment..." -ForegroundColor Cyan
Move-FileIfExists "enrollment.py" "enrollment/enrollment.py"
Move-FileIfExists "load_enrollments.py" "enrollment/load_enrollments.py"

# Mover carpeta de enrollments
if (Test-Path "enrollments") {
    Get-ChildItem -Path "enrollments" | Move-Item -Destination "enrollment/data/enrollments" -Force
    Remove-Item "enrollments" -Force -ErrorAction SilentlyContinue
    Write-Host "  ✓ Movido: enrollments/* → enrollment/data/enrollments/" -ForegroundColor Green
}

# Mover archivos de database
Write-Host "`nMoviendo archivos de database..." -ForegroundColor Cyan
Move-FileIfExists "gloria_stress_system.db" "database/gloria_stress_system.db"
Move-FileIfExists "init_database.py" "database/init_database.py"

# Mover archivos de reporting
Write-Host "`nMoviendo archivos de reporting..." -ForegroundColor Cyan
Move-FileIfExists "report_generator.py" "reporting/report_generator.py"

# Mover scripts
Write-Host "`nMoviendo scripts..." -ForegroundColor Cyan
Move-FileIfExists "quick_start.py" "scripts/testing/quick_start.py"

# Mover carpeta data si existe
if (Test-Path "data" -PathType Container) {
    Get-ChildItem -Path "data" -Exclude "raw","processed","embeddings","exports" | 
        Move-Item -Destination "data/raw" -Force -ErrorAction SilentlyContinue
    Write-Host "  ✓ Movido contenido de data/ a data/raw/" -ForegroundColor Green
}

Write-Host "`nCreando archivos .gitkeep en carpetas vacías..." -ForegroundColor Cyan
$gitkeepDirs = @(
    "edge/logs",
    "backend/logs",
    "logs/backend",
    "logs/edge",
    "logs/app",
    "data/raw",
    "data/processed",
    "data/embeddings",
    "data/exports",
    "models/trained",
    "models/evaluation",
    "reporting/outputs/daily",
    "reporting/outputs/weekly",
    "reporting/outputs/monthly",
    "tests/fixtures",
    "enrollment/data/enrollments"
)

foreach ($dir in $gitkeepDirs) {
    if (Test-Path $dir) {
        New-Item -Path "$dir/.gitkeep" -ItemType File -Force | Out-Null
    }
}

Write-Host "`nCreando archivos __init__.py..." -ForegroundColor Cyan
$initPyDirs = @(
    "edge/src",
    "edge/utils",
    "backend/app",
    "backend/app/models",
    "backend/app/api",
    "backend/app/api/v1",
    "backend/app/api/v1/endpoints",
    "backend/app/services",
    "backend/app/utils",
    "backend/tests",
    "enrollment/utils",
    "reporting/utils",
    "models/training/utils",
    "tests"
)

foreach ($dir in $initPyDirs) {
    if (Test-Path $dir) {
        New-Item -Path "$dir/__init__.py" -ItemType File -Force | Out-Null
    }
}

Write-Host "`nCreando archivo .env.example..." -ForegroundColor Cyan
$envExample = @"
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
"@

Set-Content -Path ".env.example" -Value $envExample

Write-Host "`nCreando README.md en carpetas principales..." -ForegroundColor Cyan

# docs/README.md
$docsReadme = @"
# Documentación del Proyecto StressVision

Esta carpeta contiene toda la documentación del proyecto organizada por categorías.

## Estructura

- ``fases/``: Documentación de cada fase del desarrollo
- ``guias/``: Guías de uso y quick starts
- ``arquitectura/``: Documentación técnica y diagramas
- ``implementacion/``: Detalles de implementación
- ``assets/``: Imágenes y diagramas
"@
Set-Content -Path "docs/README.md" -Value $docsReadme

# models/README.md
$modelsReadme = @"
# Modelos de Machine Learning

Esta carpeta contiene todo lo relacionado con los modelos de ML del proyecto.

## Estructura

- ``training/``: Scripts y configuraciones para entrenar modelos
- ``trained/``: Modelos entrenados listos para usar
- ``evaluation/``: Resultados de evaluación y métricas
"@
Set-Content -Path "models/README.md" -Value $modelsReadme

# edge/README.md
$edgeReadme = @"
# Edge Device (Raspberry Pi)

Código y configuración para dispositivos edge (Raspberry Pi).

## Estructura

- ``src/``: Código fuente principal
- ``config/``: Archivos de configuración
- ``tests/``: Tests específicos del edge
- ``scripts/``: Scripts de utilidad
"@
Set-Content -Path "edge/README.md" -Value $edgeReadme

# backend/README.md
$backendReadme = @"
# Backend API

Servidor backend y API REST del sistema.

## Estructura

- ``app/``: Aplicación principal
- ``tests/``: Tests del backend
- ``scripts/``: Scripts de utilidad
"@
Set-Content -Path "backend/README.md" -Value $backendReadme

# enrollment/README.md
$enrollmentReadme = @"
# Sistema de Enrollment

Sistema de registro de empleados y generación de embeddings faciales.

## Estructura

- ``data/``: Datos de enrollment (fotos y embeddings)
- ``scripts/``: Scripts de utilidad
- ``utils/``: Utilidades compartidas
"@
Set-Content -Path "enrollment/README.md" -Value $enrollmentReadme

# reporting/README.md
$reportingReadme = @"
# Sistema de Reportes

Generación de reportes automáticos y bajo demanda.

## Estructura

- ``templates/``: Plantillas HTML/PDF
- ``outputs/``: Reportes generados
- ``utils/``: Utilidades para generación de reportes
"@
Set-Content -Path "reporting/README.md" -Value $reportingReadme

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "  ✓ MIGRACIÓN COMPLETADA" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Próximos pasos:" -ForegroundColor Yellow
Write-Host "1. Revisar que todos los archivos se movieron correctamente"
Write-Host "2. Ejecutar: python check_imports.py"
Write-Host "3. Actualizar imports según GUIA_ACTUALIZACION_IMPORTS.md"
Write-Host "4. Actualizar .gitignore: Copy-Item .gitignore.new .gitignore"
Write-Host "5. Probar que el sistema funciona"
Write-Host "6. Commit de la nueva estructura"
Write-Host ""
Write-Host "Backup guardado en: $backupDir" -ForegroundColor Cyan
Write-Host ""
Write-Host "Para deshacer la migración:" -ForegroundColor Red
Write-Host "  Remove-Item docs,models,edge,backend,enrollment,reporting,scripts,tests,deployment -Recurse -Force"
Write-Host "  Copy-Item $backupDir\* . -Recurse -Force"
Write-Host ""


