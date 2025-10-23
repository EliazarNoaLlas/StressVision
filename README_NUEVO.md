# StressVision 👁️🧠

> Sistema de Detección de Estrés Laboral en Tiempo Real mediante Visión por Computadora e Inteligencia Artificial

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Active Development](https://img.shields.io/badge/status-active%20development-green.svg)]()

## 📋 Tabla de Contenidos

- [Descripción](#-descripción)
- [Características](#-características)
- [Arquitectura](#-arquitectura)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Instalación](#-instalación)
- [Uso Rápido](#-uso-rápido)
- [Documentación](#-documentación)
- [Tecnologías](#-tecnologías)
- [Privacidad y Ética](#-privacidad-y-ética)
- [Contribución](#-contribución)
- [Licencia](#-licencia)

## 🎯 Descripción

**StressVision** es un sistema integral de detección de estrés laboral que utiliza visión por computadora y deep learning para monitorear en tiempo real el estado emocional de empleados en entornos laborales. El sistema identifica automáticamente a empleados registrados y analiza sus expresiones faciales para detectar signos de estrés, generando alertas y reportes que permiten a la organización tomar acciones preventivas.

### Caso de Uso: GLORIA S.A.

Desarrollado inicialmente para GLORIA S.A., líder en la industria láctea peruana, este sistema busca:

- ✅ Reducir incidentes laborales relacionados con el estrés
- ✅ Mejorar el bienestar de los empleados
- ✅ Optimizar la productividad mediante intervención temprana
- ✅ Cumplir con normativas de seguridad laboral
- ✅ Generar datos accionables para RRHH

## ✨ Características

### Core Features

- **🎭 Detección de Emociones**: Reconocimiento de 7 emociones básicas (felicidad, tristeza, enojo, miedo, sorpresa, disgusto, neutral)
- **👤 Reconocimiento Facial**: Identificación automática de empleados registrados mediante embeddings faciales
- **⚡ Procesamiento en Tiempo Real**: Inferencia optimizada en Raspberry Pi (< 100ms por frame)
- **📊 Análisis de Estrés**: Algoritmo propietario que calcula nivel de estrés basado en patrón emocional temporal
- **🔔 Sistema de Alertas**: Notificaciones automáticas cuando se detectan niveles críticos de estrés
- **📈 Reportes Automáticos**: Generación de reportes diarios, semanales y mensuales
- **🔐 Privacy by Design**: Procesamiento local de imágenes, almacenamiento encriptado de datos biométricos

### Características Técnicas

- **Edge Computing**: Procesamiento en dispositivo (Raspberry Pi) para reducir latencia y proteger privacidad
- **Arquitectura Modular**: Componentes desacoplados (backend, edge, enrollment, reporting)
- **API REST**: Endpoints documentados para integración con sistemas existentes
- **WebSocket**: Comunicación bidireccional en tiempo real
- **Modelo Optimizado**: Modelos TensorFlow Lite cuantizados para inferencia eficiente
- **Base de Datos Local**: SQLite con opción de migración a PostgreSQL
- **Escalable**: Diseño que soporta múltiples dispositivos edge y empleados

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                    STRESSVISION - ARQUITECTURA                  │
└─────────────────────────────────────────────────────────────────┘

   📷 Cámara IP / USB                    💻 Servidor Central
        │                                        │
        ▼                                        ▼
   ┌──────────────┐                     ┌──────────────┐
   │ Raspberry Pi │ ◄──WebSocket──────► │   Backend    │
   │   (Edge)     │                     │   (Flask)    │
   │              │                     │              │
   │ • Captura    │                     │ • API REST   │
   │ • Detección  │                     │ • WebSocket  │
   │ • Inferencia │                     │ • Alertas    │
   └──────────────┘                     └──────┬───────┘
                                               │
                                         ┌─────┴─────┐
                                         ▼           ▼
                                   ┌──────────┐ ┌──────────┐
                                   │ Database │ │ Reports  │
                                   │ (SQLite) │ │  (PDF)   │
                                   └──────────┘ └──────────┘
```

Ver [DIAGRAMA_ESTRUCTURA.md](DIAGRAMA_ESTRUCTURA.md) para detalles completos.

## 📁 Estructura del Proyecto

```
StressVision/
├── backend/          # 🖥️ Servidor API y lógica de negocio
├── edge/             # 🔌 Código para Raspberry Pi
├── models/           # 🧠 Entrenamiento y modelos ML
├── enrollment/       # 👤 Sistema de registro de empleados
├── reporting/        # 📊 Generación de reportes
├── database/         # 🗄️ Esquemas y migraciones
├── docs/             # 📚 Documentación completa
├── scripts/          # 🛠️ Scripts de utilidad
├── tests/            # 🧪 Tests unitarios e integración
└── data/             # 💾 Datos (no versionado)
```

Ver [ESTRUCTURA_PROPUESTA.md](ESTRUCTURA_PROPUESTA.md) para detalles completos.

## 🚀 Instalación

### Requisitos Previos

- Python 3.8 o superior
- pip y virtualenv
- Git
- (Opcional) Raspberry Pi 4 con cámara

### Instalación Rápida

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/StressVision.git
cd StressVision

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
.\venv\Scripts\activate   # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
cp .env.example .env
# Editar .env con tus valores

# 5. Inicializar base de datos
python database/init_database.py

# 6. (Opcional) Descargar modelos pre-entrenados
# Los modelos se descargarán automáticamente en la primera ejecución
```

### Instalación por Componente

#### Backend

```bash
cd backend
pip install -r requirements.txt
python main.py
```

#### Edge (Raspberry Pi)

```bash
cd edge
pip install -r requirements.txt
# Configurar en edge/config/pi_config.py
python scripts/start_pi_system.py
```

#### Training

```bash
cd models/training
pip install -r requirements.txt
python scripts/train_model.py
```

## 🎮 Uso Rápido

### Quick Start (Sistema Completo)

```bash
# Iniciar sistema completo en modo simulación
python scripts/testing/quick_start.py
```

### Paso a Paso

#### 1. Registrar Empleado

```bash
cd enrollment
python enrollment.py \
  --employee-id EMP001 \
  --name "Juan Pérez" \
  --department "Producción"
```

Esto iniciará la captura de 10 fotos del empleado y generará su embedding facial.

#### 2. Iniciar Backend

```bash
cd backend
python main.py
```

El servidor estará disponible en `http://localhost:8000`

#### 3. Iniciar Edge Device

```bash
cd edge
python scripts/start_pi_system.py
```

O en modo simulación:

```bash
python edge/src/pi_simulator.py
```

#### 4. Generar Reporte

```bash
cd reporting
python report_generator.py \
  --type weekly \
  --employee EMP001 \
  --output outputs/weekly/
```

### API REST

El backend expone los siguientes endpoints:

```http
GET    /api/v1/employees              # Listar empleados
POST   /api/v1/employees              # Registrar empleado
GET    /api/v1/detections             # Obtener detecciones
GET    /api/v1/alerts                 # Obtener alertas
POST   /api/v1/reports                # Generar reporte
GET    /health                        # Health check
```

Ver documentación completa en `/docs` cuando el servidor esté corriendo.

## 📚 Documentación

### Documentación General

- [📖 Guía Completa del Proyecto](docs/PROYECTO_COMPLETO_FINAL.md)
- [🏗️ Arquitectura del Sistema](docs/arquitectura/DIAGRAMA_FLUJO.md)
- [📊 Estado del Proyecto](docs/arquitectura/ESTADO_PROYECTO_COMPLETO.md)
- [🚀 Quick Start](docs/guias/COMANDOS_RAPIDOS.md)

### Documentación por Fase

- [Fase 4: Preparación de Datos y Entrenamiento](docs/fases/FASE4_DOCUMENTACION.md)
- [Fase 5: Evaluación y Optimización](docs/fases/FASE5_DOCUMENTACION.md)
- [Fase 6: Deployment y Validación](docs/fases/FASE6_DOCUMENTACION.md)

### Guías Específicas

- [👤 Instrucciones de Enrollment](docs/guias/INSTRUCCIONES_ENROLLMENT.md)
- [🔧 Comandos Rápidos](docs/guias/COMANDOS_RAPIDOS.md)
- [📦 Actualización de Imports](GUIA_ACTUALIZACION_IMPORTS.md)

### Componentes

- [🖥️ Backend API](backend/README.md)
- [🔌 Edge Device](edge/README.md)
- [🧠 Modelos ML](models/README.md)
- [👤 Enrollment](enrollment/README.md)
- [📊 Reporting](reporting/README.md)

## 🛠️ Tecnologías

### Machine Learning & Computer Vision

- **TensorFlow / Keras**: Entrenamiento de modelos
- **TensorFlow Lite**: Inferencia optimizada
- **OpenCV**: Procesamiento de imágenes
- **dlib / face_recognition**: Reconocimiento facial
- **NumPy / Pandas**: Manipulación de datos

### Backend

- **Flask / FastAPI**: Framework web
- **SQLAlchemy**: ORM
- **Socket.IO**: WebSocket
- **SQLite / PostgreSQL**: Base de datos
- **Celery**: Tareas asíncronas (futuro)

### Edge

- **Raspberry Pi OS**: Sistema operativo
- **picamera**: Control de cámara
- **RPi.GPIO**: Control de hardware

### DevOps & Tools

- **Docker**: Containerización
- **pytest**: Testing
- **Git**: Control de versiones
- **GitHub Actions**: CI/CD (futuro)

## 🔐 Privacidad y Ética

StressVision está diseñado con la privacidad como prioridad:

### Principios

1. **Consentimiento Informado**: Los empleados deben firmar consentimiento antes del enrollment
2. **Transparencia**: Los empleados saben cuándo y dónde están siendo monitoreados
3. **Minimización de Datos**: Solo se almacenan embeddings, no imágenes crudas
4. **Acceso Restringido**: Solo personal autorizado de RRHH puede acceder a datos individuales
5. **Derecho al Olvido**: Los empleados pueden solicitar la eliminación de sus datos
6. **Anonimización en Reportes**: Los reportes agregados no identifican individuos
7. **Almacenamiento Local**: Los datos permanecen en servidores de la empresa

### Cumplimiento Legal

- ✅ Ley de Protección de Datos Personales (Perú - Ley N° 29733)
- ✅ Directiva de Seguridad de la Información (Perú)
- 🔄 GDPR (para operaciones internacionales) - En proceso

### Datos Recolectados

| Dato                  | Propósito                       | Retención    |
|-----------------------|---------------------------------|--------------|
| Embedding facial      | Identificación                  | Hasta salida |
| Emociones detectadas  | Análisis de estrés              | 6 meses      |
| Timestamps            | Correlación temporal            | 6 meses      |
| Nivel de estrés       | Alertas y reportes              | 6 meses      |

**No se almacenan**: Imágenes/videos crudos, audio, ubicación precisa.

## 🧪 Testing

```bash
# Ejecutar todos los tests
python -m pytest tests/

# Tests con coverage
python -m pytest --cov=backend --cov=edge tests/

# Tests específicos
python -m pytest tests/integration/test_end_to_end.py

# Tests unitarios
python -m pytest tests/unit/
```

## 📊 Métricas del Sistema

### Performance

- **Latencia de inferencia**: < 100ms por frame (Raspberry Pi 4)
- **Throughput**: 10-15 FPS
- **Precisión de emociones**: ~85% (validación cruzada)
- **Precisión de reconocimiento**: ~95% (con condiciones óptimas)

### Escalabilidad

- ✅ Soporta hasta 50 empleados por dispositivo edge
- ✅ Soporta múltiples dispositivos edge por backend
- ✅ Base de datos puede escalar a PostgreSQL

## 🗺️ Roadmap

### Versión 1.0 (Actual)

- [x] Detección de emociones en tiempo real
- [x] Reconocimiento facial
- [x] Sistema de alertas básico
- [x] Reportes en PDF
- [x] Deployment en Raspberry Pi

### Versión 2.0 (Planificado)

- [ ] Dashboard web en React
- [ ] Múltiples cámaras por ubicación
- [ ] Análisis predictivo con LSTM
- [ ] Integración con HR systems
- [ ] App móvil para alertas
- [ ] Soporte multi-idioma

### Versión 3.0 (Futuro)

- [ ] Detección de fatiga por análisis de ojos
- [ ] Reconocimiento de gestos corporales
- [ ] IA explicable (XAI) para transparencia
- [ ] Edge AI distribuido
- [ ] Federated Learning para privacidad

## 🤝 Contribución

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

### Guidelines

- Sigue PEP 8 para código Python
- Agrega tests para nuevas funcionalidades
- Actualiza la documentación
- Asegúrate de que todos los tests pasen

## 👥 Equipo

- **Arquitecto de Sistema**: [Tu Nombre]
- **ML Engineer**: [Tu Nombre]
- **Edge Developer**: [Tu Nombre]
- **DevOps**: [Tu Nombre]

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## 📧 Contacto

- **Email**: contacto@stressvision.com
- **GitHub**: https://github.com/tu-usuario/StressVision
- **Documentación**: https://stressvision.readthedocs.io

## 🙏 Agradecimientos

- GLORIA S.A. por el caso de uso y apoyo
- Comunidad de TensorFlow
- Datasets públicos: FER2013, AffectNet
- Librerías open source utilizadas

## 📊 Estado del Proyecto

```
Fase 1: Diseño                    ✅ Completado
Fase 2: Prototipo                 ✅ Completado
Fase 3: Arquitectura              ✅ Completado
Fase 4: Entrenamiento             ✅ Completado
Fase 5: Evaluación                ✅ Completado
Fase 6: Deployment                ✅ Completado
Fase 7: Dashboard                 🔄 En progreso
Fase 8: Producción                ⏳ Pendiente
```

---

**Hecho con ❤️ para un mejor ambiente laboral**



