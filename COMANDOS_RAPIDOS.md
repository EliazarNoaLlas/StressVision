# ⚡ Comandos Rápidos - Stress Vision

Guía de referencia rápida con todos los comandos necesarios.

---

## 🚀 Inicio Rápido (Opción Recomendada)

```bash
# 1. Instalar todo y configurar automáticamente
python quick_start.py
```

---

## 📦 Instalación Manual

### Paso 1: Crear entorno virtual (Recomendado)

#### Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux/Mac:
```bash
python3 -m venv venv
source venv/bin/activate
```

### Paso 2: Instalar dependencias

```bash
pip install -r requirements.txt
```

**Tiempo estimado:** 5-10 minutos

---

## 🗄️ Base de Datos

### Crear base de datos SQLite
```bash
python init_database.py
```

### Ver esquema de tablas
```bash
python init_database.py
# Cuando pregunte, seleccionar "s" para ver esquema
```

### Verificar base de datos con SQLite CLI
```bash
sqlite3 gloria_stress_system.db

# Comandos útiles dentro de SQLite:
.tables                  # Listar tablas
.schema employees        # Ver esquema de tabla
SELECT COUNT(*) FROM employees;    # Contar empleados
.exit                    # Salir
```

---

## 👤 Enrollment de Empleados

### Enrollment individual (Prueba)
```bash
python enrollment.py
# Seleccionar opción: 1
# Ingresar datos del empleado
```

### Enrollment batch (20 personas)
```bash
python enrollment.py
# Seleccionar opción: 2
# Confirmar con "s"
```

**Tiempo estimado:** 15-20 minutos por persona

---

## 📥 Cargar Enrollments a Base de Datos

### Cargar todos los enrollments
```bash
python load_enrollments.py
# Seleccionar opción: 1
# Confirmar directorio (Enter para default: enrollments)
```

### Cargar enrollment individual
```bash
python load_enrollments.py
# Seleccionar opción: 2
# Ingresar ruta: enrollments/EMP001_embedding.json
```

### Listar empleados registrados
```bash
python load_enrollments.py
# Seleccionar opción: 3
```

### Verificar embeddings
```bash
python load_enrollments.py
# Seleccionar opción: 4
```

---

## 🧪 Pruebas y Verificación

### Probar todo el sistema
```bash
python test_system.py
```

Esto ejecuta 8 pruebas:
1. ✅ Versión de Python
2. ✅ Librerías instaladas
3. ✅ Acceso a cámara
4. ✅ Modelos de ML
5. ✅ Base de datos
6. ✅ Enrollments
7. ✅ Detección facial
8. ✅ Streamlit

### Probar cámara específicamente
```bash
python -c "import cv2; cap = cv2.VideoCapture(0); print('✅ Cámara OK' if cap.isOpened() else '❌ Error'); cap.release()"
```

### Verificar que PyTorch detecta GPU (si tienes)
```bash
python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}'); print(f'Device: {torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")}')"
```

### Ver versiones de librerías
```bash
pip list | grep -E "torch|opencv|streamlit|facenet"
```

---

## 🎯 Ejecutar Aplicación Principal

### Iniciar Streamlit
```bash
streamlit run main.py
```

Se abrirá automáticamente en: `http://localhost:8501`

### Iniciar en puerto específico
```bash
streamlit run main.py --server.port 8080
```

### Iniciar sin abrir navegador automáticamente
```bash
streamlit run main.py --server.headless true
```

---

## 🔍 Consultas SQL Útiles

### Ver todos los empleados
```sql
sqlite3 gloria_stress_system.db "SELECT employee_code, full_name, department, face_encoding_quality FROM employees WHERE is_active = 1;"
```

### Contar empleados por departamento
```sql
sqlite3 gloria_stress_system.db "SELECT department, COUNT(*) as total FROM employees WHERE is_active = 1 GROUP BY department;"
```

### Ver empleados con calidad baja (< 0.7)
```sql
sqlite3 gloria_stress_system.db "SELECT employee_code, full_name, face_encoding_quality FROM employees WHERE face_encoding_quality < 0.7 ORDER BY face_encoding_quality;"
```

### Ver últimas detecciones
```sql
sqlite3 gloria_stress_system.db "SELECT * FROM detection_events ORDER BY timestamp DESC LIMIT 10;"
```

### Exportar empleados a CSV
```sql
sqlite3 -header -csv gloria_stress_system.db "SELECT employee_code, full_name, department, shift, face_encoding_quality FROM employees WHERE is_active = 1;" > empleados.csv
```

---

## 🛠️ Mantenimiento

### Backup de base de datos
```bash
# Windows
copy gloria_stress_system.db gloria_stress_system_backup_%date:~-4,4%%date:~-10,2%%date:~-7,2%.db

# Linux/Mac
cp gloria_stress_system.db gloria_stress_system_backup_$(date +%Y%m%d).db
```

### Vacuumar base de datos (optimizar)
```bash
sqlite3 gloria_stress_system.db "VACUUM;"
```

### Ver tamaño de base de datos
```bash
# Windows
dir gloria_stress_system.db

# Linux/Mac
ls -lh gloria_stress_system.db
```

### Limpiar archivos temporales
```bash
# Windows
del /S *.pyc
rmdir /S /Q __pycache__

# Linux/Mac
find . -type f -name "*.pyc" -delete
find . -type d -name "__pycache__" -exec rm -r {} +
```

---

## 🧹 Limpieza y Reset

### Eliminar base de datos (¡CUIDADO!)
```bash
# Windows
del gloria_stress_system.db

# Linux/Mac
rm gloria_stress_system.db
```

### Eliminar enrollments (¡CUIDADO!)
```bash
# Windows
rmdir /S /Q enrollments

# Linux/Mac
rm -rf enrollments
```

### Desinstalar todas las dependencias
```bash
pip freeze > installed.txt
pip uninstall -r installed.txt -y
```

---

## 🔧 Troubleshooting

### Reinstalar PyTorch (si hay problemas)
```bash
pip uninstall torch torchvision facenet-pytorch
pip install torch torchvision facenet-pytorch
```

### Reinstalar OpenCV
```bash
pip uninstall opencv-python opencv-python-headless
pip install opencv-python
```

### Limpiar cache de pip
```bash
pip cache purge
```

### Ver procesos de Python activos
```bash
# Windows
tasklist | findstr python

# Linux/Mac
ps aux | grep python
```

### Matar todos los procesos de Python (¡CUIDADO!)
```bash
# Windows
taskkill /F /IM python.exe

# Linux/Mac
pkill -f python
```

### Ver logs de Streamlit
```bash
# Los logs aparecen en la terminal donde ejecutaste streamlit run
# También en: ~/.streamlit/logs/ (Linux/Mac)
```

---

## 📊 Comandos de Información

### Ver información del sistema
```bash
python -c "import platform; import sys; print(f'OS: {platform.system()} {platform.release()}'); print(f'Python: {sys.version}'); print(f'Architecture: {platform.machine()}')"
```

### Ver uso de memoria
```bash
# Windows
tasklist /FI "IMAGENAME eq python.exe"

# Linux/Mac
ps aux | grep python | awk '{print $2, $4, $11}'
```

### Ver espacio en disco
```bash
# Windows
dir

# Linux/Mac
df -h .
```

---

## 🎓 Comandos de Desarrollo

### Generar requirements.txt actualizado
```bash
pip freeze > requirements.txt
```

### Instalar en modo editable (desarrollo)
```bash
pip install -e .
```

### Ejecutar linter
```bash
pip install flake8
flake8 *.py
```

### Formatear código
```bash
pip install black
black *.py
```

---

## 📚 Ayuda y Documentación

### Ver ayuda de enrollment
```bash
python enrollment.py --help
```

### Ver ayuda de Streamlit
```bash
streamlit --help
```

### Abrir documentación de Streamlit
```bash
streamlit docs
```

---

## 🔗 URLs Importantes

| Servicio | URL |
|----------|-----|
| Aplicación Streamlit | http://localhost:8501 |
| Documentación Streamlit | https://docs.streamlit.io |
| FaceNet PyTorch | https://github.com/timesler/facenet-pytorch |
| DeepFace | https://github.com/serengil/deepface |
| SQLite Browser | https://sqlitebrowser.org/ |

---

## 📝 Archivos de Configuración

### Crear archivo .env (opcional)
```bash
# Windows
echo DATABASE_NAME=gloria_stress_system.db > .env
echo STRESS_THRESHOLD=40 >> .env

# Linux/Mac
cat > .env << EOF
DATABASE_NAME=gloria_stress_system.db
STRESS_THRESHOLD=40
EOF
```

### Configuración de Streamlit
```bash
# Crear carpeta de configuración
mkdir .streamlit

# Crear archivo de configuración
cat > .streamlit/config.toml << EOF
[theme]
primaryColor="#667eea"
backgroundColor="#ffffff"
secondaryBackgroundColor="#f0f2f6"
textColor="#262730"
font="sans serif"

[server]
port=8501
headless=false
EOF
```

---

## 🎯 Workflows Comunes

### Workflow 1: Primera vez (Setup completo)
```bash
python -m venv venv
venv\Scripts\activate              # Windows
pip install -r requirements.txt
python init_database.py
python enrollment.py               # Opción 2 (batch)
python load_enrollments.py         # Opción 1 (cargar todos)
python test_system.py              # Verificar
streamlit run main.py              # Iniciar
```

### Workflow 2: Agregar nuevo empleado
```bash
python enrollment.py               # Opción 1 (individual)
python load_enrollments.py         # Opción 2 (individual)
```

### Workflow 3: Verificación diaria
```bash
python test_system.py
python load_enrollments.py         # Opción 3 (listar)
streamlit run main.py
```

### Workflow 4: Backup antes de cambios importantes
```bash
cp gloria_stress_system.db gloria_stress_system_backup.db
tar -czf enrollments_backup.tar.gz enrollments/
# Hacer cambios
# Si algo sale mal: restaurar backups
```

---

## 🆘 Comandos de Emergencia

### Sistema no responde
```bash
# Ctrl+C en la terminal
# Si no funciona:
taskkill /F /IM python.exe          # Windows
pkill -9 python                     # Linux/Mac
```

### Base de datos corrupta
```bash
sqlite3 gloria_stress_system.db "PRAGMA integrity_check;"
# Si falla: restaurar desde backup
```

### Cámara no funciona
```bash
# Verificar dispositivos
python -c "import cv2; print([i for i in range(5) if cv2.VideoCapture(i).isOpened()])"

# Windows: Cerrar apps que usan cámara
# Configuración → Privacidad → Cámara → Permitir
```

---

## 📖 Lectura Recomendada

- `INSTRUCCIONES_ENROLLMENT.md` - Guía completa paso a paso
- `RESUMEN_IMPLEMENTACION.md` - Resumen de implementación
- `DIAGRAMA_FLUJO.md` - Diagramas del sistema
- `README.md` - Documentación general

---

## ✅ Checklist de Comandos Ejecutados

Usa esta lista para verificar que has completado todo:

- [ ] `pip install -r requirements.txt`
- [ ] `python init_database.py`
- [ ] `python enrollment.py` (20 personas)
- [ ] `python load_enrollments.py` (cargar todos)
- [ ] `python load_enrollments.py` (verificar)
- [ ] `python test_system.py`
- [ ] `streamlit run main.py`

---

**Gloria S.A. - Stress Vision**
Sistema de Detección de Estrés Laboral v2.0




