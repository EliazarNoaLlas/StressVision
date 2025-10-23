# 📋 Guía de Instalación y Enrollment - Stress Vision

## 🚀 Fase 2 y 3: Base de Datos y Enrollment de Personal

Esta guía te llevará paso a paso para configurar la base de datos SQLite y realizar el enrollment de 20 personas.

---

## 📦 Paso 1: Instalación de Dependencias

### 1.1 Asegúrate de tener Python 3.8+ instalado

```bash
python --version
```

### 1.2 Instalar las dependencias necesarias

```bash
# Activar entorno virtual (recomendado)
python -m venv venv

# Windows
venv\Scripts\activate

# Instalar todas las dependencias
pip install -r requirements.txt
```

**Nota:** La instalación puede tomar varios minutos debido a PyTorch y TensorFlow.

---

## 🗄️ Paso 2: Crear la Base de Datos SQLite

### 2.1 Ejecutar el script de inicialización

```bash
python init_database.py
```

Este script:
- ✅ Crea la base de datos `gloria_stress_system.db`
- ✅ Genera todas las tablas necesarias (employees, sessions, detection_events, etc.)
- ✅ Crea índices para optimizar consultas
- ✅ Verifica la integridad de la base de datos

### 2.2 Verificación

Cuando termine, verás algo como:

```
✅ Base de datos creada exitosamente

📊 Tablas creadas:
  1. employees                    (0 registros)
  2. sessions                     (0 registros)
  3. detection_events             (0 registros)
  4. employee_stress_summary      (0 registros)
  5. reports_15min                (0 registros)
  6. alerts                       (0 registros)
  7. audit_log                    (0 registros)
  8. notification_config          (1 registro)

✅ Verificación de integridad: OK
```

---

## 👤 Paso 3: Enrollment de Empleados

### 3.1 Preparación

**Requisitos previos:**
- ✅ Cámara web conectada y funcionando
- ✅ Buena iluminación en el espacio
- ✅ Consentimiento informado de los empleados
- ✅ Lista de códigos de empleado asignados

### 3.2 Ejecutar el script de enrollment

```bash
python enrollment.py
```

### 3.3 Opciones disponibles

El sistema te mostrará:

```
OPCIONES DE ENROLLMENT
======================
1. Enrollment individual
2. Enrollment batch (20 personas)
3. Salir
```

#### Opción 1: Enrollment Individual

Ideal para hacer pruebas o agregar empleados uno por uno.

1. Selecciona opción `1`
2. Ingresa los datos:
   - **Código**: `EMP001`
   - **Nombre**: `Juan Pérez García`
   - **Departamento**: `Producción`
   - **Turno**: `morning` (o `afternoon`, `night`)

3. Sigue las instrucciones en pantalla:
   - Mira directamente a la cámara
   - Mantén expresión neutral
   - Presiona **ESPACIO** para capturar cada foto
   - El sistema capturará 10 fotos automáticamente
   - Varía ligeramente la pose entre capturas

4. Al finalizar verás:
   ```
   ✅ ENROLLMENT COMPLETADO
   📊 Muestras capturadas: 10
   🎯 Calidad del embedding: 0.85/1.0
   ✅ Excelente calidad. Reconocimiento óptimo garantizado.
   ```

#### Opción 2: Enrollment Batch (20 Personas)

Para procesar los 20 empleados del piloto de manera secuencial.

1. Selecciona opción `2`
2. El sistema mostrará la lista de 20 empleados predefinidos:
   - EMP001: Juan Pérez García (Producción - Mañana)
   - EMP002: María González López (Producción - Mañana)
   - ... (18 más)

3. Confirma con `s` para comenzar

4. Para cada empleado:
   - Aparecerá su nombre en pantalla
   - Realiza el proceso de captura (10 fotos)
   - Presiona ENTER para continuar con el siguiente

**⏱️ Tiempo estimado:** 15-20 minutos por persona = **5-7 horas total**

### 3.4 Archivos generados

Después del enrollment, se crean los siguientes archivos en la carpeta `enrollments/`:

```
enrollments/
├── EMP001_embedding.json       # Datos del embedding
├── EMP001_sample_1.jpg         # Muestra facial 1
├── EMP001_sample_2.jpg         # Muestra facial 2
├── ...
├── EMP001_sample_10.jpg        # Muestra facial 10
└── ... (repetir para cada empleado)
```

### 3.5 Interpretación de la Calidad

El sistema calcula un **score de calidad** (0-1) basado en la consistencia entre las muestras:

- **< 0.60**: ⚠️ Calidad baja - Se recomienda repetir enrollment
- **0.60-0.75**: ⚠️ Calidad aceptable - Funcionalidad garantizada
- **> 0.75**: ✅ Excelente calidad - Reconocimiento óptimo

---

## 📥 Paso 4: Cargar Enrollments a la Base de Datos

### 4.1 Ejecutar el script de carga

```bash
python load_enrollments.py
```

### 4.2 Opciones disponibles

```
OPCIONES:
1. Cargar todos los enrollments desde directorio
2. Cargar enrollment individual
3. Listar empleados registrados
4. Verificar embeddings
5. Salir
```

#### Opción 1: Cargar todos (Recomendado)

1. Selecciona opción `1`
2. Confirma el directorio (por defecto: `enrollments`)
3. El sistema procesará todos los archivos JSON

Verás algo como:

```
📥 CARGANDO ENROLLMENTS A BASE DE DATOS
📁 Directorio: enrollments
📊 Archivos encontrados: 20
🗄️  Base de datos: gloria_stress_system.db

[1/20] Procesando: Juan Pérez García (EMP001)
  ✓ Insertado (ID: 1)
    • Calidad: 0.85
    • Muestras: 10
    • Departamento: Producción

[2/20] Procesando: María González López (EMP002)
  ✓ Insertado (ID: 2)
...

📊 RESUMEN DE CARGA
✅ Nuevos empleados: 20
🔄 Empleados actualizados: 0
❌ Fallos: 0
📈 Total procesados: 20
👥 Total empleados activos en BD: 20
```

#### Opción 3: Listar empleados

Para verificar que todos se cargaron correctamente:

```
👥 EMPLEADOS REGISTRADOS (20 total)
================================================================================
ID    Código     Nombre                         Departamento         Calidad
--------------------------------------------------------------------------------
1     EMP001     Juan Pérez García              Producción           0.85
2     EMP002     María González López           Producción           0.82
...
```

#### Opción 4: Verificar embeddings

Verifica que todos los embeddings tengan el formato correcto (512 dimensiones):

```
🔍 VERIFICACIÓN DE EMBEDDINGS
✅ EMP001 - Juan Pérez García: OK (dim: 512, calidad: 0.85)
✅ EMP002 - María González López: OK (dim: 512, calidad: 0.82)
...

📊 RESULTADO DE VERIFICACIÓN
✅ Válidos: 20
❌ Inválidos: 0
📈 Total: 20
```

---

## ✅ Paso 5: Verificación Final

### 5.1 Verificar la base de datos

Puedes usar cualquier cliente SQLite para inspeccionar la base de datos:

```bash
# Usando SQLite CLI
sqlite3 gloria_stress_system.db

# Consultas útiles:
SELECT COUNT(*) FROM employees;
SELECT employee_code, full_name, face_encoding_quality FROM employees;
```

O usar herramientas gráficas:
- **DB Browser for SQLite** (https://sqlitebrowser.org/)
- **DBeaver** (https://dbeaver.io/)

### 5.2 Estructura esperada

```sql
-- Deberías tener 20 empleados registrados:
SELECT 
    employee_code,
    full_name,
    department,
    shift,
    face_encoding_quality,
    is_active
FROM employees
WHERE is_active = 1
ORDER BY employee_code;
```

---

## 🎯 Paso 6: Próximos Pasos

### 6.1 Iniciar el sistema de monitoreo

```bash
streamlit run main.py
```

### 6.2 Integrar reconocimiento facial

El siguiente paso sería:
- Modificar `main.py` para usar los embeddings de la BD
- Implementar identificación en tiempo real
- Asociar detecciones con empleados

---

## 🔧 Troubleshooting

### Problema: Cámara no detectada

**Solución:**
```bash
# Verificar cámaras disponibles
python -c "import cv2; print([i for i in range(4) if cv2.VideoCapture(i).isOpened()])"

# Windows: Configuración → Privacidad → Cámara
# Dar permisos a Python
```

### Problema: Error al cargar modelos de PyTorch

**Solución:**
```bash
# Reinstalar PyTorch
pip uninstall torch torchvision facenet-pytorch
pip install torch torchvision facenet-pytorch
```

### Problema: Calidad de embedding muy baja

**Causas comunes:**
- ❌ Mala iluminación
- ❌ Persona se movió mucho entre capturas
- ❌ Rostro parcialmente oculto (mascarilla, lentes)
- ❌ Cámara de baja calidad

**Solución:**
- ✅ Mejorar iluminación
- ✅ Pedir a la persona que permanezca quieta
- ✅ Repetir el enrollment

### Problema: Base de datos bloqueada

**Solución:**
```bash
# Windows PowerShell
Get-Process | Where-Object {$_.ProcessName -like "*python*"} | Stop-Process

# Linux/Mac
pkill -f python
```

---

## 📊 Estructura de Archivos Final

Después de completar todos los pasos:

```
StressVision/
├── gloria_stress_system.db          # ✅ Base de datos SQLite
├── enrollments/                     # ✅ Directorio de enrollments
│   ├── EMP001_embedding.json
│   ├── EMP001_sample_1.jpg
│   ├── ...
│   └── EMP020_embedding.json
├── init_database.py                 # ✅ Script de inicialización
├── enrollment.py                    # ✅ Script de enrollment
├── load_enrollments.py              # ✅ Script de carga
├── main.py                          # Aplicación principal
├── requirements.txt                 # ✅ Dependencias actualizadas
└── INSTRUCCIONES_ENROLLMENT.md      # Este archivo
```

---

## 📞 Soporte

Si encuentras problemas:

1. Revisa la sección de **Troubleshooting** arriba
2. Verifica los logs en consola
3. Asegúrate de tener todas las dependencias instaladas
4. Verifica que la cámara funcione con otras aplicaciones

---

## 📝 Notas Importantes

### Privacidad y GDPR

- ✅ Obtener consentimiento informado de todos los empleados
- ✅ Explicar cómo se usarán los datos
- ✅ Dar opción de no participar
- ✅ Permitir eliminación de datos a petición

### Seguridad

- 🔒 La base de datos contiene datos biométricos sensibles
- 🔒 Mantener el archivo `.db` seguro
- 🔒 No compartir embeddings faciales
- 🔒 Implementar control de acceso en producción

### Calidad de Datos

- 📊 Calidad promedio recomendada: > 0.75
- 📊 Mínimo de muestras por persona: 5 (recomendado: 10)
- 📊 Revisión periódica de embeddings

---

## ✅ Checklist de Implementación

### Fase 2: Base de Datos ✅
- [x] Dependencias instaladas
- [x] Base de datos SQLite creada
- [x] Tablas y esquema verificados
- [x] Índices creados

### Fase 3: Enrollment ✅
- [ ] Consentimientos firmados (20/20)
- [ ] Sesión informativa realizada
- [ ] Códigos de empleado asignados
- [ ] Enrollments completados (0/20)
- [ ] Embeddings cargados a BD (0/20)
- [ ] Verificación final exitosa

---

**¡Éxito en la implementación! 🚀**

Gloria S.A. - Sistema de Detección de Estrés Laboral




