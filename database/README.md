# Base de Datos - StressVision

## Descripción

Este módulo contiene todo lo relacionado con la base de datos del sistema StressVision, incluyendo esquemas, migraciones, scripts de mantenimiento y herramientas de administración.

## Estructura

```
database/
├── README.md                    # Este archivo
├── schema.sql                   # Esquema completo de la base de datos
├── init_database.py             # Script de inicialización
├── gloria_stress_system.db      # Base de datos SQLite (no versionado)
│
├── migrations/                  # Migraciones de esquema
│   └── 001_initial_schema.sql  # Migración inicial
│
└── scripts/                     # Scripts de mantenimiento
    ├── backup.sh               # Script de backup automático
    └── cleanup.sql             # Limpieza de datos antiguos
```

## Base de Datos

### Información General

- **Motor**: SQLite 3
- **Nombre**: gloria_stress_system.db
- **Ubicación**: `database/gloria_stress_system.db`
- **Tamaño típico**: 10-100 MB (según datos históricos)
- **Codificación**: UTF-8

### Tablas Principales

#### 1. `employees` - Empleados Registrados

Almacena información de empleados registrados en el sistema.

| Campo          | Tipo        | Descripción                          |
|----------------|-------------|--------------------------------------|
| employee_id    | TEXT        | ID único del empleado (PK)           |
| name           | TEXT        | Nombre completo                      |
| department     | TEXT        | Departamento/área                    |
| position       | TEXT        | Cargo/posición                       |
| enrollment_date| TIMESTAMP   | Fecha de registro                    |
| face_embedding | TEXT (JSON) | Embedding facial (512 dimensiones)   |
| is_active      | BOOLEAN     | Estado activo/inactivo               |
| consent_signed | BOOLEAN     | Consentimiento firmado               |
| consent_date   | TIMESTAMP   | Fecha de firma de consentimiento     |

#### 2. `detection_sessions` - Sesiones de Detección

Representa una sesión de monitoreo continuo.

| Campo              | Tipo      | Descripción                          |
|--------------------|-----------|--------------------------------------|
| session_id         | INTEGER   | ID único de sesión (PK, autoincrement)|
| device_id          | TEXT      | ID del dispositivo edge              |
| location           | TEXT      | Ubicación física del dispositivo     |
| start_time         | TIMESTAMP | Inicio de la sesión                  |
| end_time           | TIMESTAMP | Fin de la sesión                     |
| frames_processed   | INTEGER   | Total de frames procesados           |
| detections_count   | INTEGER   | Total de detecciones                 |

#### 3. `emotion_detections` - Detecciones de Emociones

Almacena cada detección individual de emoción.

| Campo              | Tipo      | Descripción                          |
|--------------------|-----------|--------------------------------------|
| detection_id       | INTEGER   | ID único de detección (PK)           |
| session_id         | INTEGER   | ID de sesión (FK)                    |
| employee_id        | TEXT      | ID del empleado (FK)                 |
| timestamp          | TIMESTAMP | Momento de la detección              |
| emotion            | TEXT      | Emoción detectada                    |
| confidence         | REAL      | Confianza de la predicción (0-1)     |
| stress_level       | REAL      | Nivel de estrés calculado (0-1)      |
| face_box_x         | INTEGER   | Coordenada X del rostro              |
| face_box_y         | INTEGER   | Coordenada Y del rostro              |
| face_box_w         | INTEGER   | Ancho del rostro                     |
| face_box_h         | INTEGER   | Alto del rostro                      |

#### 4. `stress_alerts` - Alertas de Estrés

Registra alertas generadas por niveles críticos de estrés.

| Campo              | Type      | Descripción                          |
|--------------------|-----------|--------------------------------------|
| alert_id           | INTEGER   | ID único de alerta (PK)              |
| employee_id        | TEXT      | ID del empleado (FK)                 |
| timestamp          | TIMESTAMP | Momento de la alerta                 |
| alert_type         | TEXT      | Tipo de alerta (warning, critical)   |
| stress_level       | REAL      | Nivel de estrés que generó alerta    |
| duration_minutes   | INTEGER   | Duración del estado de estrés        |
| is_resolved        | BOOLEAN   | Si la alerta fue resuelta            |
| resolved_at        | TIMESTAMP | Momento de resolución                |
| notes              | TEXT      | Notas del supervisor                 |

#### 5. `reports` - Reportes Generados

Registro de reportes generados por el sistema.

| Campo              | Tipo      | Descripción                          |
|--------------------|-----------|--------------------------------------|
| report_id          | INTEGER   | ID único de reporte (PK)             |
| report_type        | TEXT      | Tipo (daily, weekly, monthly)        |
| period_start       | TIMESTAMP | Inicio del período                   |
| period_end         | TIMESTAMP | Fin del período                      |
| generated_at       | TIMESTAMP | Momento de generación                |
| file_path          | TEXT      | Ruta del archivo PDF                 |
| employee_id        | TEXT      | ID empleado (NULL = todos)           |

#### 6. `audit_log` - Log de Auditoría

Registra todas las acciones importantes del sistema.

| Campo              | Tipo      | Descripción                          |
|--------------------|-----------|--------------------------------------|
| log_id             | INTEGER   | ID único de log (PK)                 |
| timestamp          | TIMESTAMP | Momento del evento                   |
| event_type         | TEXT      | Tipo de evento                       |
| user_id            | TEXT      | Usuario que realizó la acción        |
| employee_id        | TEXT      | Empleado relacionado (si aplica)     |
| action             | TEXT      | Descripción de la acción             |
| ip_address         | TEXT      | IP de origen                         |
| details            | TEXT      | Detalles adicionales (JSON)          |

## Uso

### Inicializar Base de Datos

```bash
# Crear base de datos desde cero
python database/init_database.py

# Con datos de ejemplo
python database/init_database.py --seed
```

### Ejecutar Migraciones

```bash
# Aplicar todas las migraciones pendientes
python database/init_database.py --migrate

# Aplicar una migración específica
sqlite3 database/gloria_stress_system.db < database/migrations/001_initial_schema.sql
```

### Backup

```bash
# Backup manual
bash database/scripts/backup.sh

# Backup programado (agregar a crontab)
0 2 * * * /ruta/a/database/scripts/backup.sh
```

### Limpieza de Datos Antiguos

```bash
# Eliminar datos > 6 meses
sqlite3 database/gloria_stress_system.db < database/scripts/cleanup.sql
```

### Consultas Comunes

```sql
-- Obtener todas las detecciones de un empleado hoy
SELECT * FROM emotion_detections 
WHERE employee_id = 'EMP001' 
  AND DATE(timestamp) = DATE('now');

-- Empleados con mayor nivel de estrés promedio (última semana)
SELECT 
    e.employee_id,
    e.name,
    AVG(ed.stress_level) as avg_stress
FROM employees e
JOIN emotion_detections ed ON e.employee_id = ed.employee_id
WHERE ed.timestamp >= datetime('now', '-7 days')
GROUP BY e.employee_id
ORDER BY avg_stress DESC
LIMIT 10;

-- Alertas no resueltas
SELECT 
    sa.*,
    e.name,
    e.department
FROM stress_alerts sa
JOIN employees e ON sa.employee_id = e.employee_id
WHERE sa.is_resolved = 0
ORDER BY sa.timestamp DESC;

-- Estadísticas por departamento (último mes)
SELECT 
    e.department,
    COUNT(DISTINCT e.employee_id) as employees,
    COUNT(ed.detection_id) as detections,
    AVG(ed.stress_level) as avg_stress,
    COUNT(CASE WHEN sa.alert_id IS NOT NULL THEN 1 END) as alerts
FROM employees e
LEFT JOIN emotion_detections ed ON e.employee_id = ed.employee_id
    AND ed.timestamp >= datetime('now', '-30 days')
LEFT JOIN stress_alerts sa ON e.employee_id = sa.employee_id
    AND sa.timestamp >= datetime('now', '-30 days')
GROUP BY e.department;
```

## Mantenimiento

### Optimización

```sql
-- Analizar y optimizar base de datos
ANALYZE;
VACUUM;

-- Reconstruir índices
REINDEX;
```

### Integridad

```sql
-- Verificar integridad
PRAGMA integrity_check;

-- Verificar claves foráneas
PRAGMA foreign_key_check;
```

### Estadísticas

```sql
-- Tamaño de la base de datos
SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size();

-- Número de registros por tabla
SELECT 
    'employees' as table_name, COUNT(*) as count FROM employees
UNION ALL
SELECT 'detection_sessions', COUNT(*) FROM detection_sessions
UNION ALL
SELECT 'emotion_detections', COUNT(*) FROM emotion_detections
UNION ALL
SELECT 'stress_alerts', COUNT(*) FROM stress_alerts
UNION ALL
SELECT 'reports', COUNT(*) FROM reports
UNION ALL
SELECT 'audit_log', COUNT(*) FROM audit_log;
```

## Migración a PostgreSQL

Para migrar a PostgreSQL en producción:

1. Exportar datos:
   ```bash
   python database/scripts/export_to_postgres.py
   ```

2. Ajustar tipos de datos:
   - `TEXT` → `VARCHAR` o `TEXT`
   - `REAL` → `FLOAT` o `NUMERIC`
   - `INTEGER` → `INTEGER` o `BIGINT`

3. Actualizar conexiones en `backend/app/database.py`

## Seguridad

### Datos Sensibles

⚠️ **IMPORTANTE**: Esta base de datos contiene datos biométricos y personales.

- **Embeddings faciales**: Datos biométricos protegidos por ley
- **Información personal**: Nombres, departamentos, cargos
- **Datos de salud**: Niveles de estrés, patrones emocionales

### Recomendaciones

1. ✅ **No versionar** el archivo `.db` en git
2. ✅ **Encriptar backups** con contraseña fuerte
3. ✅ **Limitar acceso** solo a personal autorizado
4. ✅ **Logs de auditoría** para todas las consultas
5. ✅ **Retención de datos**: Máximo 6 meses
6. ✅ **Derecho al olvido**: Script para eliminar datos de empleado

### Cumplimiento Legal

- ✅ Ley N° 29733 (Protección de Datos Personales - Perú)
- ✅ Consentimiento informado requerido
- ✅ Registro ante autoridad de protección de datos
- ✅ Política de privacidad disponible

## Troubleshooting

### Base de datos bloqueada

```bash
# Si aparece "database is locked"
fuser database/gloria_stress_system.db  # Ver qué proceso la usa
# Cerrar procesos que estén usando la BD
```

### Corrupción de datos

```bash
# Verificar integridad
sqlite3 database/gloria_stress_system.db "PRAGMA integrity_check;"

# Si está corrupta, restaurar desde backup
cp database/backups/latest.db database/gloria_stress_system.db
```

### Bajo rendimiento

```sql
-- Crear índices adicionales si hay consultas lentas
CREATE INDEX idx_detections_timestamp ON emotion_detections(timestamp);
CREATE INDEX idx_detections_employee ON emotion_detections(employee_id);
CREATE INDEX idx_alerts_timestamp ON stress_alerts(timestamp);
```

## Contacto

Para problemas con la base de datos:
- Email: db-admin@stressvision.com
- Documentación: Ver `docs/arquitectura/database_schema.md`

---

**Última actualización**: 22/10/2025


