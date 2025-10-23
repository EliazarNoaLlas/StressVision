-- ============================================================================
-- Nombre del archivo: cleanup.sql
-- Propósito: Script de limpieza de datos antiguos de StressVision
-- Autor: Equipo de Desarrollo StressVision
-- Fecha de creación: 22/10/2025
-- Empresa/Organización: GLORIA S.A. - StressVision Project
-- ============================================================================
-- Descripción:
-- Este script elimina datos antiguos de la base de datos para mantener el
-- rendimiento y cumplir con políticas de retención de datos. Incluye:
-- - Eliminación de detecciones antiguas (> 6 meses)
-- - Eliminación de logs de auditoría antiguos (> 1 año)
-- - Limpieza de sesiones completadas antiguas
-- - Limpieza de alertas resueltas antiguas
-- - Limpieza de reportes antiguos
-- - Optimización de la base de datos
--
-- IMPORTANTE: Este script ELIMINA datos permanentemente.
-- Asegúrese de tener backups recientes antes de ejecutar.
--
-- Uso:
--   sqlite3 database/gloria_stress_system.db < database/scripts/cleanup.sql
--
-- Para programar limpieza automática (crontab):
--   # Limpieza mensual el primer día de cada mes a las 3 AM
--   0 3 1 * * cd /ruta/a/proyecto && sqlite3 database/gloria_stress_system.db < database/scripts/cleanup.sql
-- ============================================================================

-- Habilitar claves foráneas para mantener integridad referencial
PRAGMA foreign_keys = ON;

-- ============================================================================
-- CONFIGURACIÓN DE RETENCIÓN
-- ============================================================================

-- Obtener configuraciones de retención desde system_config
-- Si no existen, usar valores por defecto

-- Crear tabla temporal para almacenar las configuraciones
CREATE TEMP TABLE IF NOT EXISTS temp_config AS
SELECT 
    CAST(COALESCE(
        (SELECT config_value FROM system_config WHERE config_key = 'data_retention_days'),
        '180'
    ) AS INTEGER) as retention_days_detections,
    CAST(COALESCE(
        (SELECT config_value FROM system_config WHERE config_key = 'log_retention_days'),
        '365'
    ) AS INTEGER) as retention_days_logs;

-- ============================================================================
-- REGISTRO DE INICIO DE LIMPIEZA
-- ============================================================================

-- Registrar en el log de auditoría que se inició el proceso de limpieza
INSERT INTO audit_log (event_type, user_id, action, severity)
VALUES (
    'cleanup_started',
    'system',
    'Iniciando proceso de limpieza de datos antiguos',
    'info'
);

-- ============================================================================
-- ESTADÍSTICAS ANTES DE LA LIMPIEZA
-- ============================================================================

-- Crear tabla temporal para almacenar las estadísticas
CREATE TEMP TABLE IF NOT EXISTS cleanup_stats (
    table_name TEXT,
    records_before INTEGER,
    records_deleted INTEGER,
    records_after INTEGER
);

-- Registrar conteo antes de la limpieza
INSERT INTO cleanup_stats (table_name, records_before, records_deleted, records_after)
SELECT 
    'emotion_detections' as table_name,
    (SELECT COUNT(*) FROM emotion_detections) as records_before,
    0 as records_deleted,
    0 as records_after;

INSERT INTO cleanup_stats (table_name, records_before, records_deleted, records_after)
SELECT 
    'detection_sessions',
    (SELECT COUNT(*) FROM detection_sessions),
    0, 0;

INSERT INTO cleanup_stats (table_name, records_before, records_deleted, records_after)
SELECT 
    'stress_alerts',
    (SELECT COUNT(*) FROM stress_alerts),
    0, 0;

INSERT INTO cleanup_stats (table_name, records_before, records_deleted, records_after)
SELECT 
    'reports',
    (SELECT COUNT(*) FROM reports),
    0, 0;

INSERT INTO cleanup_stats (table_name, records_before, records_deleted, records_after)
SELECT 
    'audit_log',
    (SELECT COUNT(*) FROM audit_log),
    0, 0;

-- ============================================================================
-- LIMPIEZA DE DETECCIONES ANTIGUAS
-- ============================================================================

-- Eliminar detecciones de emociones más antiguas que retention_days_detections
-- Estas son las detecciones individuales, la tabla más grande del sistema

-- Primero, contar cuántas se eliminarán
CREATE TEMP TABLE IF NOT EXISTS temp_detections_to_delete AS
SELECT detection_id
FROM emotion_detections
WHERE timestamp < datetime('now', '-' || (SELECT retention_days_detections FROM temp_config) || ' days');

-- Registrar en el log cuántas se eliminarán
INSERT INTO audit_log (event_type, user_id, action, severity, details)
SELECT 
    'cleanup_detections',
    'system',
    'Eliminando ' || COUNT(*) || ' detecciones antiguas',
    'info',
    json_object(
        'table', 'emotion_detections',
        'count', COUNT(*),
        'retention_days', (SELECT retention_days_detections FROM temp_config)
    )
FROM temp_detections_to_delete;

-- Eliminar las detecciones antiguas
-- ON DELETE CASCADE eliminará también referencias en otras tablas si las hay
DELETE FROM emotion_detections
WHERE detection_id IN (SELECT detection_id FROM temp_detections_to_delete);

-- Actualizar estadísticas
UPDATE cleanup_stats
SET 
    records_deleted = (SELECT COUNT(*) FROM temp_detections_to_delete),
    records_after = (SELECT COUNT(*) FROM emotion_detections)
WHERE table_name = 'emotion_detections';

-- Limpiar tabla temporal
DROP TABLE temp_detections_to_delete;

-- ============================================================================
-- LIMPIEZA DE SESIONES ANTIGUAS
-- ============================================================================

-- Eliminar sesiones completadas más antiguas que retention_days_detections
-- Solo eliminar sesiones que ya no tienen detecciones asociadas

-- Contar sesiones a eliminar
CREATE TEMP TABLE IF NOT EXISTS temp_sessions_to_delete AS
SELECT s.session_id
FROM detection_sessions s
WHERE s.end_time IS NOT NULL  -- Solo sesiones completadas
  AND s.end_time < datetime('now', '-' || (SELECT retention_days_detections FROM temp_config) || ' days')
  AND NOT EXISTS (
      -- No eliminar si aún tiene detecciones
      SELECT 1 FROM emotion_detections ed 
      WHERE ed.session_id = s.session_id
  );

-- Registrar en el log
INSERT INTO audit_log (event_type, user_id, action, severity, details)
SELECT 
    'cleanup_sessions',
    'system',
    'Eliminando ' || COUNT(*) || ' sesiones antiguas',
    'info',
    json_object('table', 'detection_sessions', 'count', COUNT(*))
FROM temp_sessions_to_delete;

-- Eliminar las sesiones
DELETE FROM detection_sessions
WHERE session_id IN (SELECT session_id FROM temp_sessions_to_delete);

-- Actualizar estadísticas
UPDATE cleanup_stats
SET 
    records_deleted = (SELECT COUNT(*) FROM temp_sessions_to_delete),
    records_after = (SELECT COUNT(*) FROM detection_sessions)
WHERE table_name = 'detection_sessions';

-- Limpiar tabla temporal
DROP TABLE temp_sessions_to_delete;

-- ============================================================================
-- LIMPIEZA DE ALERTAS RESUELTAS ANTIGUAS
-- ============================================================================

-- Eliminar alertas que fueron resueltas hace más de retention_days_detections
-- Las alertas no resueltas NUNCA se eliminan automáticamente

-- Contar alertas a eliminar
CREATE TEMP TABLE IF NOT EXISTS temp_alerts_to_delete AS
SELECT alert_id
FROM stress_alerts
WHERE is_resolved = 1  -- Solo alertas resueltas
  AND resolved_at < datetime('now', '-' || (SELECT retention_days_detections FROM temp_config) || ' days');

-- Registrar en el log
INSERT INTO audit_log (event_type, user_id, action, severity, details)
SELECT 
    'cleanup_alerts',
    'system',
    'Eliminando ' || COUNT(*) || ' alertas resueltas antiguas',
    'info',
    json_object('table', 'stress_alerts', 'count', COUNT(*))
FROM temp_alerts_to_delete;

-- Eliminar las alertas
DELETE FROM stress_alerts
WHERE alert_id IN (SELECT alert_id FROM temp_alerts_to_delete);

-- Actualizar estadísticas
UPDATE cleanup_stats
SET 
    records_deleted = (SELECT COUNT(*) FROM temp_alerts_to_delete),
    records_after = (SELECT COUNT(*) FROM stress_alerts)
WHERE table_name = 'stress_alerts';

-- Limpiar tabla temporal
DROP TABLE temp_alerts_to_delete;

-- ============================================================================
-- LIMPIEZA DE REPORTES ANTIGUOS
-- ============================================================================

-- Eliminar reportes generados hace más de retention_days_detections
-- Solo eliminar registros de la BD, los archivos PDF deben eliminarse manualmente

-- Contar reportes a eliminar
CREATE TEMP TABLE IF NOT EXISTS temp_reports_to_delete AS
SELECT report_id, file_path
FROM reports
WHERE generated_at < datetime('now', '-' || (SELECT retention_days_detections FROM temp_config) || ' days');

-- Registrar en el log (incluyendo las rutas de archivos para eliminación manual)
INSERT INTO audit_log (event_type, user_id, action, severity, details)
SELECT 
    'cleanup_reports',
    'system',
    'Eliminando ' || COUNT(*) || ' registros de reportes antiguos',
    'warning',
    json_object(
        'table', 'reports',
        'count', COUNT(*),
        'note', 'Los archivos PDF deben eliminarse manualmente'
    )
FROM temp_reports_to_delete;

-- Eliminar los registros de reportes
DELETE FROM reports
WHERE report_id IN (SELECT report_id FROM temp_reports_to_delete);

-- Actualizar estadísticas
UPDATE cleanup_stats
SET 
    records_deleted = (SELECT COUNT(*) FROM temp_reports_to_delete),
    records_after = (SELECT COUNT(*) FROM reports)
WHERE table_name = 'reports';

-- Limpiar tabla temporal
DROP TABLE temp_reports_to_delete;

-- ============================================================================
-- LIMPIEZA DE LOGS DE AUDITORÍA ANTIGUOS
-- ============================================================================

-- Eliminar logs de auditoría más antiguos que retention_days_logs
-- Mantener TODOS los logs de tipo 'critical' independientemente de la fecha

-- Contar logs a eliminar
CREATE TEMP TABLE IF NOT EXISTS temp_logs_to_delete AS
SELECT log_id
FROM audit_log
WHERE timestamp < datetime('now', '-' || (SELECT retention_days_logs FROM temp_config) || ' days')
  AND severity != 'critical'  -- Nunca eliminar logs críticos
  AND event_type NOT IN ('employee_enrolled', 'employee_deleted');  -- Mantener eventos importantes

-- Registrar cuántos logs se eliminarán
INSERT INTO audit_log (event_type, user_id, action, severity, details)
SELECT 
    'cleanup_audit_log',
    'system',
    'Eliminando ' || COUNT(*) || ' logs de auditoría antiguos',
    'info',
    json_object(
        'table', 'audit_log',
        'count', COUNT(*),
        'retention_days', (SELECT retention_days_logs FROM temp_config),
        'note', 'Se mantienen logs críticos y eventos importantes'
    )
FROM temp_logs_to_delete;

-- Eliminar los logs antiguos
DELETE FROM audit_log
WHERE log_id IN (SELECT log_id FROM temp_logs_to_delete);

-- Actualizar estadísticas
UPDATE cleanup_stats
SET 
    records_deleted = (SELECT COUNT(*) FROM temp_logs_to_delete),
    records_after = (SELECT COUNT(*) FROM audit_log)
WHERE table_name = 'audit_log';

-- Limpiar tabla temporal
DROP TABLE temp_logs_to_delete;

-- ============================================================================
-- LIMPIEZA DE EMPLEADOS INACTIVOS SIN DATOS
-- ============================================================================

-- Opcional: Eliminar empleados marcados como inactivos que no tienen
-- detecciones ni alertas asociadas (huérfanos)
-- COMENTADO por defecto por seguridad, descomentar si se desea

/*
-- Contar empleados huérfanos inactivos
CREATE TEMP TABLE IF NOT EXISTS temp_inactive_employees AS
SELECT e.employee_id, e.name
FROM employees e
WHERE e.is_active = 0
  AND NOT EXISTS (SELECT 1 FROM emotion_detections WHERE employee_id = e.employee_id)
  AND NOT EXISTS (SELECT 1 FROM stress_alerts WHERE employee_id = e.employee_id)
  AND e.enrollment_date < datetime('now', '-180 days');

-- Registrar en el log
INSERT INTO audit_log (event_type, user_id, action, severity, details)
SELECT 
    'cleanup_inactive_employees',
    'system',
    'Eliminando ' || COUNT(*) || ' empleados inactivos sin datos',
    'warning',
    json_object('count', COUNT(*))
FROM temp_inactive_employees;

-- Eliminar empleados huérfanos
DELETE FROM employees
WHERE employee_id IN (SELECT employee_id FROM temp_inactive_employees);

DROP TABLE temp_inactive_employees;
*/

-- ============================================================================
-- OPTIMIZACIÓN DE LA BASE DE DATOS
-- ============================================================================

-- Registrar inicio de optimización
INSERT INTO audit_log (event_type, user_id, action, severity)
VALUES ('cleanup_optimize_start', 'system', 'Iniciando optimización de la base de datos', 'info');

-- Analizar las tablas para actualizar estadísticas del query planner
-- Esto mejora el rendimiento de las consultas
ANALYZE;

-- Reconstruir índices para eliminar fragmentación
REINDEX;

-- Liberar espacio no utilizado y desfragmentar la base de datos
-- VACUUM puede tardar varios minutos en bases de datos grandes
VACUUM;

-- Registrar fin de optimización
INSERT INTO audit_log (event_type, user_id, action, severity)
VALUES ('cleanup_optimize_end', 'system', 'Optimización de la base de datos completada', 'info');

-- ============================================================================
-- REPORTE FINAL DE LIMPIEZA
-- ============================================================================

-- Crear una vista temporal con el resumen de la limpieza
CREATE TEMP VIEW v_cleanup_summary AS
SELECT 
    table_name as "Tabla",
    records_before as "Registros Antes",
    records_deleted as "Registros Eliminados",
    records_after as "Registros Después",
    ROUND((CAST(records_deleted AS REAL) / NULLIF(records_before, 0)) * 100, 2) as "% Eliminado"
FROM cleanup_stats
ORDER BY records_deleted DESC;

-- Mostrar el resumen (se verá en la salida de sqlite3)
.mode column
.headers on
.width 20 15 20 15 12

SELECT '============================================' as '';
SELECT '  RESUMEN DE LIMPIEZA DE BASE DE DATOS' as '';
SELECT '============================================' as '';
SELECT '' as '';

SELECT * FROM v_cleanup_summary;

SELECT '' as '';
SELECT 'Fecha de limpieza: ' || datetime('now', 'localtime') as '';
SELECT 'Configuración de retención:' as '';
SELECT '  - Detecciones: ' || retention_days_detections || ' días' as ''
FROM temp_config;
SELECT '  - Logs: ' || retention_days_logs || ' días' as ''
FROM temp_config;
SELECT '' as '';
SELECT '============================================' as '';

-- Registrar finalización exitosa de la limpieza
INSERT INTO audit_log (event_type, user_id, action, severity, details)
SELECT 
    'cleanup_completed',
    'system',
    'Proceso de limpieza completado exitosamente',
    'info',
    json_object(
        'total_records_deleted', (SELECT SUM(records_deleted) FROM cleanup_stats),
        'timestamp', datetime('now'),
        'tables_affected', (SELECT COUNT(*) FROM cleanup_stats WHERE records_deleted > 0)
    );

-- ============================================================================
-- VERIFICACIÓN FINAL
-- ============================================================================

-- Verificar integridad de la base de datos después de la limpieza
PRAGMA integrity_check;

-- Verificar claves foráneas
PRAGMA foreign_key_check;

-- ============================================================================
-- LIMPIEZA DE TABLAS TEMPORALES
-- ============================================================================

-- Limpiar todas las tablas y vistas temporales creadas
DROP TABLE IF EXISTS temp_config;
DROP TABLE IF EXISTS cleanup_stats;
DROP VIEW IF EXISTS v_cleanup_summary;

-- ============================================================================
-- FIN DEL SCRIPT DE LIMPIEZA
-- ============================================================================

-- Mensaje final
SELECT '✓ Limpieza completada exitosamente' as "Resultado";
SELECT 'Revise el audit_log para ver detalles completos' as "Nota";


