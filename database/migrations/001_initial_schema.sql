-- ============================================================================
-- Nombre del archivo: 001_initial_schema.sql
-- Propósito: Migración inicial - Creación del esquema base de StressVision
-- Autor: Equipo de Desarrollo StressVision
-- Fecha de creación: 22/10/2025
-- Empresa/Organización: GLORIA S.A. - StressVision Project
-- ============================================================================
-- Descripción:
-- Esta es la primera migración del sistema StressVision. Crea todas las
-- tablas, índices y configuraciones iniciales necesarias para el
-- funcionamiento básico del sistema.
-- 
-- Esta migración es idéntica al archivo schema.sql y sirve como punto de
-- partida para futuras migraciones incrementales.
-- 
-- IMPORTANTE: Esta migración solo debe ejecutarse UNA VEZ al crear el
-- sistema por primera vez. Migraciones posteriores (002_, 003_, etc.)
-- contendrán solo cambios incrementales.
-- ============================================================================

-- Registrar el inicio de la migración en el log de auditoría
-- (Esto se ejecutará solo si la tabla audit_log ya existe)
INSERT OR IGNORE INTO audit_log (event_type, user_id, action, severity)
VALUES ('migration_started', 'system', 'Iniciando migración 001_initial_schema', 'info');

-- ============================================================================
-- NOTA: 
-- Si esta migración se ejecuta después de que schema.sql ya creó las tablas,
-- los comandos "CREATE TABLE IF NOT EXISTS" simplemente no harán nada.
-- Esto hace que la migración sea idempotente (puede ejecutarse múltiples
-- veces sin causar errores).
-- ============================================================================

-- Habilitar claves foráneas
PRAGMA foreign_keys = ON;

-- Configurar encoding UTF-8
PRAGMA encoding = "UTF-8";

-- ============================================================================
-- CREACIÓN DE TABLAS
-- ============================================================================

-- Tabla de empleados
CREATE TABLE IF NOT EXISTS employees (
    employee_id TEXT PRIMARY KEY NOT NULL,
    name TEXT NOT NULL,
    department TEXT NOT NULL,
    position TEXT,
    email TEXT,
    enrollment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    face_embedding TEXT NOT NULL,
    is_active BOOLEAN DEFAULT 1,
    consent_signed BOOLEAN DEFAULT 0,
    consent_date TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT,
    CHECK (length(face_embedding) > 0),
    CHECK (consent_date IS NULL OR consent_signed = 1)
);

-- Tabla de sesiones de detección
CREATE TABLE IF NOT EXISTS detection_sessions (
    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id TEXT NOT NULL,
    location TEXT,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP,
    frames_processed INTEGER DEFAULT 0,
    detections_count INTEGER DEFAULT 0,
    unique_employees INTEGER DEFAULT 0,
    duration_seconds INTEGER,
    status TEXT DEFAULT 'active',
    model_version TEXT,
    notes TEXT,
    CHECK (end_time IS NULL OR end_time >= start_time),
    CHECK (frames_processed >= 0),
    CHECK (detections_count >= 0),
    CHECK (unique_employees >= 0)
);

-- Tabla de detecciones de emociones
CREATE TABLE IF NOT EXISTS emotion_detections (
    detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    employee_id TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    emotion TEXT NOT NULL,
    confidence REAL NOT NULL,
    stress_level REAL,
    face_box_x INTEGER,
    face_box_y INTEGER,
    face_box_w INTEGER,
    face_box_h INTEGER,
    face_quality REAL,
    head_pose_pitch REAL,
    head_pose_yaw REAL,
    head_pose_roll REAL,
    emotion_probabilities TEXT,
    FOREIGN KEY (session_id) REFERENCES detection_sessions(session_id) ON DELETE CASCADE,
    FOREIGN KEY (employee_id) REFERENCES employees(employee_id) ON DELETE SET NULL,
    CHECK (confidence >= 0 AND confidence <= 1),
    CHECK (stress_level IS NULL OR (stress_level >= 0 AND stress_level <= 1)),
    CHECK (emotion IN ('happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral'))
);

-- Tabla de alertas de estrés
CREATE TABLE IF NOT EXISTS stress_alerts (
    alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    alert_type TEXT NOT NULL,
    stress_level REAL NOT NULL,
    duration_minutes INTEGER,
    consecutive_detections INTEGER,
    is_resolved BOOLEAN DEFAULT 0,
    resolved_at TIMESTAMP,
    resolved_by TEXT,
    action_taken TEXT,
    notes TEXT,
    priority TEXT DEFAULT 'medium',
    employee_notified BOOLEAN DEFAULT 0,
    supervisor_notified BOOLEAN DEFAULT 0,
    FOREIGN KEY (employee_id) REFERENCES employees(employee_id) ON DELETE CASCADE,
    CHECK (alert_type IN ('warning', 'critical')),
    CHECK (stress_level >= 0 AND stress_level <= 1),
    CHECK (is_resolved = 0 OR resolved_at IS NOT NULL),
    CHECK (priority IN ('low', 'medium', 'high'))
);

-- Tabla de reportes
CREATE TABLE IF NOT EXISTS reports (
    report_id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_type TEXT NOT NULL,
    period_start TIMESTAMP NOT NULL,
    period_end TIMESTAMP NOT NULL,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    generated_by TEXT,
    file_path TEXT NOT NULL,
    employee_id TEXT,
    department TEXT,
    format TEXT DEFAULT 'pdf',
    file_size INTEGER,
    summary TEXT,
    status TEXT DEFAULT 'generated',
    FOREIGN KEY (employee_id) REFERENCES employees(employee_id) ON DELETE SET NULL,
    CHECK (period_end >= period_start),
    CHECK (report_type IN ('daily', 'weekly', 'monthly', 'custom')),
    CHECK (format IN ('pdf', 'csv', 'json', 'xlsx'))
);

-- Tabla de log de auditoría
CREATE TABLE IF NOT EXISTS audit_log (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    event_type TEXT NOT NULL,
    user_id TEXT NOT NULL,
    employee_id TEXT,
    action TEXT NOT NULL,
    ip_address TEXT,
    details TEXT,
    result TEXT DEFAULT 'success',
    error_message TEXT,
    severity TEXT DEFAULT 'info',
    FOREIGN KEY (employee_id) REFERENCES employees(employee_id) ON DELETE SET NULL,
    CHECK (result IN ('success', 'failure', 'partial')),
    CHECK (severity IN ('info', 'warning', 'error', 'critical'))
);

-- Tabla de configuración del sistema
CREATE TABLE IF NOT EXISTS system_config (
    config_key TEXT PRIMARY KEY NOT NULL,
    config_value TEXT NOT NULL,
    value_type TEXT DEFAULT 'string',
    description TEXT,
    category TEXT DEFAULT 'system',
    is_editable BOOLEAN DEFAULT 1,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_by TEXT,
    CHECK (value_type IN ('string', 'integer', 'float', 'boolean', 'json'))
);

-- ============================================================================
-- CREACIÓN DE ÍNDICES
-- ============================================================================

-- Índices para employees
CREATE INDEX IF NOT EXISTS idx_employees_department ON employees(department);
CREATE INDEX IF NOT EXISTS idx_employees_active ON employees(is_active);
CREATE INDEX IF NOT EXISTS idx_employees_name ON employees(name);

-- Índices para detection_sessions
CREATE INDEX IF NOT EXISTS idx_sessions_device ON detection_sessions(device_id);
CREATE INDEX IF NOT EXISTS idx_sessions_start ON detection_sessions(start_time);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON detection_sessions(status);

-- Índices para emotion_detections
CREATE INDEX IF NOT EXISTS idx_detections_employee_timestamp 
ON emotion_detections(employee_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_detections_session 
ON emotion_detections(session_id);
CREATE INDEX IF NOT EXISTS idx_detections_timestamp 
ON emotion_detections(timestamp);
CREATE INDEX IF NOT EXISTS idx_detections_stress 
ON emotion_detections(stress_level);

-- Índices para stress_alerts
CREATE INDEX IF NOT EXISTS idx_alerts_employee ON stress_alerts(employee_id);
CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON stress_alerts(timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_unresolved 
ON stress_alerts(is_resolved, priority, timestamp);

-- Índices para reports
CREATE INDEX IF NOT EXISTS idx_reports_type ON reports(report_type);
CREATE INDEX IF NOT EXISTS idx_reports_generated ON reports(generated_at);
CREATE INDEX IF NOT EXISTS idx_reports_employee ON reports(employee_id);

-- Índices para audit_log
CREATE INDEX IF NOT EXISTS idx_audit_event_type ON audit_log(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_employee ON audit_log(employee_id);

-- Índice para system_config
CREATE INDEX IF NOT EXISTS idx_config_category ON system_config(category);

-- ============================================================================
-- CREACIÓN DE TRIGGERS
-- ============================================================================

-- Trigger para actualizar last_updated en employees
DROP TRIGGER IF EXISTS update_employee_timestamp;
CREATE TRIGGER update_employee_timestamp
AFTER UPDATE ON employees
FOR EACH ROW
BEGIN
    UPDATE employees 
    SET last_updated = CURRENT_TIMESTAMP 
    WHERE employee_id = NEW.employee_id;
END;

-- Trigger para calcular duration_seconds en detection_sessions
DROP TRIGGER IF EXISTS calculate_session_duration;
CREATE TRIGGER calculate_session_duration
AFTER UPDATE OF end_time ON detection_sessions
FOR EACH ROW
WHEN NEW.end_time IS NOT NULL
BEGIN
    UPDATE detection_sessions
    SET duration_seconds = (
        CAST((julianday(NEW.end_time) - julianday(NEW.start_time)) * 86400 AS INTEGER)
    )
    WHERE session_id = NEW.session_id;
END;

-- Trigger para registrar enrollment en audit_log
DROP TRIGGER IF EXISTS audit_employee_enrollment;
CREATE TRIGGER audit_employee_enrollment
AFTER INSERT ON employees
FOR EACH ROW
BEGIN
    INSERT INTO audit_log (event_type, user_id, employee_id, action, severity)
    VALUES ('employee_enrolled', 'system', NEW.employee_id, 
            'Nuevo empleado registrado: ' || NEW.name, 'info');
END;

-- Trigger para registrar eliminación en audit_log
DROP TRIGGER IF EXISTS audit_employee_deletion;
CREATE TRIGGER audit_employee_deletion
AFTER DELETE ON employees
FOR EACH ROW
BEGIN
    INSERT INTO audit_log (event_type, user_id, employee_id, action, severity)
    VALUES ('employee_deleted', 'system', OLD.employee_id, 
            'Empleado eliminado: ' || OLD.name, 'warning');
END;

-- Trigger para registrar alertas críticas en audit_log
DROP TRIGGER IF EXISTS audit_critical_alert;
CREATE TRIGGER audit_critical_alert
AFTER INSERT ON stress_alerts
FOR EACH ROW
WHEN NEW.alert_type = 'critical'
BEGIN
    INSERT INTO audit_log (event_type, user_id, employee_id, action, severity)
    VALUES ('alert_generated', 'system', NEW.employee_id, 
            'Alerta crítica generada - Nivel de estrés: ' || NEW.stress_level, 'critical');
END;

-- ============================================================================
-- CREACIÓN DE VISTAS
-- ============================================================================

-- Vista de empleados activos con estadísticas
DROP VIEW IF EXISTS v_active_employees;
CREATE VIEW v_active_employees AS
SELECT 
    e.employee_id,
    e.name,
    e.department,
    e.position,
    e.enrollment_date,
    e.consent_signed,
    COUNT(DISTINCT ed.detection_id) as total_detections,
    AVG(ed.stress_level) as avg_stress_level,
    COUNT(DISTINCT sa.alert_id) as total_alerts,
    MAX(ed.timestamp) as last_detection
FROM employees e
LEFT JOIN emotion_detections ed ON e.employee_id = ed.employee_id
LEFT JOIN stress_alerts sa ON e.employee_id = sa.employee_id
WHERE e.is_active = 1
GROUP BY e.employee_id;

-- Vista de alertas pendientes
DROP VIEW IF EXISTS v_pending_alerts;
CREATE VIEW v_pending_alerts AS
SELECT 
    sa.alert_id,
    sa.employee_id,
    e.name as employee_name,
    e.department,
    e.position,
    sa.timestamp,
    sa.alert_type,
    sa.stress_level,
    sa.duration_minutes,
    sa.priority
FROM stress_alerts sa
JOIN employees e ON sa.employee_id = e.employee_id
WHERE sa.is_resolved = 0
ORDER BY sa.priority DESC, sa.timestamp DESC;

-- Vista de estadísticas diarias
DROP VIEW IF EXISTS v_daily_detection_stats;
CREATE VIEW v_daily_detection_stats AS
SELECT 
    DATE(timestamp) as detection_date,
    COUNT(detection_id) as total_detections,
    COUNT(DISTINCT employee_id) as unique_employees,
    AVG(stress_level) as avg_stress_level,
    MAX(stress_level) as max_stress_level,
    AVG(confidence) as avg_confidence
FROM emotion_detections
GROUP BY DATE(timestamp)
ORDER BY detection_date DESC;

-- ============================================================================
-- INSERCIÓN DE DATOS INICIALES
-- ============================================================================

-- Insertar configuraciones predeterminadas del sistema
INSERT OR IGNORE INTO system_config (config_key, config_value, value_type, description, category) VALUES
('stress_threshold_warning', '0.65', 'float', 'Umbral de estrés para generar advertencia', 'alerts'),
('stress_threshold_critical', '0.85', 'float', 'Umbral de estrés para generar alerta crítica', 'alerts'),
('min_confidence_threshold', '0.60', 'float', 'Confianza mínima para aceptar una detección', 'detection'),
('alert_cooldown_minutes', '30', 'integer', 'Tiempo mínimo entre alertas del mismo empleado', 'alerts'),
('consecutive_detections_alert', '5', 'integer', 'Detecciones consecutivas de alto estrés para alerta', 'alerts'),
('data_retention_days', '180', 'integer', 'Días de retención de detecciones (6 meses)', 'system'),
('log_retention_days', '365', 'integer', 'Días de retención de logs de auditoría (1 año)', 'system'),
('report_generation_time', '02:00', 'string', 'Hora de generación automática de reportes', 'reports'),
('auto_generate_daily_reports', 'true', 'boolean', 'Generar reportes diarios automáticamente', 'reports'),
('system_version', '1.0.0', 'string', 'Versión del sistema StressVision', 'system'),
('model_version', '1.0.0', 'string', 'Versión del modelo de detección de emociones', 'system'),
('maintenance_mode', 'false', 'boolean', 'Modo de mantenimiento del sistema', 'system');

-- Registrar la finalización exitosa de la migración
INSERT INTO audit_log (event_type, user_id, action, severity)
VALUES ('migration_completed', 'system', 'Migración 001_initial_schema completada exitosamente', 'info');

-- ============================================================================
-- VERIFICACIÓN FINAL
-- ============================================================================

-- Verificar integridad de la base de datos
PRAGMA integrity_check;

-- Verificar claves foráneas
PRAGMA foreign_key_check;

-- Analizar las tablas para optimizar consultas futuras
ANALYZE;

-- ============================================================================
-- FIN DE LA MIGRACIÓN
-- ============================================================================

-- Mensaje de confirmación (se mostrará en los logs)
SELECT 'Migración 001_initial_schema aplicada exitosamente' AS resultado;


