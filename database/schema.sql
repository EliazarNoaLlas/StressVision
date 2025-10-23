-- ============================================================================
-- Nombre del archivo: schema.sql
-- Propósito: Esquema completo de la base de datos del sistema StressVision
-- Autor: Equipo de Desarrollo StressVision
-- Fecha de creación: 22/10/2025
-- Empresa/Organización: GLORIA S.A. - StressVision Project
-- ============================================================================
-- Descripción:
-- Este archivo contiene el esquema completo de la base de datos SQLite para
-- el sistema de detección de estrés laboral StressVision. Incluye todas las
-- tablas, índices, triggers y constraints necesarios para el funcionamiento
-- del sistema.
-- ============================================================================

-- Habilitar claves foráneas en SQLite
-- SQLite no las habilita por defecto, esto asegura la integridad referencial
PRAGMA foreign_keys = ON;

-- Configurar encoding a UTF-8
-- Garantiza compatibilidad con caracteres especiales en español
PRAGMA encoding = "UTF-8";

-- ============================================================================
-- TABLA: employees
-- ============================================================================
-- Descripción: Almacena la información de todos los empleados registrados
-- en el sistema. Incluye datos personales, información laboral y el embedding
-- facial necesario para el reconocimiento.
-- ============================================================================

CREATE TABLE IF NOT EXISTS employees (
    -- ID único del empleado (Ej: EMP001, EMP002)
    -- PRIMARY KEY asegura unicidad y crea índice automático
    employee_id TEXT PRIMARY KEY NOT NULL,
    
    -- Nombre completo del empleado
    -- NOT NULL asegura que siempre se proporcione un nombre
    name TEXT NOT NULL,
    
    -- Departamento o área de trabajo
    department TEXT NOT NULL,
    
    -- Cargo o posición laboral
    position TEXT,
    
    -- Email corporativo del empleado
    email TEXT,
    
    -- Fecha y hora de registro en el sistema
    -- DEFAULT CURRENT_TIMESTAMP asigna automáticamente la fecha actual
    enrollment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Embedding facial (vector de 512 dimensiones) almacenado como JSON
    -- Este vector permite el reconocimiento facial del empleado
    face_embedding TEXT NOT NULL,
    
    -- Indica si el empleado está activo en el sistema
    -- 1 = activo, 0 = inactivo (ej: empleado que ya no trabaja en la empresa)
    is_active BOOLEAN DEFAULT 1,
    
    -- Indica si el empleado firmó el consentimiento informado
    -- Requisito legal para el procesamiento de datos biométricos
    consent_signed BOOLEAN DEFAULT 0,
    
    -- Fecha en que se firmó el consentimiento
    consent_date TIMESTAMP,
    
    -- Última actualización del registro
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Notas adicionales sobre el empleado (opcional)
    notes TEXT,
    
    -- Constraint: El embedding no puede estar vacío
    CHECK (length(face_embedding) > 0),
    
    -- Constraint: Si hay fecha de consentimiento, debe estar firmado
    CHECK (consent_date IS NULL OR consent_signed = 1)
);

-- Índice para búsquedas rápidas por departamento
CREATE INDEX IF NOT EXISTS idx_employees_department ON employees(department);

-- Índice para filtrar empleados activos
CREATE INDEX IF NOT EXISTS idx_employees_active ON employees(is_active);

-- Índice para búsquedas por nombre (útil para autocompletado)
CREATE INDEX IF NOT EXISTS idx_employees_name ON employees(name);

-- ============================================================================
-- TABLA: detection_sessions
-- ============================================================================
-- Descripción: Representa una sesión de monitoreo continuo. Una sesión
-- comienza cuando se inicia el sistema de detección en un dispositivo edge
-- y termina cuando se detiene.
-- ============================================================================

CREATE TABLE IF NOT EXISTS detection_sessions (
    -- ID único de la sesión (autoincremental)
    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- ID del dispositivo edge que generó la sesión (Ej: PI001, PI002)
    device_id TEXT NOT NULL,
    
    -- Ubicación física del dispositivo (Ej: "Planta A - Línea 3")
    location TEXT,
    
    -- Momento de inicio de la sesión
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Momento de finalización de la sesión
    -- NULL mientras la sesión está activa
    end_time TIMESTAMP,
    
    -- Total de frames de video procesados en esta sesión
    frames_processed INTEGER DEFAULT 0,
    
    -- Total de detecciones de rostros realizadas
    detections_count INTEGER DEFAULT 0,
    
    -- Total de empleados únicos detectados
    unique_employees INTEGER DEFAULT 0,
    
    -- Duración total de la sesión en segundos (calculado)
    duration_seconds INTEGER,
    
    -- Estado de la sesión: 'active', 'completed', 'error'
    status TEXT DEFAULT 'active',
    
    -- Versión del modelo de IA utilizado
    model_version TEXT,
    
    -- Notas o errores durante la sesión
    notes TEXT,
    
    -- Constraint: end_time debe ser posterior a start_time
    CHECK (end_time IS NULL OR end_time >= start_time),
    
    -- Constraint: contadores no pueden ser negativos
    CHECK (frames_processed >= 0),
    CHECK (detections_count >= 0),
    CHECK (unique_employees >= 0)
);

-- Índice para búsquedas por dispositivo
CREATE INDEX IF NOT EXISTS idx_sessions_device ON detection_sessions(device_id);

-- Índice para búsquedas por fecha de inicio
CREATE INDEX IF NOT EXISTS idx_sessions_start ON detection_sessions(start_time);

-- Índice para filtrar sesiones activas
CREATE INDEX IF NOT EXISTS idx_sessions_status ON detection_sessions(status);

-- ============================================================================
-- TABLA: emotion_detections
-- ============================================================================
-- Descripción: Almacena cada detección individual de emoción. Esta es la
-- tabla más activa del sistema, recibiendo inserts continuos durante las
-- sesiones de monitoreo.
-- ============================================================================

CREATE TABLE IF NOT EXISTS emotion_detections (
    -- ID único de la detección (autoincremental)
    detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- ID de la sesión a la que pertenece esta detección
    -- CASCADE significa que si se elimina una sesión, se eliminan sus detecciones
    session_id INTEGER NOT NULL,
    FOREIGN KEY (session_id) REFERENCES detection_sessions(session_id) ON DELETE CASCADE,
    
    -- ID del empleado detectado
    -- Si es NULL, significa que se detectó un rostro pero no se reconoció
    employee_id TEXT,
    FOREIGN KEY (employee_id) REFERENCES employees(employee_id) ON DELETE SET NULL,
    
    -- Momento exacto de la detección
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Emoción detectada: 'happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral'
    emotion TEXT NOT NULL,
    
    -- Confianza de la predicción de emoción (0.0 a 1.0)
    -- Valores altos indican mayor certeza
    confidence REAL NOT NULL,
    
    -- Nivel de estrés calculado (0.0 a 1.0)
    -- Calculado basado en el patrón emocional y otros factores
    stress_level REAL,
    
    -- Coordenadas de la caja delimitadora del rostro en el frame
    -- Útil para debugging y análisis de calidad
    face_box_x INTEGER,
    face_box_y INTEGER,
    face_box_w INTEGER,
    face_box_h INTEGER,
    
    -- Calidad de la detección facial (0.0 a 1.0)
    -- Indica qué tan clara y completa es la imagen del rostro
    face_quality REAL,
    
    -- Ángulo de inclinación del rostro (en grados)
    -- Útil para filtrar detecciones de mala calidad
    head_pose_pitch REAL,
    head_pose_yaw REAL,
    head_pose_roll REAL,
    
    -- Vector de probabilidades de todas las emociones (JSON)
    -- Ej: {"happy": 0.1, "sad": 0.7, "angry": 0.2, ...}
    emotion_probabilities TEXT,
    
    -- Constraint: confidence debe estar entre 0 y 1
    CHECK (confidence >= 0 AND confidence <= 1),
    
    -- Constraint: stress_level debe estar entre 0 y 1
    CHECK (stress_level IS NULL OR (stress_level >= 0 AND stress_level <= 1)),
    
    -- Constraint: emociones válidas
    CHECK (emotion IN ('happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral'))
);

-- Índice compuesto para consultas por empleado y fecha
-- Este es el índice más importante para reportes
CREATE INDEX IF NOT EXISTS idx_detections_employee_timestamp 
ON emotion_detections(employee_id, timestamp);

-- Índice para consultas por sesión
CREATE INDEX IF NOT EXISTS idx_detections_session 
ON emotion_detections(session_id);

-- Índice para búsquedas por timestamp
CREATE INDEX IF NOT EXISTS idx_detections_timestamp 
ON emotion_detections(timestamp);

-- Índice para filtrar por nivel de estrés alto
CREATE INDEX IF NOT EXISTS idx_detections_stress 
ON emotion_detections(stress_level);

-- ============================================================================
-- TABLA: stress_alerts
-- ============================================================================
-- Descripción: Almacena alertas generadas cuando se detectan niveles
-- críticos de estrés. Estas alertas requieren acción por parte de
-- supervisores o personal de RRHH.
-- ============================================================================

CREATE TABLE IF NOT EXISTS stress_alerts (
    -- ID único de la alerta (autoincremental)
    alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- ID del empleado que generó la alerta
    employee_id TEXT NOT NULL,
    FOREIGN KEY (employee_id) REFERENCES employees(employee_id) ON DELETE CASCADE,
    
    -- Momento en que se generó la alerta
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Tipo de alerta: 'warning' (advertencia) o 'critical' (crítica)
    -- warning: nivel de estrés moderado-alto sostenido
    -- critical: nivel de estrés muy alto o emergencia
    alert_type TEXT NOT NULL,
    
    -- Nivel de estrés que activó la alerta (0.0 a 1.0)
    stress_level REAL NOT NULL,
    
    -- Duración del estado de estrés en minutos
    -- Se calcula basándose en las detecciones previas
    duration_minutes INTEGER,
    
    -- Número de detecciones consecutivas de alto estrés
    consecutive_detections INTEGER,
    
    -- Indica si la alerta fue atendida y resuelta
    is_resolved BOOLEAN DEFAULT 0,
    
    -- Momento en que se resolvió la alerta
    resolved_at TIMESTAMP,
    
    -- Usuario o supervisor que resolvió la alerta
    resolved_by TEXT,
    
    -- Acción tomada para resolver la situación
    -- Ej: "Empleado enviado a descanso", "Supervisor notificado"
    action_taken TEXT,
    
    -- Notas adicionales del supervisor
    notes TEXT,
    
    -- Prioridad de la alerta: 'low', 'medium', 'high'
    priority TEXT DEFAULT 'medium',
    
    -- Indica si se notificó al empleado
    employee_notified BOOLEAN DEFAULT 0,
    
    -- Indica si se notificó al supervisor
    supervisor_notified BOOLEAN DEFAULT 0,
    
    -- Constraint: tipos de alerta válidos
    CHECK (alert_type IN ('warning', 'critical')),
    
    -- Constraint: stress_level válido
    CHECK (stress_level >= 0 AND stress_level <= 1),
    
    -- Constraint: si está resuelta, debe tener fecha de resolución
    CHECK (is_resolved = 0 OR resolved_at IS NOT NULL),
    
    -- Constraint: prioridades válidas
    CHECK (priority IN ('low', 'medium', 'high'))
);

-- Índice para consultas por empleado
CREATE INDEX IF NOT EXISTS idx_alerts_employee 
ON stress_alerts(employee_id);

-- Índice para búsquedas por fecha
CREATE INDEX IF NOT EXISTS idx_alerts_timestamp 
ON stress_alerts(timestamp);

-- Índice compuesto para filtrar alertas no resueltas por prioridad
CREATE INDEX IF NOT EXISTS idx_alerts_unresolved 
ON stress_alerts(is_resolved, priority, timestamp);

-- ============================================================================
-- TABLA: reports
-- ============================================================================
-- Descripción: Registro de todos los reportes generados por el sistema.
-- Permite rastrear qué reportes se han generado, cuándo y para quién.
-- ============================================================================

CREATE TABLE IF NOT EXISTS reports (
    -- ID único del reporte (autoincremental)
    report_id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Tipo de reporte: 'daily', 'weekly', 'monthly', 'custom'
    report_type TEXT NOT NULL,
    
    -- Fecha y hora de inicio del período cubierto por el reporte
    period_start TIMESTAMP NOT NULL,
    
    -- Fecha y hora de fin del período cubierto por el reporte
    period_end TIMESTAMP NOT NULL,
    
    -- Momento en que se generó el reporte
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Usuario que generó el reporte
    generated_by TEXT,
    
    -- Ruta del archivo PDF generado
    file_path TEXT NOT NULL,
    
    -- ID del empleado (NULL = reporte general de todos los empleados)
    employee_id TEXT,
    FOREIGN KEY (employee_id) REFERENCES employees(employee_id) ON DELETE SET NULL,
    
    -- Departamento (NULL = todos los departamentos)
    department TEXT,
    
    -- Formato del reporte: 'pdf', 'csv', 'json'
    format TEXT DEFAULT 'pdf',
    
    -- Tamaño del archivo en bytes
    file_size INTEGER,
    
    -- Resumen ejecutivo en JSON
    -- Ej: {"total_detections": 1500, "avg_stress": 0.35, "alerts": 5}
    summary TEXT,
    
    -- Estado del reporte: 'generated', 'sent', 'archived', 'deleted'
    status TEXT DEFAULT 'generated',
    
    -- Constraint: period_end debe ser posterior a period_start
    CHECK (period_end >= period_start),
    
    -- Constraint: tipos de reporte válidos
    CHECK (report_type IN ('daily', 'weekly', 'monthly', 'custom')),
    
    -- Constraint: formatos válidos
    CHECK (format IN ('pdf', 'csv', 'json', 'xlsx'))
);

-- Índice para búsquedas por tipo de reporte
CREATE INDEX IF NOT EXISTS idx_reports_type 
ON reports(report_type);

-- Índice para búsquedas por fecha de generación
CREATE INDEX IF NOT EXISTS idx_reports_generated 
ON reports(generated_at);

-- Índice para búsquedas por empleado
CREATE INDEX IF NOT EXISTS idx_reports_employee 
ON reports(employee_id);

-- ============================================================================
-- TABLA: audit_log
-- ============================================================================
-- Descripción: Registro de auditoría de todas las acciones importantes
-- realizadas en el sistema. Esencial para cumplimiento legal y seguridad.
-- ============================================================================

CREATE TABLE IF NOT EXISTS audit_log (
    -- ID único del log (autoincremental)
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Momento exacto del evento
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Tipo de evento:
    -- 'employee_enrolled', 'employee_deleted', 'detection_performed',
    -- 'alert_generated', 'report_generated', 'data_accessed', 
    -- 'settings_changed', 'login', 'logout'
    event_type TEXT NOT NULL,
    
    -- Usuario que realizó la acción (puede ser 'system' para acciones automáticas)
    user_id TEXT NOT NULL,
    
    -- Empleado relacionado con la acción (si aplica)
    employee_id TEXT,
    FOREIGN KEY (employee_id) REFERENCES employees(employee_id) ON DELETE SET NULL,
    
    -- Descripción detallada de la acción
    action TEXT NOT NULL,
    
    -- Dirección IP desde donde se realizó la acción
    ip_address TEXT,
    
    -- Detalles adicionales en formato JSON
    -- Ej: {"previous_value": "A", "new_value": "B"}
    details TEXT,
    
    -- Resultado de la acción: 'success', 'failure', 'partial'
    result TEXT DEFAULT 'success',
    
    -- Mensaje de error (si hubo fallo)
    error_message TEXT,
    
    -- Nivel de severidad: 'info', 'warning', 'error', 'critical'
    severity TEXT DEFAULT 'info',
    
    -- Constraint: resultados válidos
    CHECK (result IN ('success', 'failure', 'partial')),
    
    -- Constraint: niveles de severidad válidos
    CHECK (severity IN ('info', 'warning', 'error', 'critical'))
);

-- Índice para búsquedas por tipo de evento
CREATE INDEX IF NOT EXISTS idx_audit_event_type 
ON audit_log(event_type);

-- Índice para búsquedas por timestamp
CREATE INDEX IF NOT EXISTS idx_audit_timestamp 
ON audit_log(timestamp);

-- Índice para búsquedas por usuario
CREATE INDEX IF NOT EXISTS idx_audit_user 
ON audit_log(user_id);

-- Índice para búsquedas por empleado
CREATE INDEX IF NOT EXISTS idx_audit_employee 
ON audit_log(employee_id);

-- ============================================================================
-- TABLA: system_config
-- ============================================================================
-- Descripción: Almacena configuraciones del sistema en formato clave-valor.
-- Permite modificar parámetros sin necesidad de cambiar código.
-- ============================================================================

CREATE TABLE IF NOT EXISTS system_config (
    -- Clave de configuración (PRIMARY KEY)
    config_key TEXT PRIMARY KEY NOT NULL,
    
    -- Valor de la configuración (puede ser cualquier tipo en formato texto)
    config_value TEXT NOT NULL,
    
    -- Tipo de dato: 'string', 'integer', 'float', 'boolean', 'json'
    value_type TEXT DEFAULT 'string',
    
    -- Descripción de qué hace esta configuración
    description TEXT,
    
    -- Categoría de la configuración: 'detection', 'alerts', 'reports', 'system'
    category TEXT DEFAULT 'system',
    
    -- Indica si la configuración puede ser modificada por usuarios
    is_editable BOOLEAN DEFAULT 1,
    
    -- Última modificación
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Usuario que hizo la última modificación
    updated_by TEXT,
    
    -- Constraint: tipos de valor válidos
    CHECK (value_type IN ('string', 'integer', 'float', 'boolean', 'json'))
);

-- Índice para búsquedas por categoría
CREATE INDEX IF NOT EXISTS idx_config_category 
ON system_config(category);

-- ============================================================================
-- TRIGGERS
-- ============================================================================
-- Los triggers son procedimientos que se ejecutan automáticamente en
-- respuesta a ciertos eventos en las tablas.
-- ============================================================================

-- Trigger: Actualizar last_updated en employees al modificar
CREATE TRIGGER IF NOT EXISTS update_employee_timestamp
AFTER UPDATE ON employees
FOR EACH ROW
BEGIN
    UPDATE employees 
    SET last_updated = CURRENT_TIMESTAMP 
    WHERE employee_id = NEW.employee_id;
END;

-- Trigger: Calcular duration_seconds al cerrar una sesión
CREATE TRIGGER IF NOT EXISTS calculate_session_duration
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

-- Trigger: Registrar en audit_log cuando se registra un nuevo empleado
CREATE TRIGGER IF NOT EXISTS audit_employee_enrollment
AFTER INSERT ON employees
FOR EACH ROW
BEGIN
    INSERT INTO audit_log (event_type, user_id, employee_id, action, severity)
    VALUES ('employee_enrolled', 'system', NEW.employee_id, 
            'Nuevo empleado registrado: ' || NEW.name, 'info');
END;

-- Trigger: Registrar en audit_log cuando se elimina un empleado
CREATE TRIGGER IF NOT EXISTS audit_employee_deletion
AFTER DELETE ON employees
FOR EACH ROW
BEGIN
    INSERT INTO audit_log (event_type, user_id, employee_id, action, severity)
    VALUES ('employee_deleted', 'system', OLD.employee_id, 
            'Empleado eliminado: ' || OLD.name, 'warning');
END;

-- Trigger: Registrar en audit_log cuando se genera una alerta crítica
CREATE TRIGGER IF NOT EXISTS audit_critical_alert
AFTER INSERT ON stress_alerts
FOR EACH ROW
WHEN NEW.alert_type = 'critical'
BEGIN
    INSERT INTO audit_log (event_type, user_id, employee_id, action, severity)
    VALUES ('alert_generated', 'system', NEW.employee_id, 
            'Alerta crítica generada - Nivel de estrés: ' || NEW.stress_level, 'critical');
END;

-- ============================================================================
-- VISTAS (VIEWS)
-- ============================================================================
-- Las vistas son consultas predefinidas que se pueden usar como tablas.
-- ============================================================================

-- Vista: Resumen actual de empleados activos
CREATE VIEW IF NOT EXISTS v_active_employees AS
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

-- Vista: Alertas pendientes con información del empleado
CREATE VIEW IF NOT EXISTS v_pending_alerts AS
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

-- Vista: Estadísticas de detecciones por día
CREATE VIEW IF NOT EXISTS v_daily_detection_stats AS
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
-- DATOS INICIALES DE CONFIGURACIÓN
-- ============================================================================

-- Configuraciones predeterminadas del sistema
INSERT OR IGNORE INTO system_config (config_key, config_value, value_type, description, category) VALUES
-- Umbrales de detección
('stress_threshold_warning', '0.65', 'float', 'Umbral de estrés para generar advertencia', 'alerts'),
('stress_threshold_critical', '0.85', 'float', 'Umbral de estrés para generar alerta crítica', 'alerts'),
('min_confidence_threshold', '0.60', 'float', 'Confianza mínima para aceptar una detección', 'detection'),

-- Configuración de alertas
('alert_cooldown_minutes', '30', 'integer', 'Tiempo mínimo entre alertas del mismo empleado', 'alerts'),
('consecutive_detections_alert', '5', 'integer', 'Detecciones consecutivas de alto estrés para alerta', 'alerts'),

-- Retención de datos
('data_retention_days', '180', 'integer', 'Días de retención de detecciones (6 meses)', 'system'),
('log_retention_days', '365', 'integer', 'Días de retención de logs de auditoría (1 año)', 'system'),

-- Configuración de reportes
('report_generation_time', '02:00', 'string', 'Hora de generación automática de reportes', 'reports'),
('auto_generate_daily_reports', 'true', 'boolean', 'Generar reportes diarios automáticamente', 'reports'),

-- Sistema
('system_version', '1.0.0', 'string', 'Versión del sistema StressVision', 'system'),
('model_version', '1.0.0', 'string', 'Versión del modelo de detección de emociones', 'system'),
('maintenance_mode', 'false', 'boolean', 'Modo de mantenimiento del sistema', 'system');

-- ============================================================================
-- FIN DEL ESQUEMA
-- ============================================================================

-- Verificar integridad de claves foráneas
PRAGMA foreign_key_check;

-- Analizar las tablas para optimizar consultas
ANALYZE;


