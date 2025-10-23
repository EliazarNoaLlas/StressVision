"""
Sistema de Base de Datos para Stress Vision
Inicialización y creación de esquema SQLite adaptado desde PostgreSQL

Autor: Gloria S.A.
Fecha: 2024
"""

import sqlite3
import os
from datetime import datetime


class DatabaseInitializer:
    def __init__(self, db_path="gloria_stress_system.db"):
        """
        Inicializa el sistema de base de datos.
        
        Args:
            db_path: Ruta al archivo de base de datos SQLite
        """
        self.db_path = db_path
        
    def create_database(self):
        """Crea todas las tablas del esquema de base de datos."""
        
        # Verificar si la base de datos ya existe
        db_exists = os.path.exists(self.db_path)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        print(f"{'🔄 Recreando' if db_exists else '🆕 Creando'} base de datos: {self.db_path}")
        
        # ========================================
        # TABLA: employees
        # Registro de personas
        # ========================================
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_code TEXT UNIQUE NOT NULL,
            full_name TEXT NOT NULL,
            department TEXT,
            shift TEXT, -- 'morning', 'afternoon', 'night'
            
            -- Face embedding (vector de 128 dimensiones guardado como JSON)
            face_embedding TEXT, -- JSON array de 128 floats
            face_encoding_quality REAL, -- confianza del encoding (0-1)
            
            -- Metadata
            enrollment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            consent_given INTEGER DEFAULT 0, -- 0=False, 1=True
            consent_date TIMESTAMP,
            thumbnail_base64 TEXT, -- imagen pequeña para UI
            
            -- Control
            is_active INTEGER DEFAULT 1, -- 0=False, 1=True
            last_seen TIMESTAMP,
            
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_employee_code ON employees(employee_code)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_is_active ON employees(is_active)")
        
        # ========================================
        # TABLA: sessions
        # Sesiones de monitoreo por dispositivo
        # ========================================
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id TEXT NOT NULL, -- 'pi-01', 'pi-02', 'webcam-01'
            location TEXT, -- 'Planta A - Línea 3'
            start_timestamp TIMESTAMP NOT NULL,
            end_timestamp TIMESTAMP,
            fps_avg REAL,
            frames_processed INTEGER DEFAULT 0,
            status TEXT DEFAULT 'active', -- 'active', 'paused', 'ended'
            
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_device ON sessions(device_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_time ON sessions(start_timestamp, end_timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_status ON sessions(status)")
        
        # ========================================
        # TABLA: detection_events
        # Eventos individuales de detección
        # ========================================
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS detection_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            timestamp TIMESTAMP NOT NULL,
            
            -- Identificación
            device_id TEXT NOT NULL,
            track_id TEXT, -- ID temporal del tracker
            employee_id INTEGER, -- NULL si no reconocido
            recognition_confidence REAL, -- 0-1
            
            -- Detección emocional
            emotion TEXT NOT NULL, -- 'neutral','stress','angry','sad','fear','happy','surprise'
            emotion_confidence REAL NOT NULL, -- 0-1
            emotion_vector TEXT, -- JSON: {neutral: 0.1, stress: 0.7, ...}
            
            -- Geometría facial (guardado como JSON)
            bounding_box TEXT, -- JSON: {x, y, w, h}
            landmarks TEXT, -- JSON: puntos clave faciales
            
            -- Metadata
            frame_quality REAL, -- blur, iluminación, etc.
            processing_time_ms REAL,
            
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            FOREIGN KEY (session_id) REFERENCES sessions(id),
            FOREIGN KEY (employee_id) REFERENCES employees(id)
        )
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_detection_timestamp ON detection_events(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_detection_employee ON detection_events(employee_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_detection_emotion ON detection_events(emotion)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_detection_device ON detection_events(device_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_detection_session ON detection_events(session_id)")
        
        # ========================================
        # TABLA: employee_stress_summary
        # Agregaciones por empleado por período
        # ========================================
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS employee_stress_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER,
            period_start TIMESTAMP NOT NULL,
            period_end TIMESTAMP NOT NULL,
            
            -- Contadores por emoción
            total_detections INTEGER,
            neutral_count INTEGER,
            stress_count INTEGER,
            angry_count INTEGER,
            sad_count INTEGER,
            fear_count INTEGER,
            happy_count INTEGER,
            surprise_count INTEGER,
            fatigue_count INTEGER,
            
            -- Métricas
            stress_percentage REAL, -- % de tiempo en estrés
            avg_confidence REAL,
            max_consecutive_stress_minutes INTEGER,
            
            -- Correlación con productividad
            productivity_score REAL,
            attendance_rate REAL,
            
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            FOREIGN KEY (employee_id) REFERENCES employees(id)
        )
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_summary_employee_period ON employee_stress_summary(employee_id, period_start)")
        
        # ========================================
        # TABLA: reports_15min
        # Reportes automáticos cada 15 minutos
        # ========================================
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS reports_15min (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_timestamp TIMESTAMP NOT NULL,
            end_timestamp TIMESTAMP NOT NULL,
            device_id TEXT,
            
            -- Resumen general
            total_detections INTEGER,
            total_employees_detected INTEGER,
            overall_stress_percentage REAL,
            
            -- Resumen por empleado (JSON)
            per_employee_summary TEXT, -- JSON
            -- Ejemplo: {"5": {"stress_pct": 0.42, "counts": {...}}, ...}
            
            -- Alertas generadas
            alerts_triggered INTEGER,
            alert_details TEXT, -- JSON
            
            generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_reports_time ON reports_15min(start_timestamp, end_timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_reports_device ON reports_15min(device_id)")
        
        # ========================================
        # TABLA: alerts
        # Alertas disparadas
        # ========================================
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP NOT NULL,
            employee_id INTEGER,
            alert_type TEXT, -- 'high_stress_prolonged', 'repeated_anger', 'fatigue_detected'
            severity TEXT, -- 'low', 'medium', 'high'
            description TEXT,
            
            -- Estado
            status TEXT DEFAULT 'pending', -- 'pending','acknowledged','resolved'
            acknowledged_by TEXT,
            acknowledged_at TIMESTAMP,
            resolution_notes TEXT,
            
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            FOREIGN KEY (employee_id) REFERENCES employees(id)
        )
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_employee ON alerts(employee_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)")
        
        # ========================================
        # TABLA: audit_log
        # Registro de accesos y acciones
        # ========================================
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            user_id TEXT, -- usuario que realizó la acción
            action TEXT, -- 'view_dashboard', 'export_report', 'access_employee_data'
            entity_type TEXT, -- 'employee', 'report', 'alert'
            entity_id INTEGER,
            ip_address TEXT,
            details TEXT -- JSON
        )
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action)")
        
        # ========================================
        # TABLA: notification_config
        # Configuración de notificaciones
        # ========================================
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS notification_config (
            id INTEGER PRIMARY KEY,
            smtp_server TEXT,
            smtp_port INTEGER,
            sender_email TEXT,
            sender_password TEXT,
            recipients TEXT, -- Lista separada por comas
            
            -- Configuración de alertas
            enable_email_alerts INTEGER DEFAULT 1, -- 0=False, 1=True
            stress_threshold REAL DEFAULT 40.0,
            alert_cooldown_minutes INTEGER DEFAULT 30,
            
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Insertar configuración por defecto si no existe
        cursor.execute("""
        INSERT OR IGNORE INTO notification_config (id, stress_threshold, alert_cooldown_minutes)
        VALUES (1, 40.0, 30)
        """)
        
        conn.commit()
        
        # Mostrar resumen
        print("\n✅ Base de datos creada exitosamente")
        print("\n📊 Tablas creadas:")
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        tables = cursor.fetchall()
        
        for i, (table_name,) in enumerate(tables, 1):
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"  {i}. {table_name:30s} ({count} registros)")
        
        conn.close()
        
        return True
    
    def verify_database(self):
        """Verifica la integridad de la base de datos."""
        global conn
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Verificar integridad
            cursor.execute("PRAGMA integrity_check")
            result = cursor.fetchone()
            
            if result[0] == 'ok':
                print("\n✅ Verificación de integridad: OK")
                return True
            else:
                print(f"\n❌ Error de integridad: {result[0]}")
                return False
                
        except Exception as e:
            print(f"\n❌ Error al verificar base de datos: {e}")
            return False
        finally:
            conn.close()
    
    def show_schema(self):
        """Muestra el esquema de todas las tablas."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        tables = cursor.fetchall()
        
        print("\n" + "="*80)
        print("ESQUEMA DE BASE DE DATOS")
        print("="*80)
        
        for (table_name,) in tables:
            print(f"\n📋 Tabla: {table_name}")
            print("-" * 80)
            
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            print(f"{'#':<3} {'Campo':<30} {'Tipo':<15} {'Not Null':<10} {'Default':<15}")
            print("-" * 80)
            
            for col in columns:
                cid, name, type_, notnull, default, pk = col
                print(f"{cid:<3} {name:<30} {type_:<15} {bool(notnull)!s:<10} {str(default):<15}")
        
        print("\n" + "="*80)
        
        conn.close()
    
    def add_sample_data(self):
        """Agrega datos de ejemplo para pruebas."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Agregar sesión de ejemplo
            cursor.execute("""
            INSERT INTO sessions (device_id, location, start_timestamp, status)
            VALUES ('webcam-01', 'Oficina Principal', ?, 'active')
            """, (datetime.now(),))
            
            print("\n✅ Datos de ejemplo agregados")
            
            conn.commit()
        except Exception as e:
            print(f"\n⚠️  Error al agregar datos de ejemplo: {e}")
            conn.rollback()
        finally:
            conn.close()


def main():
    """Función principal para inicializar la base de datos."""
    print("\n" + "="*80)
    print("🗄️  INICIALIZADOR DE BASE DE DATOS - STRESS VISION")
    print("="*80)
    print("\nGloria S.A. - Sistema de Monitoreo de Estrés Laboral")
    print("Adaptado de PostgreSQL a SQLite para despliegue local")
    print("\n" + "="*80)
    
    db = DatabaseInitializer("../gloria_stress_system.db")
    
    # Crear base de datos
    success = db.create_database()
    
    if success:
        # Verificar integridad
        db.verify_database()
        
        # Mostrar esquema
        show_schema = input("\n¿Desea ver el esquema completo de la base de datos? (s/n): ")
        if show_schema.lower() == 's':
            db.show_schema()
        
        # Agregar datos de ejemplo
        add_sample = input("\n¿Desea agregar datos de ejemplo? (s/n): ")
        if add_sample.lower() == 's':
            db.add_sample_data()
        
        print("\n" + "="*80)
        print("✅ INICIALIZACIÓN COMPLETADA")
        print("="*80)
        print(f"\n📁 Base de datos creada en: {os.path.abspath(db.db_path)}")
        print(f"📊 Tamaño: {os.path.getsize(db.db_path) / 1024:.2f} KB")
        print("\n💡 Próximos pasos:")
        print("   1. Ejecutar: python enrollment.py (para registrar empleados)")
        print("   2. Ejecutar: python load_enrollments.py (para cargar embeddings)")
        print("   3. Ejecutar: streamlit run main.py (para iniciar el sistema)")
        print("\n" + "="*80)
    else:
        print("\n❌ Error al crear la base de datos")


if __name__ == "__main__":
    main()


