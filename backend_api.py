"""
Backend API Principal - Stress Vision
FastAPI backend con endpoints REST y WebSocket para sistema de detección de estrés

ADAPTADO PARA:
- SQLite (en lugar de PostgreSQL)
- Sin Celery/Redis (usa APScheduler)
- WebSocket simple (sin Socket.IO complejo)
- Funciona 100% local

Autor: Gloria S.A.
Fecha: 2024
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import sqlite3
import json
import asyncio
from contextlib import contextmanager
import threading


# ============== CONFIGURACIÓN ==============

app = FastAPI(
    title="Stress Detection API",
    description="Backend para detección de estrés laboral en tiempo real - Gloria S.A.",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción: especificar dominios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración
DB_PATH = 'gloria_stress_system.db'

# Estado en memoria
active_devices = {}  # {device_id: {session_id, last_seen, status}}
live_detections = {}  # {employee_id: último estado}
websocket_clients = []  # Lista de clientes WebSocket conectados


# ============== DATABASE HELPERS ==============

@contextmanager
def get_db():
    """Context manager para conexión a base de datos."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Permite acceso por nombre de columna
    try:
        yield conn
    finally:
        conn.close()


def dict_from_row(row):
    """Convierte sqlite3.Row a dict."""
    if row is None:
        return None
    return {key: row[key] for key in row.keys()}


# ============== WEBSOCKET MANAGER ==============

class ConnectionManager:
    """Maneja conexiones WebSocket."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"✓ WebSocket conectado. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"✗ WebSocket desconectado. Total: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        """Envía mensaje a todos los clientes conectados."""
        dead_connections = []
        
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                dead_connections.append(connection)
        
        # Limpiar conexiones muertas
        for connection in dead_connections:
            self.disconnect(connection)


manager = ConnectionManager()


# ============== STARTUP/SHUTDOWN ==============

@app.on_event("startup")
async def startup():
    """Inicializar al arrancar."""
    print("\n" + "="*80)
    print("🚀 BACKEND API - STRESS VISION")
    print("="*80)
    print("\nGloria S.A. - Sistema de Detección de Estrés Laboral")
    print("\n" + "="*80)
    
    # Verificar base de datos
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM employees WHERE is_active = 1")
        emp_count = cursor.fetchone()[0]
    
    print(f"\n✅ Backend inicializado")
    print(f"   • Base de datos: {DB_PATH}")
    print(f"   • Empleados activos: {emp_count}")
    print(f"   • API Docs: http://localhost:8000/api/docs")
    print(f"   • WebSocket: ws://localhost:8000/ws")
    print("\n" + "="*80 + "\n")


@app.on_event("shutdown")
async def shutdown():
    """Cerrar al apagar."""
    print("\n✓ Backend detenido correctamente")


# ============== REST API ENDPOINTS ==============

@app.get("/")
async def root():
    """Health check."""
    return {
        "status": "ok",
        "service": "Gloria Stress Detection API",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "database": DB_PATH,
        "active_devices": len(active_devices),
        "websocket_clients": len(manager.active_connections)
    }


@app.get("/health")
async def health():
    """Health check simple."""
    return {"status": "ok"}


# ========== EMPLOYEES ==========

@app.get("/api/employees")
async def get_employees(active_only: bool = True):
    """Obtener lista de empleados."""
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        if active_only:
            cursor.execute("""
                SELECT id, employee_code, full_name, department, 
                       shift, is_active, last_seen, face_encoding_quality
                FROM employees
                WHERE is_active = 1 AND consent_given = 1
                ORDER BY full_name
            """)
        else:
            cursor.execute("""
                SELECT id, employee_code, full_name, department, 
                       shift, is_active, last_seen, face_encoding_quality
                FROM employees
                WHERE consent_given = 1
                ORDER BY full_name
            """)
        
        employees = [dict_from_row(row) for row in cursor.fetchall()]
    
    return {"employees": employees, "total": len(employees)}


@app.get("/api/employees/{employee_id}")
async def get_employee(employee_id: int):
    """Obtener información de un empleado específico."""
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, employee_code, full_name, department, 
                   shift, enrollment_date, is_active, last_seen,
                   face_encoding_quality
            FROM employees
            WHERE id = ?
        """, (employee_id,))
        
        employee = dict_from_row(cursor.fetchone())
    
    if not employee:
        raise HTTPException(status_code=404, detail="Empleado no encontrado")
    
    return employee


@app.get("/api/employees/{employee_id}/status")
async def get_employee_status(employee_id: int, minutes: int = 15):
    """Obtener estado en vivo y reciente de un empleado."""
    
    # Estado en memoria
    live_status = live_detections.get(employee_id)
    
    # Historial reciente
    cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Detecciones recientes
        cursor.execute("""
            SELECT timestamp, emotion, emotion_confidence, device_id
            FROM detection_events
            WHERE employee_id = ?
              AND timestamp > ?
            ORDER BY timestamp DESC
            LIMIT 100
        """, (employee_id, cutoff_time.isoformat()))
        
        detections = [dict_from_row(row) for row in cursor.fetchall()]
    
    # Calcular métricas
    emotion_counts = {}
    for det in detections:
        em = det['emotion']
        emotion_counts[em] = emotion_counts.get(em, 0) + 1
    
    total = len(detections)
    stress_related = emotion_counts.get('stress', 0) + emotion_counts.get('sad', 0)
    stress_pct = (stress_related / total * 100) if total > 0 else 0
    
    return {
        "employee_id": employee_id,
        "live_status": live_status,
        "recent_detections": detections,
        "summary": {
            "total_detections": total,
            "emotion_counts": emotion_counts,
            "stress_percentage": round(stress_pct, 2),
            "period_minutes": minutes
        }
    }


# ========== SESSIONS ==========

@app.post("/api/sessions")
async def create_session(device_id: str, location: str = "Unknown"):
    """Crear nueva sesión de monitoreo."""
    
    start_time = datetime.utcnow()
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO sessions (device_id, location, start_timestamp, status)
            VALUES (?, ?, ?, 'active')
        """, (device_id, location, start_time.isoformat()))
        
        session_id = cursor.lastrowid
        conn.commit()
    
    # Actualizar dispositivos activos
    active_devices[device_id] = {
        'session_id': session_id,
        'last_seen': start_time,
        'status': 'active',
        'location': location
    }
    
    print(f"📝 Nueva sesión: {session_id} - {device_id} ({location})")
    
    return {"session_id": session_id, "device_id": device_id}


@app.post("/api/sessions/{session_id}/end")
async def end_session(session_id: int):
    """Finalizar sesión de monitoreo."""
    
    end_time = datetime.utcnow()
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE sessions
            SET end_timestamp = ?, status = 'ended'
            WHERE id = ?
        """, (end_time.isoformat(), session_id))
        
        conn.commit()
    
    print(f"🏁 Sesión finalizada: {session_id}")
    
    return {"status": "ok", "session_id": session_id}


# ========== DETECTIONS ==========

@app.post("/api/detections")
async def receive_detection(detection: dict):
    """Recibir detección desde edge device."""
    
    # Validar campos requeridos
    required_fields = ['device_id', 'timestamp', 'emotion', 'emotion_confidence']
    for field in required_fields:
        if field not in detection:
            raise HTTPException(status_code=400, detail=f"Campo requerido: {field}")
    
    # Guardar en base de datos
    with get_db() as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO detection_events (
                session_id, timestamp, device_id, track_id,
                employee_id, recognition_confidence,
                emotion, emotion_confidence, emotion_vector,
                bounding_box, processing_time_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            detection.get('session_id'),
            detection['timestamp'],
            detection['device_id'],
            detection.get('track_id'),
            detection.get('employee_id'),
            detection.get('recognition_confidence', 0.0),
            detection['emotion'],
            detection['emotion_confidence'],
            json.dumps(detection.get('emotion_vector', {})),
            json.dumps(detection.get('bounding_box', {})),
            detection.get('processing_time_ms', 0.0)
        ))
        
        detection_id = cursor.lastrowid
        conn.commit()
    
    # Actualizar estado en vivo
    employee_id = detection.get('employee_id')
    if employee_id:
        live_detections[employee_id] = {
            'last_emotion': detection['emotion'],
            'confidence': detection['emotion_confidence'],
            'timestamp': detection['timestamp'],
            'device_id': detection['device_id']
        }
    
    # Actualizar dispositivo
    device_id = detection['device_id']
    if device_id in active_devices:
        active_devices[device_id]['last_seen'] = datetime.utcnow()
    
    # Broadcast via WebSocket
    await manager.broadcast({
        'type': 'detection',
        'data': detection
    })
    
    # Verificar si genera alerta
    alert_created = await check_and_create_alert(detection)
    
    return {
        "status": "ok",
        "detection_id": detection_id,
        "alert_created": alert_created
    }


async def check_and_create_alert(detection: dict) -> bool:
    """Verifica si debe crear alerta basado en detección."""
    
    employee_id = detection.get('employee_id')
    emotion = detection.get('emotion')
    confidence = detection.get('emotion_confidence', 0.0)
    
    # Solo alertar para emociones negativas
    if not employee_id or emotion not in ['stress', 'sad']:
        return False
    
    # Verificar historial reciente (últimos 15 minutos)
    cutoff = datetime.utcnow() - timedelta(minutes=15)
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Contar detecciones de estrés recientes
        cursor.execute("""
            SELECT COUNT(*) as count
            FROM detection_events
            WHERE employee_id = ?
              AND emotion IN ('stress', 'sad')
              AND timestamp > ?
        """, (employee_id, cutoff.isoformat()))
        
        stress_count = cursor.fetchone()[0]
        
        # Si tiene más de 10 detecciones de estrés en 15 min → alerta
        if stress_count >= 10:
            # Verificar si ya existe alerta activa reciente
            cursor.execute("""
                SELECT id FROM alerts
                WHERE employee_id = ?
                  AND status = 'pending'
                  AND timestamp > ?
            """, (employee_id, (datetime.utcnow() - timedelta(hours=1)).isoformat()))
            
            existing_alert = cursor.fetchone()
            
            if not existing_alert:
                # Obtener info del empleado
                cursor.execute("""
                    SELECT full_name, employee_code
                    FROM employees
                    WHERE id = ?
                """, (employee_id,))
                
                emp_info = dict_from_row(cursor.fetchone())
                
                # Crear alerta
                cursor.execute("""
                    INSERT INTO alerts (
                        timestamp, employee_id, alert_type,
                        severity, description, status
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    datetime.utcnow().isoformat(),
                    employee_id,
                    'high_stress_prolonged',
                    'high' if confidence > 0.8 else 'medium',
                    f"Estrés prolongado detectado: {stress_count} eventos en 15 minutos",
                    'pending'
                ))
                
                alert_id = cursor.lastrowid
                conn.commit()
                
                # Broadcast alerta via WebSocket
                alert_data = {
                    'type': 'alert',
                    'alert_id': alert_id,
                    'timestamp': datetime.utcnow().isoformat(),
                    'employee_id': employee_id,
                    'employee_name': emp_info['full_name'] if emp_info else 'Unknown',
                    'employee_code': emp_info['employee_code'] if emp_info else 'N/A',
                    'alert_type': 'high_stress_prolonged',
                    'severity': 'high' if confidence > 0.8 else 'medium',
                    'description': f"Estrés prolongado: {stress_count} eventos",
                    'stress_count': stress_count
                }
                
                await manager.broadcast(alert_data)
                
                print(f"⚠️  ALERTA creada: #{alert_id} - Employee {employee_id} ({stress_count} eventos)")
                
                return True
    
    return False


# ========== ALERTS ==========

@app.get("/api/alerts")
async def get_alerts(
    status: Optional[str] = 'pending',
    limit: int = 50,
    employee_id: Optional[int] = None
):
    """Obtener alertas."""
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        query = """
            SELECT a.*, e.full_name, e.employee_code, e.department
            FROM alerts a
            JOIN employees e ON a.employee_id = e.id
            WHERE 1=1
        """
        params = []
        
        if status:
            query += " AND a.status = ?"
            params.append(status)
        
        if employee_id:
            query += " AND a.employee_id = ?"
            params.append(employee_id)
        
        query += " ORDER BY a.timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        alerts = [dict_from_row(row) for row in cursor.fetchall()]
    
    return {"alerts": alerts, "total": len(alerts)}


@app.post("/api/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: int, user_id: str):
    """Marcar alerta como reconocida."""
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE alerts
            SET status = 'acknowledged',
                acknowledged_by = ?,
                acknowledged_at = ?
            WHERE id = ?
        """, (user_id, datetime.utcnow().isoformat(), alert_id))
        
        conn.commit()
    
    # Broadcast actualización
    await manager.broadcast({
        'type': 'alert_acknowledged',
        'alert_id': alert_id,
        'user_id': user_id
    })
    
    return {"status": "ok", "alert_id": alert_id}


@app.post("/api/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: int, user_id: str, notes: str = ""):
    """Resolver alerta."""
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE alerts
            SET status = 'resolved',
                acknowledged_by = ?,
                acknowledged_at = ?,
                resolution_notes = ?
            WHERE id = ?
        """, (user_id, datetime.utcnow().isoformat(), notes, alert_id))
        
        conn.commit()
    
    await manager.broadcast({
        'type': 'alert_resolved',
        'alert_id': alert_id
    })
    
    return {"status": "ok", "alert_id": alert_id}


# ========== DASHBOARD ==========

@app.get("/api/dashboard/overview")
async def get_dashboard_overview():
    """Resumen general para dashboard principal."""
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Total empleados activos
        cursor.execute("""
            SELECT COUNT(*) FROM employees
            WHERE is_active = 1 AND consent_given = 1
        """)
        total_employees = cursor.fetchone()[0]
        
        # Detecciones última hora
        cutoff = (datetime.utcnow() - timedelta(hours=1)).isoformat()
        cursor.execute("""
            SELECT COUNT(*) FROM detection_events
            WHERE timestamp > ?
        """, (cutoff,))
        detections_last_hour = cursor.fetchone()[0]
        
        # Alertas pendientes
        cursor.execute("""
            SELECT COUNT(*) FROM alerts
            WHERE status = 'pending'
        """)
        pending_alerts = cursor.fetchone()[0]
        
        # Nivel de estrés general (última hora)
        cursor.execute("""
            SELECT 
                COUNT(*) FILTER (WHERE emotion IN ('stress', 'sad')) as stress_count,
                COUNT(*) as total_count
            FROM detection_events
            WHERE timestamp > ?
        """, (cutoff,))
        
        row = cursor.fetchone()
        stress_count = row[0] if row[0] else 0
        total_count = row[1] if row[1] else 0
        
        stress_pct = (stress_count / total_count * 100) if total_count > 0 else 0
        
        # Dispositivos activos
        active_device_count = len([d for d in active_devices.values() 
                                   if d.get('status') == 'active'])
        
        # Empleados detectados en última hora
        cursor.execute("""
            SELECT COUNT(DISTINCT employee_id) FROM detection_events
            WHERE timestamp > ? AND employee_id IS NOT NULL
        """, (cutoff,))
        employees_detected = cursor.fetchone()[0]
    
    return {
        "total_employees": total_employees,
        "detections_last_hour": detections_last_hour,
        "employees_detected_last_hour": employees_detected,
        "pending_alerts": pending_alerts,
        "overall_stress_percentage": round(stress_pct, 2),
        "active_devices": active_device_count,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/api/dashboard/stats")
async def get_dashboard_stats(hours: int = 24):
    """Estadísticas detalladas para dashboard."""
    
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Detecciones por emoción
        cursor.execute("""
            SELECT emotion, COUNT(*) as count
            FROM detection_events
            WHERE timestamp > ?
            GROUP BY emotion
            ORDER BY count DESC
        """, (cutoff,))
        
        emotions_by_count = [dict_from_row(row) for row in cursor.fetchall()]
        
        # Detecciones por hora
        cursor.execute("""
            SELECT 
                strftime('%Y-%m-%d %H:00:00', timestamp) as hour,
                COUNT(*) as count
            FROM detection_events
            WHERE timestamp > ?
            GROUP BY hour
            ORDER BY hour
        """, (cutoff,))
        
        detections_by_hour = [dict_from_row(row) for row in cursor.fetchall()]
        
        # Top empleados con más detecciones
        cursor.execute("""
            SELECT 
                e.id,
                e.full_name,
                e.employee_code,
                COUNT(d.id) as detection_count
            FROM employees e
            JOIN detection_events d ON e.id = d.employee_id
            WHERE d.timestamp > ?
            GROUP BY e.id
            ORDER BY detection_count DESC
            LIMIT 10
        """, (cutoff,))
        
        top_employees = [dict_from_row(row) for row in cursor.fetchall()]
    
    return {
        "period_hours": hours,
        "emotions_distribution": emotions_by_count,
        "detections_by_hour": detections_by_hour,
        "top_employees": top_employees
    }


# ========== REPORTS ==========

@app.get("/api/reports")
async def get_reports(limit: int = 20):
    """Obtener reportes de 15 minutos."""
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, start_timestamp, end_timestamp,
                   total_detections, total_employees_detected,
                   overall_stress_percentage, alerts_triggered,
                   generated_at
            FROM reports_15min
            ORDER BY start_timestamp DESC
            LIMIT ?
        """, (limit,))
        
        reports = [dict_from_row(row) for row in cursor.fetchall()]
    
    return {"reports": reports, "total": len(reports)}


@app.get("/api/reports/{report_id}")
async def get_report_detail(report_id: int):
    """Obtener detalle de un reporte específico."""
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM reports_15min
            WHERE id = ?
        """, (report_id,))
        
        report = dict_from_row(cursor.fetchone())
    
    if not report:
        raise HTTPException(status_code=404, detail="Reporte no encontrado")
    
    # Parsear JSON fields
    if report.get('per_employee_summary'):
        try:
            report['per_employee_summary'] = json.loads(report['per_employee_summary'])
        except:
            pass
    
    return report


# ========== WEBSOCKET ==========

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint para actualizaciones en tiempo real.
    
    Envía:
    - Nuevas detecciones
    - Alertas generadas
    - Reportes de 15 minutos
    - Actualizaciones de estado
    """
    
    await manager.connect(websocket)
    
    try:
        # Enviar estado inicial
        await websocket.send_json({
            'type': 'connected',
            'message': 'Conectado al servidor',
            'active_devices': len(active_devices),
            'timestamp': datetime.utcnow().isoformat()
        })
        
        # Mantener conexión viva
        while True:
            # Esperar mensajes del cliente (ping/pong)
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                
                # Echo back (opcional)
                if data:
                    message = json.loads(data)
                    if message.get('type') == 'ping':
                        await websocket.send_json({'type': 'pong'})
            
            except asyncio.TimeoutError:
                # Enviar heartbeat cada 30s
                await websocket.send_json({
                    'type': 'heartbeat',
                    'timestamp': datetime.utcnow().isoformat()
                })
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"Error en WebSocket: {e}")
        manager.disconnect(websocket)


# ========== EXPORT ==========

@app.get("/api/export/detections")
async def export_detections(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    employee_id: Optional[int] = None,
    emotion: Optional[str] = None
):
    """Exportar detecciones en formato JSON."""
    
    query = """
        SELECT 
            d.*,
            e.full_name,
            e.employee_code,
            e.department
        FROM detection_events d
        LEFT JOIN employees e ON d.employee_id = e.id
        WHERE 1=1
    """
    params = []
    
    if start_date:
        query += " AND d.timestamp >= ?"
        params.append(start_date)
    
    if end_date:
        query += " AND d.timestamp <= ?"
        params.append(end_date)
    
    if employee_id:
        query += " AND d.employee_id = ?"
        params.append(employee_id)
    
    if emotion:
        query += " AND d.emotion = ?"
        params.append(emotion)
    
    query += " ORDER BY d.timestamp DESC LIMIT 10000"
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        detections = [dict_from_row(row) for row in cursor.fetchall()]
    
    return {
        "detections": detections,
        "total": len(detections),
        "filters": {
            "start_date": start_date,
            "end_date": end_date,
            "employee_id": employee_id,
            "emotion": emotion
        }
    }


# ========== DEVICES ==========

@app.get("/api/devices")
async def get_devices():
    """Obtener lista de dispositivos activos."""
    
    devices = []
    for device_id, data in active_devices.items():
        devices.append({
            'device_id': device_id,
            'session_id': data.get('session_id'),
            'location': data.get('location'),
            'status': data.get('status'),
            'last_seen': data.get('last_seen').isoformat() if data.get('last_seen') else None
        })
    
    return {"devices": devices, "total": len(devices)}


# ============== MAIN ==============

def main():
    """Función principal para ejecutar el servidor."""
    import uvicorn
    
    print("\n" + "="*80)
    print(" "*20 + "🚀 STRESS VISION BACKEND API")
    print(" "*25 + "Gloria S.A.")
    print("="*80)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    main()





