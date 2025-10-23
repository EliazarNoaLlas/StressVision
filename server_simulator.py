"""
Servidor Simulado - Stress Vision
Simula el servidor backend que recibe detecciones del Raspberry Pi

Funciona como servidor local Flask que:
- Recibe detecciones via HTTP POST
- Guarda en base de datos SQLite
- Muestra estadísticas en tiempo real
- Simula el comportamiento del servidor real

Autor: Gloria S.A.
Fecha: 2024
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import json
from datetime import datetime
import threading
import time
from collections import defaultdict, deque


app = Flask(__name__)
CORS(app)

# Estadísticas en memoria
stats = {
    'total_detections': 0,
    'detections_by_device': defaultdict(int),
    'detections_by_emotion': defaultdict(int),
    'employees_detected': set(),
    'sessions': {},
    'recent_detections': deque(maxlen=100),
    'start_time': time.time()
}

# Lock para thread-safety
stats_lock = threading.Lock()

# Base de datos
DB_PATH = 'gloria_stress_system.db'


def init_db():
    """Verifica que la base de datos tenga las tablas necesarias."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Verificar tabla de sesiones
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='sessions'
        """)
        
        if not cursor.fetchone():
            print("⚠️  Advertencia: Tabla 'sessions' no existe")
            print("   Ejecute: python init_database.py")
        
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Error verificando base de datos: {e}")
        return False


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'server': 'Pi Simulator Server',
        'uptime_seconds': time.time() - stats['start_time']
    }), 200


@app.route('/sessions', methods=['POST'])
def create_session():
    """Crea una nueva sesión de monitoreo."""
    data = request.json
    
    device_id = data.get('device_id')
    location = data.get('location', 'Unknown')
    start_timestamp = data.get('start_timestamp', datetime.utcnow().isoformat() + 'Z')
    
    # Generar session_id
    session_id = f"session_{device_id}_{int(time.time())}"
    
    # Guardar en BD
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO sessions (device_id, location, start_timestamp, status)
            VALUES (?, ?, ?, 'active')
        """, (device_id, location, start_timestamp))
        
        conn.commit()
        session_db_id = cursor.lastrowid
        conn.close()
        
        # Guardar en memoria
        with stats_lock:
            stats['sessions'][session_id] = {
                'db_id': session_db_id,
                'device_id': device_id,
                'location': location,
                'start_timestamp': start_timestamp,
                'detections_count': 0
            }
        
        print(f"📝 Nueva sesión creada: {session_id} ({device_id})")
        
        return jsonify({
            'session_id': session_id,
            'db_id': session_db_id
        }), 200
    
    except Exception as e:
        print(f"❌ Error creando sesión: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/sessions/<session_id>/end', methods=['POST'])
def end_session(session_id):
    """Finaliza una sesión de monitoreo."""
    data = request.json
    end_timestamp = data.get('end_timestamp', datetime.utcnow().isoformat() + 'Z')
    
    # Actualizar en BD
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        if session_id in stats['sessions']:
            db_id = stats['sessions'][session_id]['db_id']
            
            cursor.execute("""
                UPDATE sessions 
                SET end_timestamp = ?, status = 'ended'
                WHERE id = ?
            """, (end_timestamp, db_id))
            
            conn.commit()
        
        conn.close()
        
        print(f"🏁 Sesión finalizada: {session_id}")
        
        return jsonify({'status': 'ok'}), 200
    
    except Exception as e:
        print(f"❌ Error finalizando sesión: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/detections', methods=['POST'])
def receive_detection():
    """Recibe una detección del Raspberry Pi."""
    data = request.json
    
    # Validar datos básicos
    required_fields = ['device_id', 'timestamp', 'emotion']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing field: {field}'}), 400
    
    # Extraer datos
    device_id = data['device_id']
    emotion = data['emotion']
    employee_id = data.get('employee_id')
    
    # Actualizar estadísticas
    with stats_lock:
        stats['total_detections'] += 1
        stats['detections_by_device'][device_id] += 1
        stats['detections_by_emotion'][emotion] += 1
        
        if employee_id:
            stats['employees_detected'].add(employee_id)
        
        stats['recent_detections'].append({
            'timestamp': data['timestamp'],
            'device_id': device_id,
            'emotion': emotion,
            'employee_id': employee_id,
            'confidence': data.get('emotion_confidence', 0.0)
        })
        
        # Actualizar contador de sesión
        session_id = data.get('session_id')
        if session_id and session_id in stats['sessions']:
            stats['sessions'][session_id]['detections_count'] += 1
    
    # Guardar en base de datos
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Obtener session_db_id si existe
        session_db_id = None
        if session_id and session_id in stats['sessions']:
            session_db_id = stats['sessions'][session_id]['db_id']
        
        # Insertar detección
        cursor.execute("""
            INSERT INTO detection_events (
                session_id, timestamp, device_id, track_id,
                employee_id, recognition_confidence,
                emotion, emotion_confidence, emotion_vector,
                bounding_box, processing_time_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_db_id,
            data['timestamp'],
            device_id,
            data.get('track_id'),
            employee_id,
            data.get('recognition_confidence', 0.0),
            emotion,
            data.get('emotion_confidence', 0.0),
            json.dumps(data.get('emotion_vector', {})),
            json.dumps(data.get('bounding_box', {})),
            data.get('processing_time_ms', 0.0)
        ))
        
        conn.commit()
        conn.close()
        
        # Log (cada 10 detecciones)
        if stats['total_detections'] % 10 == 0:
            print(f"📊 Detecciones recibidas: {stats['total_detections']}")
        
        return jsonify({'status': 'ok', 'detection_id': cursor.lastrowid}), 200
    
    except Exception as e:
        print(f"❌ Error guardando detección: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/stats', methods=['GET'])
def get_stats():
    """Retorna estadísticas actuales."""
    
    with stats_lock:
        return jsonify({
            'total_detections': stats['total_detections'],
            'detections_by_device': dict(stats['detections_by_device']),
            'detections_by_emotion': dict(stats['detections_by_emotion']),
            'unique_employees': len(stats['employees_detected']),
            'active_sessions': len([s for s in stats['sessions'].values() 
                                   if s.get('status') != 'ended']),
            'uptime_seconds': time.time() - stats['start_time'],
            'recent_detections': list(stats['recent_detections'])[-10:]
        }), 200


def print_stats_loop():
    """Loop que imprime estadísticas periódicamente."""
    
    while True:
        time.sleep(30)  # Cada 30 segundos
        
        with stats_lock:
            print("\n" + "="*80)
            print("📊 ESTADÍSTICAS DEL SERVIDOR")
            print("="*80)
            print(f"   Total detecciones: {stats['total_detections']}")
            print(f"   Empleados únicos: {len(stats['employees_detected'])}")
            print(f"   Sesiones activas: {len(stats['sessions'])}")
            
            if stats['detections_by_device']:
                print(f"\n   Por dispositivo:")
                for device_id, count in stats['detections_by_device'].items():
                    print(f"      • {device_id}: {count}")
            
            if stats['detections_by_emotion']:
                print(f"\n   Por emoción:")
                for emotion, count in sorted(stats['detections_by_emotion'].items(), 
                                            key=lambda x: x[1], reverse=True):
                    print(f"      • {emotion}: {count}")
            
            print("="*80 + "\n")


def main():
    """Función principal."""
    
    print("\n" + "="*80)
    print(" "*20 + "🖥️  SERVIDOR SIMULADO")
    print(" "*10 + "Sistema de Detección de Estrés - Gloria S.A.")
    print("="*80)
    
    # Verificar base de datos
    print("\n🔍 Verificando base de datos...")
    if not init_db():
        print("⚠️  Advertencia: La base de datos puede no estar configurada correctamente")
    else:
        print("✅ Base de datos OK")
    
    # Iniciar thread de estadísticas
    stats_thread = threading.Thread(target=print_stats_loop, daemon=True)
    stats_thread.start()
    
    # Iniciar servidor
    print("\n🚀 Iniciando servidor Flask...")
    print(f"   • URL: http://localhost:5000")
    print(f"   • Health check: http://localhost:5000/health")
    print(f"   • Estadísticas: http://localhost:5000/stats")
    print(f"\n💡 El servidor permanecerá activo. Presione Ctrl+C para detener.\n")
    print("="*80 + "\n")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n\n⏹️  Servidor detenido")


if __name__ == '__main__':
    main()





