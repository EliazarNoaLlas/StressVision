"""
Simulador de Raspberry Pi - Stress Vision
Sistema de inferencia en tiempo real simulado (funciona en PC sin Raspberry Pi)

DIFERENCIAS CON RASPBERRY PI REAL:
- Usa tensorflow.lite en lugar de tflite_runtime
- Base de datos SQLite en lugar de PostgreSQL
- Servidor local en lugar de servidor remoto
- Cámara del PC en lugar de cámara USB del Pi

Autor: Gloria S.A.
Fecha: 2024
"""

import cv2
import numpy as np
import tensorflow as tf
import time
import threading
import queue
import json
import sqlite3
import requests
from collections import deque
from datetime import datetime
from pathlib import Path
import os


class RaspberryPiSimulator:
    def __init__(self, config):
        """
        Simulador de sistema de inferencia para Raspberry Pi 5.
        
        Args:
            config: dict con configuración (device_id, server_url, etc.)
        """
        self.config = config
        self.device_id = config['device_id']
        
        # Modelos
        self.emotion_interpreter = None
        self.face_embedder = None
        
        # Servidor
        self.server_url = config['server_url']
        
        # Tracking
        self.tracker = SimpleCentroidTracker(max_disappeared=30)
        self.track_emotions = {}  # {track_id: deque([emotions])}
        
        # Face recognition
        self.employee_embeddings = {}
        self.recognition_threshold = config.get('recognition_threshold', 0.6)
        
        # Performance
        self.fps_counter = deque(maxlen=30)
        self.detection_count = 0
        self.recognized_employees = set()
        
        # Estado
        self.running = False
        self.session_id = None
        
        print(f"\n🤖 SIMULADOR DE RASPBERRY PI")
        print(f"   Device ID: {self.device_id}")
        print(f"   Location: {config.get('location', 'Unknown')}")
        
    def initialize(self):
        """Inicializa todos los componentes."""
        
        print(f"\n{'='*80}")
        print(f"📥 INICIALIZANDO SISTEMA")
        print(f"{'='*80}")
        
        # 1. Cargar modelo de emociones
        print("\n1️⃣  Cargando modelo de emociones...")
        emotion_model_path = self.config.get('emotion_model_path')
        
        if emotion_model_path and os.path.exists(emotion_model_path):
            print(f"   📁 Modelo: {emotion_model_path}")
            self.emotion_interpreter = tf.lite.Interpreter(model_path=emotion_model_path)
            self.emotion_interpreter.allocate_tensors()
            print(f"   ✅ Modelo de emociones cargado")
        else:
            print(f"   ⚠️  Modelo no encontrado: {emotion_model_path}")
            print(f"   ℹ️  Usando modelo mock (detección aleatoria)")
            self.emotion_interpreter = None
        
        # 2. Cargar modelo de embeddings (FaceNet)
        print("\n2️⃣  Cargando modelo de embeddings faciales...")
        face_model_path = self.config.get('face_model_path')
        
        if face_model_path and os.path.exists(face_model_path):
            print(f"   📁 Modelo: {face_model_path}")
            self.face_embedder = tf.lite.Interpreter(model_path=face_model_path)
            self.face_embedder.allocate_tensors()
            print(f"   ✅ Modelo de embeddings cargado")
        else:
            print(f"   ⚠️  Modelo no encontrado: {face_model_path}")
            print(f"   ℹ️  Usando embeddings de la base de datos")
            self.face_embedder = None
        
        # 3. Cargar embeddings de empleados desde BD
        print("\n3️⃣  Cargando embeddings de empleados...")
        self._load_employee_embeddings()
        
        # 4. Verificar servidor
        print("\n4️⃣  Verificando servidor...")
        if self._check_server():
            print(f"   ✅ Servidor disponible: {self.server_url}")
            self.session_id = self._create_session()
            print(f"   📝 Session ID: {self.session_id}")
        else:
            print(f"   ⚠️  Servidor no disponible")
            print(f"   ℹ️  Modo offline (solo logging local)")
            self.session_id = f"offline_{int(time.time())}"
        
        # 5. Verificar cámara
        print("\n5️⃣  Verificando cámara...")
        cap = cv2.VideoCapture(self.config['camera_index'])
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"   ✅ Cámara disponible: {width}x{height}")
            cap.release()
        else:
            print(f"   ❌ Error: Cámara no disponible")
            raise RuntimeError("No se pudo acceder a la cámara")
        
        print(f"\n{'='*80}")
        print(f"✅ SISTEMA INICIALIZADO CORRECTAMENTE")
        print(f"{'='*80}\n")
        
    def _load_employee_embeddings(self):
        """Carga embeddings de empleados desde base de datos SQLite."""
        
        db_path = self.config.get('db_path', 'gloria_stress_system.db')
        
        if not os.path.exists(db_path):
            print(f"   ⚠️  Base de datos no encontrada: {db_path}")
            print(f"   ℹ️  Sistema funcionará sin reconocimiento facial")
            return
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, employee_code, full_name, face_embedding
                FROM employees
                WHERE is_active = 1 AND consent_given = 1
            """)
            
            rows = cursor.fetchall()
            
            for row in rows:
                emp_id, code, name, embedding_json = row
                
                if embedding_json:
                    try:
                        embedding = np.array(json.loads(embedding_json))
                        self.employee_embeddings[emp_id] = {
                            'code': code,
                            'name': name,
                            'embedding': embedding
                        }
                    except:
                        continue
            
            cursor.close()
            conn.close()
            
            print(f"   ✅ Cargados {len(self.employee_embeddings)} empleados con embeddings")
            
            if len(self.employee_embeddings) > 0:
                print(f"   📋 Empleados registrados:")
                for emp_id, data in list(self.employee_embeddings.items())[:5]:
                    print(f"      • {data['code']}: {data['name']}")
                if len(self.employee_embeddings) > 5:
                    print(f"      • ... y {len(self.employee_embeddings)-5} más")
            
        except Exception as e:
            print(f"   ❌ Error cargando embeddings: {e}")
            self.employee_embeddings = {}
    
    def _check_server(self):
        """Verifica si el servidor está disponible."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def _create_session(self):
        """Crea nueva sesión en el servidor."""
        try:
            response = requests.post(
                f"{self.server_url}/sessions",
                json={
                    'device_id': self.device_id,
                    'location': self.config.get('location', 'Unknown'),
                    'start_timestamp': datetime.utcnow().isoformat() + 'Z'
                },
                timeout=5
            )
            
            if response.status_code == 200:
                return response.json()['session_id']
            else:
                return f"session_{int(time.time())}"
        except:
            return f"session_{int(time.time())}"
    
    def run(self):
        """Loop principal de captura e inferencia."""
        
        self.running = True
        
        # Configurar cámara
        cap = cv2.VideoCapture(self.config['camera_index'])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Detector de rostros (Haar Cascade - rápido)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        frame_count = 0
        last_report_time = time.time()
        last_detection_time = {}  # Para rate limiting por empleado
        
        print(f"\n{'='*80}")
        print(f"▶️  INICIANDO MONITOREO EN TIEMPO REAL")
        print(f"{'='*80}")
        print(f"   📹 Resolución: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"   🎯 FPS objetivo: {int(cap.get(cv2.CAP_PROP_FPS))}")
        print(f"   ⏩ Frame skip: {self.config.get('frame_skip', 3)}")
        print(f"   👁️  Preview: {'Sí' if self.config.get('show_preview', True) else 'No'}")
        print(f"\n   💡 Presione 'Q' para detener el sistema\n")
        print(f"{'='*80}\n")
        
        try:
            while self.running:
                start_time = time.perf_counter()
                
                # Capturar frame
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error capturando frame")
                    break
                
                frame_count += 1
                
                # Procesar solo cada N frames (simular limitaciones del Pi)
                if frame_count % self.config.get('frame_skip', 3) != 0:
                    continue
                
                # Detectar rostros
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(80, 80)
                )
                
                # Actualizar tracker
                centroids = [(x+w//2, y+h//2) for (x,y,w,h) in faces]
                tracked_objects = self.tracker.update(centroids)
                
                # Procesar cada rostro detectado
                for (track_id, centroid), (x, y, w, h) in zip(tracked_objects.items(), faces):
                    # Extraer ROI
                    face_roi = frame[y:y+h, x:x+w]
                    
                    # 1. Reconocimiento facial
                    employee_id, recognition_conf = self._recognize_face(face_roi)
                    
                    # 2. Detección de emoción
                    emotion, emotion_conf, emotion_vector = self._detect_emotion(face_roi)
                    
                    # 3. Smoothing temporal
                    smoothed_emotion = self._apply_temporal_smoothing(
                        track_id, emotion, emotion_conf
                    )
                    
                    # 4. Enviar detección (con rate limiting)
                    if smoothed_emotion:
                        # Rate limiting: máximo 1 detección cada 2 segundos por empleado
                        emp_key = employee_id if employee_id else f"track_{track_id}"
                        current_time = time.time()
                        
                        if emp_key not in last_detection_time or \
                           (current_time - last_detection_time[emp_key]) >= 2.0:
                            
                            detection_data = self._create_detection_data(
                                track_id, employee_id, recognition_conf,
                                smoothed_emotion, emotion_vector,
                                x, y, w, h,
                                time.perf_counter() - start_time
                            )
                            
                            # Enviar al servidor
                            self._send_detection(detection_data)
                            
                            # Actualizar estadísticas
                            self.detection_count += 1
                            if employee_id:
                                self.recognized_employees.add(employee_id)
                            
                            last_detection_time[emp_key] = current_time
                    
                    # Dibujar en frame
                    if self.config.get('show_preview', True):
                        self._draw_detection(
                            frame, x, y, w, h, track_id,
                            smoothed_emotion, employee_id, recognition_conf
                        )
                
                # Calcular FPS
                elapsed = time.perf_counter() - start_time
                fps = 1.0 / elapsed if elapsed > 0 else 0
                self.fps_counter.append(fps)
                
                # Mostrar preview
                if self.config.get('show_preview', True):
                    avg_fps = np.mean(self.fps_counter)
                    
                    # Panel de info
                    cv2.rectangle(frame, (0, 0), (350, 120), (0, 0, 0), -1)
                    cv2.rectangle(frame, (0, 0), (350, 120), (255, 255, 255), 2)
                    
                    cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Rostros: {len(faces)}", (10, 55),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Detecciones: {self.detection_count}", (10, 85),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Device: {self.device_id}", (10, 115),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                    
                    cv2.imshow(f'Pi Simulator - {self.device_id}', frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n⏹️  Deteniendo por solicitud del usuario...")
                        break
                
                # Reporte periódico
                if time.time() - last_report_time >= 60:  # Cada minuto
                    self._print_performance_stats()
                    last_report_time = time.time()
        
        except KeyboardInterrupt:
            print("\n\n⏹️  Deteniendo por Ctrl+C...")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self._finalize_session()
            print("\n✅ Sistema detenido correctamente\n")
    
    def _recognize_face(self, face_img):
        """Reconoce empleado por embedding facial."""
        
        if len(self.employee_embeddings) == 0:
            return None, 0.0
        
        # Generar embedding del rostro actual
        embedding = self._generate_face_embedding(face_img)
        
        if embedding is None:
            return None, 0.0
        
        # Comparar con embeddings conocidos
        best_match_id = None
        best_similarity = 0.0
        
        for emp_id, emp_data in self.employee_embeddings.items():
            similarity = self._cosine_similarity(embedding, emp_data['embedding'])
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = emp_id
        
        # Retornar solo si supera threshold
        if best_similarity >= self.recognition_threshold:
            return best_match_id, best_similarity
        else:
            return None, best_similarity
    
    def _generate_face_embedding(self, face_img):
        """Genera embedding facial (512-D con FaceNet o mock)."""
        
        if self.face_embedder is not None:
            try:
                # Usar modelo TFLite real
                face_resized = cv2.resize(face_img, (160, 160))
                face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                face_normalized = (face_rgb - 127.5) / 128.0
                face_input = np.expand_dims(face_normalized, axis=0).astype(np.float32)
                
                input_details = self.face_embedder.get_input_details()
                output_details = self.face_embedder.get_output_details()
                
                self.face_embedder.set_tensor(input_details[0]['index'], face_input)
                self.face_embedder.invoke()
                embedding = self.face_embedder.get_tensor(output_details[0]['index'])[0]
                
                return embedding
            except Exception as e:
                print(f"⚠️  Error generando embedding: {e}")
                return None
        else:
            # Mock: generar embedding aleatorio pero consistente
            # (útil para testing sin modelo)
            np.random.seed(hash(face_img.tobytes()) % 2**32)
            return np.random.randn(512).astype(np.float32)
    
    def _detect_emotion(self, face_img):
        """Detecta emoción en rostro."""
        
        emotion_labels = ['neutral', 'stress', 'sad', 'happy', 'fatigue']
        
        if self.emotion_interpreter is not None:
            try:
                # Usar modelo TFLite real
                face_resized = cv2.resize(face_img, (160, 160))
                face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                face_normalized = face_rgb / 255.0
                face_input = np.expand_dims(face_normalized, axis=0).astype(np.float32)
                
                input_details = self.emotion_interpreter.get_input_details()
                output_details = self.emotion_interpreter.get_output_details()
                
                self.emotion_interpreter.set_tensor(input_details[0]['index'], face_input)
                self.emotion_interpreter.invoke()
                predictions = self.emotion_interpreter.get_tensor(output_details[0]['index'])[0]
                
                emotion_idx = np.argmax(predictions)
                emotion = emotion_labels[emotion_idx]
                confidence = float(predictions[emotion_idx])
                
                emotion_vector = {label: float(pred) for label, pred in zip(emotion_labels, predictions)}
                
                return emotion, confidence, emotion_vector
            
            except Exception as e:
                print(f"⚠️  Error detectando emoción: {e}")
                # Fallback a mock
        
        # Mock: detección aleatoria (útil para testing)
        # Simular distribución realista (más neutral, menos estrés)
        probs = np.array([0.5, 0.2, 0.1, 0.15, 0.05])  # neutral, stress, sad, happy, fatigue
        probs = probs + np.random.randn(5) * 0.1
        probs = np.clip(probs, 0, 1)
        probs = probs / probs.sum()
        
        emotion_idx = np.argmax(probs)
        emotion = emotion_labels[emotion_idx]
        confidence = float(probs[emotion_idx])
        
        emotion_vector = {label: float(prob) for label, prob in zip(emotion_labels, probs)}
        
        return emotion, confidence, emotion_vector
    
    def _apply_temporal_smoothing(self, track_id, emotion, confidence, window_size=15):
        """Aplica smoothing temporal para reducir falsos positivos."""
        
        # Inicializar buffer para este track
        if track_id not in self.track_emotions:
            self.track_emotions[track_id] = deque(maxlen=window_size)
        
        # Agregar detección actual
        self.track_emotions[track_id].append({
            'emotion': emotion,
            'confidence': confidence,
            'timestamp': time.time()
        })
        
        # Necesita al menos 5 detecciones
        if len(self.track_emotions[track_id]) < 5:
            return None
        
        # Calcular emoción predominante
        recent_emotions = list(self.track_emotions[track_id])
        emotion_counts = {}
        
        for det in recent_emotions:
            em = det['emotion']
            conf = det['confidence']
            
            if em not in emotion_counts:
                emotion_counts[em] = {'count': 0, 'conf_sum': 0}
            
            emotion_counts[em]['count'] += 1
            emotion_counts[em]['conf_sum'] += conf
        
        # Emoción más frecuente
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1]['count'])[0]
        dominant_count = emotion_counts[dominant_emotion]['count']
        dominant_conf_avg = emotion_counts[dominant_emotion]['conf_sum'] / dominant_count
        
        # Requerir al menos 60% consistencia
        consistency = dominant_count / len(recent_emotions)
        
        if consistency >= 0.6 and dominant_conf_avg >= 0.5:
            return {
                'emotion': dominant_emotion,
                'confidence': dominant_conf_avg,
                'consistency': consistency
            }
        else:
            return None
    
    def _create_detection_data(self, track_id, employee_id, recognition_conf,
                               smoothed_emotion, emotion_vector, x, y, w, h, processing_time):
        """Crea estructura de datos de detección."""
        
        return {
            'type': 'detection',
            'device_id': self.device_id,
            'session_id': self.session_id,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'track_id': f"trk_{track_id}",
            'employee_id': employee_id,
            'recognition_confidence': float(recognition_conf) if employee_id else 0.0,
            'emotion': smoothed_emotion['emotion'],
            'emotion_confidence': float(smoothed_emotion['confidence']),
            'emotion_vector': emotion_vector,
            'bounding_box': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
            'processing_time_ms': float(processing_time * 1000)
        }
    
    def _send_detection(self, detection_data):
        """Envía detección al servidor."""
        
        try:
            response = requests.post(
                f"{self.server_url}/detections",
                json=detection_data,
                timeout=1
            )
            
            if response.status_code != 200:
                print(f"⚠️  Error enviando detección: {response.status_code}")
        
        except Exception as e:
            # Silencioso en modo offline
            pass
        
        # Log local (siempre)
        if self.config.get('log_detections', False):
            log_dir = Path("logs/detections")
            log_dir.mkdir(parents=True, exist_ok=True)
            
            log_file = log_dir / f"{self.device_id}_{datetime.now().strftime('%Y%m%d')}.jsonl"
            with open(log_file, 'a') as f:
                f.write(json.dumps(detection_data) + '\n')
    
    def _draw_detection(self, frame, x, y, w, h, track_id, smoothed_emotion, 
                       employee_id, recognition_conf):
        """Dibuja detección en frame."""
        
        color_map = {
            'neutral': (0, 255, 0),     # Verde
            'happy': (0, 255, 255),     # Amarillo
            'stress': (0, 0, 255),      # Rojo
            'sad': (255, 0, 0),         # Azul
            'fatigue': (128, 0, 128)    # Púrpura
        }
        
        emotion = smoothed_emotion['emotion'] if smoothed_emotion else 'unknown'
        color = color_map.get(emotion, (128, 128, 128))
        
        # Rectángulo
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Etiqueta
        if employee_id and employee_id in self.employee_embeddings:
            emp_name = self.employee_embeddings[employee_id]['name'].split()[0]
            label = f"{emp_name}"
        else:
            label = f"Track#{track_id}"
        
        if smoothed_emotion:
            label += f" | {emotion} ({smoothed_emotion['confidence']:.2f})"
        
        # Fondo para texto
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x, y-text_height-10), (x+text_width, y), color, -1)
        
        cv2.putText(frame, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 2)
    
    def _print_performance_stats(self):
        """Imprime estadísticas de performance."""
        
        avg_fps = np.mean(self.fps_counter) if self.fps_counter else 0
        
        print(f"\n{'='*80}")
        print(f"📊 ESTADÍSTICAS [{self.device_id}]")
        print(f"{'='*80}")
        print(f"   FPS promedio: {avg_fps:.1f}")
        print(f"   Detecciones enviadas: {self.detection_count}")
        print(f"   Empleados únicos reconocidos: {len(self.recognized_employees)}")
        print(f"   Tracks activos: {len(self.track_emotions)}")
        print(f"   Session ID: {self.session_id}")
        print(f"   Servidor: {self.server_url}")
        print(f"{'='*80}\n")
    
    def _finalize_session(self):
        """Finaliza sesión al detener el sistema."""
        
        try:
            requests.post(
                f"{self.server_url}/sessions/{self.session_id}/end",
                json={'end_timestamp': datetime.utcnow().isoformat() + 'Z'},
                timeout=2
            )
        except:
            pass
        
        print(f"\n📊 Resumen Final:")
        print(f"   • Detecciones totales: {self.detection_count}")
        print(f"   • Empleados reconocidos: {len(self.recognized_employees)}")
        print(f"   • Duración de sesión: {self.session_id}")
    
    @staticmethod
    def _cosine_similarity(vec1, vec2):
        """Calcula similitud coseno entre dos vectores."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0


class SimpleCentroidTracker:
    """Tracker simple basado en centroides."""
    
    def __init__(self, max_disappeared=50):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
    
    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
    
    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def update(self, input_centroids):
        if len(input_centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
        
        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            # Calcular distancias
            D = np.linalg.norm(
                np.array(object_centroids)[:, np.newaxis] - np.array(input_centroids),
                axis=2
            )
            
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                
                if D[row, col] < 50:
                    object_id = object_ids[row]
                    self.objects[object_id] = input_centroids[col]
                    self.disappeared[object_id] = 0
                    used_rows.add(row)
                    used_cols.add(col)
            
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            for col in unused_cols:
                self.register(input_centroids[col])
        
        return self.objects


def main():
    """Función principal."""
    
    print("\n" + "="*80)
    print(" "*15 + "🤖 SIMULADOR DE RASPBERRY PI")
    print(" "*10 + "Sistema de Detección de Estrés - Gloria S.A.")
    print("="*80)
    
    # Configuración
    from pi_config import get_config
    config = get_config()
    
    # Inicializar sistema
    simulator = RaspberryPiSimulator(config)
    
    try:
        simulator.initialize()
        simulator.run()
    except Exception as e:
        print(f"\n❌ Error fatal: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())


