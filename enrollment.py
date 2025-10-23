"""
Sistema de Enrollment - Stress Vision
Captura y generación de embeddings faciales para reconocimiento de empleados

Requisitos previos:
✓ Consentimiento informado firmado
✓ Sesión informativa sobre el sistema
✓ Asignación de employee_code único

Autor: Gloria S.A.
Fecha: 2024
"""

import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from datetime import datetime
import json
import os
import base64
from PIL import Image
import io


class EnrollmentSystem:
    def __init__(self):
        """
        Inicializa el sistema de enrollment con modelos de detección y reconocimiento facial.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Dispositivo: {self.device}")
        
        # Modelo de detección facial MTCNN (más robusto que Haar Cascade)
        print("📥 Cargando detector facial MTCNN...")
        self.face_detector = MTCNN(
            keep_all=False,
            device=self.device,
            min_face_size=100,
            thresholds=[0.6, 0.7, 0.7]
        )
        
        # Modelo de reconocimiento facial FaceNet (InceptionResnetV1)
        print("📥 Cargando modelo FaceNet (InceptionResnetV1)...")
        self.face_model = InceptionResnetV1(
            pretrained='vggface2'
        ).eval().to(self.device)
        
        print("✅ Modelos cargados exitosamente\n")
        
    def capture_embeddings(self, employee_code, employee_name, department="", shift="", num_samples=10):
        """
        Captura múltiples fotos del empleado y genera embeddings.
        
        Args:
            employee_code: código único del empleado (ej: EMP001)
            employee_name: nombre completo
            department: departamento/área
            shift: turno ('morning', 'afternoon', 'night')
            num_samples: número de fotos a capturar (default 10, mínimo 5)
        
        Returns:
            dict con embeddings y metadata o None si falla
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: No se pudo acceder a la cámara")
            return None
        
        # Configurar resolución de cámara
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        embeddings = []
        captured_images = []
        captured = 0
        
        print("\n" + "="*80)
        print(f"👤 ENROLLMENT: {employee_name} ({employee_code})")
        print("="*80)
        print("\n📸 Instrucciones:")
        print("  • Mire directamente a la cámara")
        print("  • Mantenga una expresión neutral")
        print("  • Varíe ligeramente la pose entre capturas (girar cabeza levemente)")
        print("  • Asegúrese de tener buena iluminación")
        print("\n⌨️  Controles:")
        print("  • Presione ESPACIO para capturar foto")
        print("  • Presione Q para cancelar")
        print("\n" + "="*80 + "\n")
        
        # Variables para feedback visual
        feedback_timer = 0
        feedback_text = ""
        feedback_color = (255, 255, 255)
        
        while captured < num_samples:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error al leer frame de la cámara")
                break
            
            # Crear copia para visualización
            display = frame.copy()
            
            # Detectar rostro con MTCNN
            try:
                boxes, probs = self.face_detector.detect(frame)
                
                if boxes is not None and len(boxes) > 0:
                    # Tomar el primer rostro detectado
                    box = boxes[0]
                    prob = probs[0]
                    
                    # Convertir coordenadas
                    x1, y1, x2, y2 = [int(b) for b in box]
                    
                    # Dibujar recuadro
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    
                    # Mostrar confianza
                    cv2.putText(
                        display,
                        f"Confianza: {prob:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                    
                    face_detected = True
                else:
                    face_detected = False
                    cv2.putText(
                        display,
                        "No se detecta rostro",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        2
                    )
            except Exception as e:
                face_detected = False
                cv2.putText(
                    display,
                    "Error en detección",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2
                )
            
            # Mostrar información del enrollment
            h, w = display.shape[:2]
            
            # Panel de información superior
            cv2.rectangle(display, (0, 0), (w, 120), (0, 0, 0), -1)
            cv2.rectangle(display, (0, 0), (w, 120), (255, 255, 255), 2)
            
            cv2.putText(display, f"Empleado: {employee_name}", (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(display, f"Codigo: {employee_code}", (20, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(display, f"Capturadas: {captured}/{num_samples}", (20, 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Barra de progreso
            progress_x = 400
            progress_y = 70
            progress_w = w - 450
            progress_h = 30
            progress = captured / num_samples
            
            cv2.rectangle(display, (progress_x, progress_y), 
                         (progress_x + progress_w, progress_y + progress_h),
                         (255, 255, 255), 2)
            cv2.rectangle(display, (progress_x + 2, progress_y + 2),
                         (progress_x + int(progress_w * progress) - 2, progress_y + progress_h - 2),
                         (0, 255, 0), -1)
            
            # Mostrar feedback temporal
            if feedback_timer > 0:
                cv2.putText(display, feedback_text, (20, h - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, feedback_color, 3)
                feedback_timer -= 1
            
            # Instrucciones
            cv2.putText(display, "ESPACIO: Capturar | Q: Salir", (20, h - 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Mostrar frame
            cv2.imshow('Enrollment - Stress Vision', display)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Capturar con ESPACIO
            if key == ord(' ') and face_detected:
                try:
                    # Extraer región facial
                    face_img = frame[y1:y2, x1:x2]
                    
                    if face_img.size == 0:
                        feedback_text = "Error: Imagen invalida"
                        feedback_color = (0, 0, 255)
                        feedback_timer = 30
                        continue
                    
                    # Generar embedding
                    embedding = self._generate_embedding(face_img)
                    
                    if embedding is not None:
                        embeddings.append(embedding)
                        captured_images.append(face_img)
                        captured += 1
                        
                        print(f"✓ Foto {captured}/{num_samples} capturada (confianza: {prob:.2f})")
                        
                        # Feedback visual
                        feedback_text = f"CAPTURA EXITOSA! ({captured}/{num_samples})"
                        feedback_color = (0, 255, 0)
                        feedback_timer = 30
                        
                        # Guardar imagen de muestra
                        os.makedirs("enrollments", exist_ok=True)
                        cv2.imwrite(
                            f"enrollments/{employee_code}_sample_{captured}.jpg",
                            face_img
                        )
                    else:
                        print("✗ Error al generar embedding, intente nuevamente")
                        feedback_text = "ERROR: Intente de nuevo"
                        feedback_color = (0, 0, 255)
                        feedback_timer = 30
                        
                except Exception as e:
                    print(f"✗ Error al procesar: {e}")
                    feedback_text = f"ERROR: {str(e)[:30]}"
                    feedback_color = (0, 0, 255)
                    feedback_timer = 30
                    
            elif key == ord(' ') and not face_detected:
                feedback_text = "No se detecta rostro!"
                feedback_color = (0, 0, 255)
                feedback_timer = 30
                
            elif key == ord('q'):
                print("\n⚠️  Enrollment cancelado por el usuario")
                cap.release()
                cv2.destroyAllWindows()
                return None
        
        cap.release()
        cv2.destroyAllWindows()
        
        if len(embeddings) < 5:
            print(f"\n❌ Insuficientes muestras ({len(embeddings)}). Mínimo requerido: 5")
            return None
        
        # Calcular embedding promedio y estadísticas
        embeddings_array = np.array(embeddings)
        mean_embedding = np.mean(embeddings_array, axis=0)
        std_embedding = np.std(embeddings_array, axis=0)
        
        # Calcular calidad (consistencia entre muestras)
        quality = self._calculate_quality(embeddings_array)
        
        # Generar thumbnail en base64
        thumbnail_base64 = self._generate_thumbnail(captured_images[0])
        
        result = {
            'employee_code': employee_code,
            'employee_name': employee_name,
            'department': department,
            'shift': shift,
            'mean_embedding': mean_embedding.tolist(),
            'std_embedding': std_embedding.tolist(),
            'num_samples': len(embeddings),
            'quality_score': float(quality),
            'thumbnail_base64': thumbnail_base64,
            'timestamp': datetime.now().isoformat(),
            'consent_given': True,
            'consent_date': datetime.now().isoformat()
        }
        
        print("\n" + "="*80)
        print("✅ ENROLLMENT COMPLETADO")
        print("="*80)
        print(f"  📊 Muestras capturadas: {len(embeddings)}")
        print(f"  🎯 Calidad del embedding: {quality:.2f}/1.0")
        print(f"  📁 Imágenes guardadas en: enrollments/")
        
        if quality < 0.6:
            print(f"  ⚠️  ADVERTENCIA: Calidad baja. Considere repetir el enrollment.")
        elif quality < 0.75:
            print(f"  ⚠️  Calidad aceptable. Funcionalidad garantizada.")
        else:
            print(f"  ✅ Excelente calidad. Reconocimiento óptimo garantizado.")
        
        print("="*80 + "\n")
        
        return result
    
    def _generate_embedding(self, face_img):
        """
        Genera embedding de 512 dimensiones para una imagen facial.
        
        Args:
            face_img: Imagen del rostro (numpy array BGR)
            
        Returns:
            numpy array con el embedding o None si falla
        """
        try:
            # Convertir a RGB
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # Convertir a PIL Image
            pil_img = Image.fromarray(face_rgb)
            
            # Redimensionar a 160x160 (requerido por FaceNet)
            pil_img = pil_img.resize((160, 160), Image.BILINEAR)
            
            # Convertir a tensor
            face_tensor = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).float()
            
            # Normalizar
            face_tensor = (face_tensor - 127.5) / 128.0
            
            # Agregar dimensión de batch
            face_tensor = face_tensor.unsqueeze(0).to(self.device)
            
            # Generar embedding
            with torch.no_grad():
                embedding = self.face_model(face_tensor)
            
            return embedding.cpu().numpy()[0]
            
        except Exception as e:
            print(f"❌ Error generando embedding: {e}")
            return None
    
    def _calculate_quality(self, embeddings_array):
        """
        Calcula score de calidad basado en consistencia entre embeddings.
        
        Args:
            embeddings_array: Array numpy de embeddings
            
        Returns:
            float: Score de calidad (0-1), mayor es mejor
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarities = []
        n = len(embeddings_array)
        
        for i in range(n):
            for j in range(i+1, n):
                sim = cosine_similarity(
                    embeddings_array[i].reshape(1, -1),
                    embeddings_array[j].reshape(1, -1)
                )[0][0]
                similarities.append(sim)
        
        if len(similarities) == 0:
            return 0.0
        
        return np.mean(similarities)
    
    def _generate_thumbnail(self, face_img, size=(100, 100)):
        """
        Genera thumbnail en base64 para almacenar en BD.
        
        Args:
            face_img: Imagen del rostro (numpy array BGR)
            size: Tamaño del thumbnail (ancho, alto)
            
        Returns:
            str: Imagen codificada en base64
        """
        try:
            # Convertir a RGB
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # Crear PIL Image
            pil_img = Image.fromarray(face_rgb)
            
            # Redimensionar
            pil_img.thumbnail(size, Image.LANCZOS)
            
            # Convertir a bytes
            buffer = io.BytesIO()
            pil_img.save(buffer, format='JPEG', quality=85)
            img_bytes = buffer.getvalue()
            
            # Codificar en base64
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            
            return f"data:image/jpeg;base64,{img_base64}"
            
        except Exception as e:
            print(f"⚠️  Error generando thumbnail: {e}")
            return ""

    def batch_enrollment(self, employees_list):
        """
        Realiza enrollment de múltiples empleados en secuencia.
        
        Args:
            employees_list: Lista de tuplas (employee_code, name, department, shift)
            
        Returns:
            list: Lista de resultados de enrollment
        """
        results = []
        total = len(employees_list)
        
        print("\n" + "="*80)
        print(f"📋 ENROLLMENT BATCH: {total} empleados")
        print("="*80)
        
        for i, employee_data in enumerate(employees_list, 1):
            code, name, dept, shift = employee_data
            
            print(f"\n[{i}/{total}] Procesando: {name} ({code})")
            print("-" * 80)
            
            result = self.capture_embeddings(code, name, dept, shift)
            
            if result:
                results.append(result)
                
                # Guardar resultado individual
                os.makedirs("enrollments", exist_ok=True)
                with open(f"enrollments/{code}_embedding.json", 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                print(f"✅ Guardado: enrollments/{code}_embedding.json")
            else:
                print(f"❌ Falló enrollment de {name}")
            
            if i < total:
                input("\n⏸️  Presione ENTER para continuar con el siguiente empleado...")
        
        print("\n" + "="*80)
        print("📊 RESUMEN FINAL")
        print("="*80)
        print(f"  ✅ Enrollments exitosos: {len(results)}/{total}")
        
        if results:
            avg_quality = np.mean([r['quality_score'] for r in results])
            print(f"  📈 Calidad promedio: {avg_quality:.2f}/1.0")
            print(f"  📁 Archivos guardados en: enrollments/")
        
        print("="*80 + "\n")
        
        return results


def main():
    """Función principal para ejecutar el sistema de enrollment."""
    
    print("\n" + "="*80)
    print("👤 SISTEMA DE ENROLLMENT - STRESS VISION")
    print("="*80)
    print("\nGloria S.A. - Registro de Empleados")
    print("Fase 3: Enrollment y Registro de Personal")
    print("\n" + "="*80)
    
    # Verificar cámara
    print("\n🔍 Verificando cámara...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: No se pudo acceder a la cámara")
        print("   Por favor, verifique que:")
        print("   • La cámara esté conectada")
        print("   • Tenga permisos de acceso")
        print("   • No esté siendo usada por otra aplicación")
        return
    cap.release()
    print("✅ Cámara disponible")
    
    # Inicializar sistema
    print("\n⏳ Inicializando sistema de enrollment...")
    enrollment_sys = EnrollmentSystem()
    
    # Opciones
    print("\n" + "="*80)
    print("OPCIONES DE ENROLLMENT")
    print("="*80)
    print("1. Enrollment individual")
    print("2. Enrollment batch (20 personas)")
    print("3. Salir")
    print("="*80)
    
    opcion = input("\nSeleccione una opción (1-3): ").strip()
    
    if opcion == "1":
        # Enrollment individual
        print("\n" + "-"*80)
        print("ENROLLMENT INDIVIDUAL")
        print("-"*80)
        
        employee_code = input("Código de empleado (ej: EMP001): ").strip().upper()
        employee_name = input("Nombre completo: ").strip()
        department = input("Departamento (ej: Producción): ").strip()
        shift = input("Turno (morning/afternoon/night): ").strip().lower()
        
        if not employee_code or not employee_name:
            print("❌ Error: Código y nombre son obligatorios")
            return
        
        result = enrollment_sys.capture_embeddings(
            employee_code, employee_name, department, shift
        )

        if result:
            # ✅ Función auxiliar para convertir tipos no serializables
            def convert_to_native(obj):
                if isinstance(obj, np.generic):  # np.float32, np.int64, etc.
                    return obj.item()
                if isinstance(obj, np.ndarray):  # vectores o embeddings
                    return obj.tolist()
                return str(obj)  # fallback seguro para cualquier otro tipo

            # Guardar resultado en JSON
            os.makedirs("enrollments", exist_ok=True)
            filepath = f"enrollments/{employee_code}_embedding.json"
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=convert_to_native)

            print(f"\n💾 Resultado guardado en: {filepath}")
            print("\n💡 Próximo paso: Ejecutar 'python load_enrollments.py' para cargar a la BD")
    
    elif opcion == "2":
        # Enrollment batch
        print("\n" + "-"*80)
        print("ENROLLMENT BATCH - 20 PERSONAS")
        print("-"*80)
        
        # Lista de 20 empleados para el piloto
        employees = [
            ("EMP001", "Juan Pérez García", "Producción", "morning"),
            ("EMP002", "María González López", "Producción", "morning"),
            ("EMP003", "Carlos Rodríguez Sánchez", "Producción", "afternoon"),
            ("EMP004", "Ana Martínez Ruiz", "Producción", "afternoon"),
            ("EMP005", "Luis Hernández Torres", "Mantenimiento", "morning"),
            ("EMP006", "Carmen Díaz Ramírez", "Calidad", "morning"),
            ("EMP007", "José López Fernández", "Producción", "night"),
            ("EMP008", "Laura García Moreno", "Producción", "morning"),
            ("EMP009", "Miguel Sánchez Jiménez", "Logística", "afternoon"),
            ("EMP010", "Isabel Romero Castro", "Calidad", "morning"),
            ("EMP011", "Francisco Muñoz Ortiz", "Producción", "afternoon"),
            ("EMP012", "Rosa Torres Delgado", "Producción", "morning"),
            ("EMP013", "Antonio Ramírez Vega", "Mantenimiento", "morning"),
            ("EMP014", "Patricia Flores Herrera", "Administración", "morning"),
            ("EMP015", "David Jiménez Navarro", "Producción", "night"),
            ("EMP016", "Elena Castro Molina", "Calidad", "afternoon"),
            ("EMP017", "Javier Moreno Serrano", "Producción", "morning"),
            ("EMP018", "Cristina Ortiz Rubio", "Recursos Humanos", "morning"),
            ("EMP019", "Roberto Vega Pascual", "Producción", "afternoon"),
            ("EMP020", "Beatriz Herrera Gil", "Producción", "morning"),
        ]
        
        print(f"\n📋 Se realizará el enrollment de {len(employees)} empleados")
        print("⏱️  Tiempo estimado: 15-20 minutos por persona")
        print("⏱️  Tiempo total estimado: 5-7 horas")
        
        confirmar = input("\n¿Desea continuar? (s/n): ").strip().lower()
        
        if confirmar == 's':
            results = enrollment_sys.batch_enrollment(employees)
            
            if results:
                print("\n💡 Próximo paso:")
                print("   Ejecutar: python load_enrollments.py")
                print("   Para cargar todos los enrollments a la base de datos")
        else:
            print("❌ Operación cancelada")
    
    else:
        print("👋 Saliendo...")
    
    print("\n" + "="*80)
    print("✅ PROCESO FINALIZADO")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()




