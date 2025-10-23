"""
Script de Prueba del Sistema - Stress Vision
Verifica que todos los componentes estén funcionando correctamente

Autor: Gloria S.A.
"""

import sys
import os


def print_section(title):
    """Imprime una sección."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print('='*80)


def test_python_version():
    """Prueba 1: Verificar versión de Python."""
    print("\n🔍 Prueba 1: Versión de Python")
    version = sys.version_info
    
    if version.major >= 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} (Correcto)")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} (Se requiere 3.8+)")
        return False


def test_imports():
    """Prueba 2: Verificar imports de librerías críticas."""
    print("\n🔍 Prueba 2: Librerías Críticas")
    
    tests = [
        ("cv2", "OpenCV"),
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("streamlit", "Streamlit"),
        ("facenet_pytorch", "FaceNet PyTorch"),
        ("sklearn", "Scikit-learn"),
        ("PIL", "Pillow"),
    ]
    
    passed = 0
    failed = 0
    
    for module, name in tests:
        try:
            __import__(module)
            print(f"   ✅ {name}")
            passed += 1
        except ImportError as e:
            print(f"   ❌ {name} - {e}")
            failed += 1
    
    print(f"\n   📊 Resultado: {passed}/{len(tests)} librerías disponibles")
    return failed == 0


def test_camera():
    """Prueba 3: Verificar acceso a cámara."""
    print("\n🔍 Prueba 3: Acceso a Cámara")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"   ✅ Cámara accesible (Resolución: {w}x{h})")
                return True
            else:
                print("   ❌ Cámara abierta pero no puede leer frames")
                return False
        else:
            print("   ❌ No se pudo abrir la cámara")
            print("   💡 Verifica permisos y que no esté siendo usada por otra app")
            return False
            
    except Exception as e:
        print(f"   ❌ Error al acceder a cámara: {e}")
        return False


def test_models():
    """Prueba 4: Cargar modelos de ML."""
    print("\n🔍 Prueba 4: Modelos de Machine Learning")
    
    try:
        import torch
        from facenet_pytorch import InceptionResnetV1, MTCNN
        
        print("   ⏳ Cargando MTCNN...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mtcnn = MTCNN(keep_all=False, device=device)
        print(f"   ✅ MTCNN cargado (Device: {device})")
        
        print("   ⏳ Cargando FaceNet (InceptionResnetV1)...")
        model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        print(f"   ✅ FaceNet cargado (Device: {device})")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error al cargar modelos: {e}")
        return False


def test_database():
    """Prueba 5: Verificar base de datos."""
    print("\n🔍 Prueba 5: Base de Datos SQLite")
    
    db_path = "gloria_stress_system.db"
    
    if not os.path.exists(db_path):
        print(f"   ⚠️  Base de datos no encontrada: {db_path}")
        print("   💡 Ejecute: python init_database.py")
        return False
    
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Verificar integridad
        cursor.execute("PRAGMA integrity_check")
        result = cursor.fetchone()
        
        if result[0] != 'ok':
            print(f"   ❌ Error de integridad: {result[0]}")
            return False
        
        # Contar tablas
        cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        table_count = cursor.fetchone()[0]
        
        # Contar empleados
        try:
            cursor.execute("SELECT COUNT(*) FROM employees")
            emp_count = cursor.fetchone()[0]
        except:
            emp_count = 0
        
        print(f"   ✅ Base de datos OK")
        print(f"   📊 Tablas: {table_count}")
        print(f"   👥 Empleados registrados: {emp_count}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"   ❌ Error al verificar BD: {e}")
        return False


def test_enrollments():
    """Prueba 6: Verificar enrollments."""
    print("\n🔍 Prueba 6: Enrollments")
    
    if not os.path.exists("enrollments"):
        print("   ⚠️  Carpeta de enrollments no encontrada")
        print("   💡 Ejecute: python enrollment.py")
        return False
    
    import glob
    enrollment_files = glob.glob("enrollments/*_embedding.json")
    image_files = glob.glob("enrollments/*_sample_*.jpg")
    
    print(f"   📄 Archivos de embedding: {len(enrollment_files)}")
    print(f"   🖼️  Imágenes de muestra: {len(image_files)}")
    
    if len(enrollment_files) == 0:
        print("   ⚠️  No hay enrollments registrados")
        print("   💡 Ejecute: python enrollment.py")
        return False
    
    # Verificar un enrollment
    try:
        import json
        with open(enrollment_files[0], 'r') as f:
            data = json.load(f)
        
        embedding = data.get('mean_embedding', [])
        quality = data.get('quality_score', 0)
        
        print(f"   ✅ Enrollments OK")
        print(f"   📊 Dimensión del embedding: {len(embedding)}")
        print(f"   🎯 Calidad promedio: {quality:.2f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error al leer enrollment: {e}")
        return False


def test_face_detection():
    """Prueba 7: Test de detección facial en vivo."""
    print("\n🔍 Prueba 7: Detección Facial en Vivo")
    
    try:
        import cv2
        import torch
        from facenet_pytorch import MTCNN
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        detector = MTCNN(keep_all=False, device=device)
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("   ❌ No se pudo abrir la cámara")
            return False
        
        print("   ⏳ Capturando frame de prueba...")
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("   ❌ No se pudo capturar frame")
            return False
        
        print("   ⏳ Detectando rostro...")
        boxes, probs = detector.detect(frame)
        
        if boxes is not None and len(boxes) > 0:
            prob = probs[0]
            print(f"   ✅ Rostro detectado (Confianza: {prob:.2f})")
            return True
        else:
            print("   ⚠️  No se detectó ningún rostro en el frame")
            print("   💡 Asegúrese de estar frente a la cámara")
            return False
            
    except Exception as e:
        print(f"   ❌ Error en detección facial: {e}")
        return False


def test_streamlit():
    """Prueba 8: Verificar que Streamlit esté instalado."""
    print("\n🔍 Prueba 8: Streamlit")
    
    try:
        import streamlit as st
        version = st.__version__
        print(f"   ✅ Streamlit {version} instalado")
        
        if os.path.exists("main.py"):
            print("   ✅ main.py encontrado")
            print("   💡 Ejecute: streamlit run main.py")
        else:
            print("   ⚠️  main.py no encontrado")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error con Streamlit: {e}")
        return False


def main():
    """Ejecuta todas las pruebas."""
    
    print_section("🧪 PRUEBA DEL SISTEMA - STRESS VISION")
    print("\nGloria S.A. - Sistema de Detección de Estrés Laboral")
    print("Este script verifica que todos los componentes estén funcionando.\n")
    
    results = []
    
    # Ejecutar pruebas
    results.append(("Python Version", test_python_version()))
    results.append(("Librerías", test_imports()))
    results.append(("Cámara", test_camera()))
    results.append(("Modelos ML", test_models()))
    results.append(("Base de Datos", test_database()))
    results.append(("Enrollments", test_enrollments()))
    results.append(("Detección Facial", test_face_detection()))
    results.append(("Streamlit", test_streamlit()))
    
    # Resumen
    print_section("📊 RESUMEN DE PRUEBAS")
    
    passed = sum(1 for _, result in results if result)
    failed = sum(1 for _, result in results if not result)
    total = len(results)
    
    print("\nResultados individuales:")
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}  {name}")
    
    print(f"\n{'='*80}")
    print(f"  Total: {passed}/{total} pruebas exitosas ({(passed/total)*100:.1f}%)")
    print('='*80)
    
    if failed == 0:
        print("\n🎉 ¡TODAS LAS PRUEBAS PASARON!")
        print("\n✅ El sistema está listo para usar")
        print("\n💡 Próximos pasos:")
        print("   1. Si no hay enrollments: python enrollment.py")
        print("   2. Cargar enrollments: python load_enrollments.py")
        print("   3. Iniciar aplicación: streamlit run main.py")
    else:
        print(f"\n⚠️  {failed} prueba(s) fallaron")
        print("\n💡 Revisa los mensajes de error arriba para solucionar los problemas")
        
        if not any(name == "Enrollments" and result for name, result in results):
            print("\n📝 NOTA: Es normal que 'Enrollments' falle si aún no has registrado empleados")
    
    print("\n" + "="*80 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Prueba cancelada por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)




