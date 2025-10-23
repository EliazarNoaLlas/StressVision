"""
Script de Prueba del Sistema Edge - Stress Vision
Prueba el simulador de Raspberry Pi y el servidor local

Autor: Gloria S.A.
Fecha: 2024
"""

import subprocess
import time
import sys
import os
import requests
import threading


def print_section(title):
    """Imprime sección."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print('='*80)


def test_imports():
    """Prueba que todas las dependencias estén instaladas."""
    print_section("1️⃣  VERIFICANDO DEPENDENCIAS")
    
    required_packages = [
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('tensorflow', 'TensorFlow'),
        ('flask', 'Flask'),
        ('requests', 'Requests'),
        ('sqlite3', 'SQLite3'),
    ]
    
    all_ok = True
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} - NO INSTALADO")
            all_ok = False
    
    return all_ok


def test_database():
    """Verifica que la base de datos esté configurada."""
    print_section("2️⃣  VERIFICANDO BASE DE DATOS")
    
    if not os.path.exists('gloria_stress_system.db'):
        print("   ❌ Base de datos no encontrada")
        print("   💡 Ejecute: python init_database.py")
        return False
    
    import sqlite3
    
    try:
        conn = sqlite3.connect('gloria_stress_system.db')
        cursor = conn.cursor()
        
        # Verificar tablas críticas
        tables = ['employees', 'sessions', 'detection_events']
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"   ✅ Tabla '{table}': {count} registros")
        
        conn.close()
        return True
    
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_camera():
    """Verifica que la cámara esté disponible."""
    print_section("3️⃣  VERIFICANDO CÁMARA")
    
    import cv2
    
    cap = cv2.VideoCapture(0)
    
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            h, w = frame.shape[:2]
            print(f"   ✅ Cámara disponible: {w}x{h}")
            return True
        else:
            print("   ❌ Cámara no puede capturar frames")
            return False
    else:
        print("   ❌ No se pudo abrir la cámara")
        return False


def test_server():
    """Prueba el servidor simulado."""
    print_section("4️⃣  PROBANDO SERVIDOR SIMULADO")
    
    print("   ⏳ Iniciando servidor en background...")
    
    # Iniciar servidor en subprocess
    server_process = subprocess.Popen(
        [sys.executable, 'server_simulator.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Esperar a que inicie
    time.sleep(3)
    
    try:
        # Probar health check
        response = requests.get('http://localhost:5000/health', timeout=2)
        
        if response.status_code == 200:
            print("   ✅ Servidor respondiendo")
            print(f"   📊 Status: {response.json()}")
            
            # Detener servidor
            server_process.terminate()
            server_process.wait(timeout=2)
            
            return True
        else:
            print(f"   ❌ Servidor respondió con código: {response.status_code}")
            server_process.terminate()
            return False
    
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Error conectando al servidor: {e}")
        server_process.terminate()
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        server_process.terminate()
        return False


def test_config():
    """Verifica que la configuración esté correcta."""
    print_section("5️⃣  VERIFICANDO CONFIGURACIÓN")
    
    try:
        from pi_config import get_config, print_config
        
        config = get_config()
        
        print("   ✅ Configuración cargada")
        print(f"   📝 Device ID: {config['device_id']}")
        print(f"   🌐 Server URL: {config['server_url']}")
        print(f"   📹 Camera index: {config['camera_index']}")
        
        if config['emotion_model_path']:
            if os.path.exists(config['emotion_model_path']):
                print(f"   ✅ Modelo de emociones encontrado")
            else:
                print(f"   ⚠️  Modelo de emociones no encontrado (usará mock)")
        else:
            print(f"   ⚠️  Modelo de emociones no configurado")
        
        return True
    
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_full_pipeline():
    """Prueba el pipeline completo durante 30 segundos."""
    print_section("6️⃣  PRUEBA DE PIPELINE COMPLETO (30 segundos)")
    
    print("   Esta prueba ejecutará el sistema completo por 30 segundos")
    print("   para verificar que todo funciona correctamente.\n")
    
    run_test = input("   ¿Ejecutar prueba? (s/n) [s]: ").strip().lower()
    
    if run_test == 'n':
        print("   ⏭️  Prueba omitida")
        return True
    
    print("\n   ⏳ Iniciando servidor...")
    server_process = subprocess.Popen(
        [sys.executable, 'server_simulator.py'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    time.sleep(3)
    
    print("   ⏳ Iniciando simulador de Pi...")
    pi_process = subprocess.Popen(
        [sys.executable, 'pi_simulator.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    print("   ⏳ Ejecutando durante 30 segundos...")
    print("   💡 Si se abre una ventana de cámara, mire hacia ella")
    
    time.sleep(30)
    
    # Detener procesos
    print("\n   ⏹️  Deteniendo procesos...")
    pi_process.terminate()
    server_process.terminate()
    
    time.sleep(2)
    
    # Verificar estadísticas
    try:
        response = requests.get('http://localhost:5000/stats', timeout=2)
        
        if response.status_code == 200:
            stats = response.json()
            print(f"\n   📊 Resultados:")
            print(f"      • Detecciones: {stats['total_detections']}")
            print(f"      • Empleados reconocidos: {stats['unique_employees']}")
            print(f"      • Sesiones: {stats['active_sessions']}")
            
            if stats['total_detections'] > 0:
                print(f"\n   ✅ Pipeline funcionando correctamente")
                return True
            else:
                print(f"\n   ⚠️  No se generaron detecciones")
                print(f"      (Esto es normal si no había rostros frente a la cámara)")
                return True
        else:
            print(f"   ⚠️  No se pudieron obtener estadísticas")
            return False
    
    except:
        print(f"   ⚠️  Servidor no respondió")
        return False


def main():
    """Ejecuta todas las pruebas."""
    
    print("\n" + "="*80)
    print(" "*15 + "🧪 SUITE DE PRUEBAS - FASE 5")
    print(" "*10 + "Simulador de Raspberry Pi - Stress Vision")
    print("="*80)
    
    results = []
    
    # Ejecutar pruebas
    results.append(("Dependencias", test_imports()))
    results.append(("Base de Datos", test_database()))
    results.append(("Cámara", test_camera()))
    results.append(("Servidor", test_server()))
    results.append(("Configuración", test_config()))
    results.append(("Pipeline Completo", test_full_pipeline()))
    
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
        print("\n💡 Para iniciar el sistema completo:")
        print("\n   Terminal 1:")
        print("   python server_simulator.py")
        print("\n   Terminal 2:")
        print("   python pi_simulator.py")
    else:
        print(f"\n⚠️  {failed} prueba(s) fallaron")
        print("\n💡 Revisa los mensajes de error arriba")
    
    print("\n" + "="*80 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Prueba cancelada")
        sys.exit(1)




