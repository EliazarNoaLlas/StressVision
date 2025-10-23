"""
Launcher del Sistema Edge Simulado - Stress Vision
Inicia servidor y simulador de Raspberry Pi automáticamente

Autor: Gloria S.A.
Fecha: 2024
"""

import subprocess
import time
import sys
import os
import requests


def check_dependencies():
    """Verifica que las dependencias estén instaladas."""
    required = ['flask', 'requests', 'cv2', 'tensorflow', 'numpy']
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            print(f"❌ Falta dependencia: {package}")
            print("   Ejecute: pip install -r requirements.txt")
            return False
    
    return True


def check_database():
    """Verifica que la base de datos exista."""
    if not os.path.exists('gloria_stress_system.db'):
        print("❌ Base de datos no encontrada")
        print("   Ejecute: python init_database.py")
        return False
    
    return True


def start_server():
    """Inicia el servidor simulado."""
    print("\n🖥️  Iniciando servidor simulado...")
    
    # Verificar si ya está corriendo
    try:
        response = requests.get('http://localhost:5000/health', timeout=1)
        if response.status_code == 200:
            print("   ⚠️  Servidor ya está corriendo")
            return None
    except:
        pass
    
    # Iniciar servidor
    if sys.platform == 'win32':
        # Windows: Abrir en nueva ventana
        server_process = subprocess.Popen(
            ['start', 'cmd', '/k', 'python', 'server_simulator.py'],
            shell=True
        )
    else:
        # Linux/Mac: Background process
        server_process = subprocess.Popen(
            [sys.executable, 'server_simulator.py'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    
    # Esperar a que inicie
    print("   ⏳ Esperando a que el servidor inicie...")
    for _ in range(10):
        time.sleep(1)
        try:
            response = requests.get('http://localhost:5000/health', timeout=1)
            if response.status_code == 200:
                print("   ✅ Servidor iniciado exitosamente")
                print("   📍 URL: http://localhost:5000")
                return server_process
        except:
            continue
    
    print("   ⚠️  Servidor tardó en iniciar, pero puede estar corriendo")
    return server_process


def start_pi_simulator():
    """Inicia el simulador de Raspberry Pi."""
    print("\n🤖 Iniciando simulador de Raspberry Pi...")
    
    if sys.platform == 'win32':
        # Windows: Abrir en nueva ventana
        pi_process = subprocess.Popen(
            ['start', 'cmd', '/k', 'python', 'pi_simulator.py'],
            shell=True
        )
    else:
        # Linux/Mac: Ventana nueva con terminal
        pi_process = subprocess.Popen(
            [sys.executable, 'pi_simulator.py']
        )
    
    print("   ✅ Simulador iniciado")
    print("   💡 Se abrirá una ventana con la cámara en vivo")
    print("   💡 Presione 'Q' en la ventana para detener")
    
    return pi_process


def show_instructions():
    """Muestra instrucciones de uso."""
    print("\n" + "="*80)
    print("📖 INSTRUCCIONES DE USO")
    print("="*80)
    print("""
El sistema se ha iniciado con 2 componentes:

1️⃣  SERVIDOR SIMULADO (Ventana 1)
   • Recibe detecciones del simulador de Pi
   • Guarda en base de datos
   • Muestra estadísticas cada 30 segundos
   • URL: http://localhost:5000
   
   Endpoints disponibles:
   • GET  /health        - Health check
   • GET  /stats         - Estadísticas en tiempo real
   • POST /sessions      - Crear sesión
   • POST /detections    - Recibir detección

2️⃣  SIMULADOR DE RASPBERRY PI (Ventana 2)
   • Captura video de la cámara
   • Detecta rostros
   • Reconoce empleados
   • Detecta emociones
   • Envía al servidor
   
   Ventana de preview:
   • Muestra video con detecciones
   • FPS en tiempo real
   • Número de rostros detectados
   • Contador de detecciones
   • Presione 'Q' para detener

PARA DETENER EL SISTEMA:
• Opción 1: Presione 'Q' en la ventana del simulador
• Opción 2: Ctrl+C en cada ventana
• Opción 3: Cierre las ventanas

PARA VER ESTADÍSTICAS:
• Abrir en navegador: http://localhost:5000/stats
• O ejecutar: curl http://localhost:5000/stats

PARA VER DETECCIONES GUARDADAS:
• Consultar base de datos:
  sqlite3 gloria_stress_system.db "SELECT * FROM detection_events ORDER BY timestamp DESC LIMIT 10;"
    """)
    print("="*80 + "\n")


def main():
    """Función principal."""
    
    print("\n" + "="*80)
    print(" "*15 + "🚀 LAUNCHER - SISTEMA EDGE SIMULADO")
    print(" "*15 + "Stress Vision - Gloria S.A.")
    print("="*80)
    
    # Verificaciones previas
    print("\n📋 Verificando requisitos previos...\n")
    
    if not check_dependencies():
        print("\n❌ Faltan dependencias. No se puede continuar.")
        return 1
    
    if not check_database():
        print("\n❌ Base de datos no configurada. No se puede continuar.")
        return 1
    
    print("\n✅ Todos los requisitos cumplidos")
    
    # Preguntar si continuar
    print("\n" + "="*80)
    proceed = input("\n¿Iniciar el sistema completo? (s/n) [s]: ").strip().lower()
    
    if proceed == 'n':
        print("❌ Operación cancelada")
        return 0
    
    # Iniciar componentes
    print("\n" + "="*80)
    print("🚀 INICIANDO SISTEMA")
    print("="*80)
    
    server_process = start_server()
    
    if server_process is None and sys.platform != 'win32':
        print("\n⚠️  No se pudo verificar el servidor")
        cont = input("¿Continuar de todas formas? (s/n): ").strip().lower()
        if cont != 's':
            return 1
    
    time.sleep(2)
    
    pi_process = start_pi_simulator()
    
    # Mostrar instrucciones
    show_instructions()
    
    # Mantener script vivo
    print("💡 Este script mantendrá el sistema activo.")
    print("   Presione Ctrl+C para detener todo el sistema.\n")
    
    try:
        # Esperar indefinidamente
        while True:
            time.sleep(5)
            
            # Verificar que el servidor siga vivo
            try:
                response = requests.get('http://localhost:5000/health', timeout=1)
                if response.status_code != 200:
                    print("\n⚠️  Servidor dejó de responder")
                    break
            except:
                # Es normal si el servidor fue cerrado manualmente
                pass
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Deteniendo sistema...")
        
        # Intentar detener procesos gracefully
        if pi_process:
            try:
                pi_process.terminate()
                pi_process.wait(timeout=5)
            except:
                pass
        
        if server_process:
            try:
                server_process.terminate()
                server_process.wait(timeout=5)
            except:
                pass
        
        print("✅ Sistema detenido")
    
    print("\n" + "="*80)
    print("👋 ¡Hasta pronto!")
    print("="*80 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())




