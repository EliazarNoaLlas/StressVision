"""
Launcher del Sistema Completo - Stress Vision
Inicia todos los componentes del sistema integrado

Componentes:
1. Backend API (FastAPI - Puerto 8000)
2. Report Generator (reportes cada 15 min)
3. Server Simulator (recibe detecciones - Puerto 5000)
4. Pi Simulator (captura y detección)
5. Dashboard Streamlit (Puerto 8501)

Autor: Gloria S.A.
Fecha: 2024
"""

import subprocess
import time
import sys
import os
import requests


def print_header(text):
    print(f"\n{'='*80}")
    print(f"  {text}")
    print('='*80)


def check_port(port):
    """Verifica si un puerto está en uso."""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    return result == 0


def kill_port(port):
    """Intenta liberar un puerto."""
    if sys.platform == 'win32':
        os.system(f'netstat -ano | findstr :{port}')
        print(f"\n💡 Para liberar el puerto {port}:")
        print(f"   1. Identificar PID arriba")
        print(f"   2. Ejecutar: taskkill /PID <PID> /F")
    else:
        os.system(f'lsof -ti:{port} | xargs kill -9 2>/dev/null')


def main():
    print_header("🚀 LAUNCHER - SISTEMA COMPLETO")
    print("\nStress Vision - Gloria S.A.")
    print("Sistema Integrado de Detección de Estrés Laboral\n")
    
    # Verificar requisitos
    print("📋 Verificando requisitos...")
    
    if not os.path.exists('gloria_stress_system.db'):
        print("\n❌ Base de datos no encontrada")
        print("   Ejecute: python init_database.py")
        return 1
    
    print("   ✅ Base de datos OK")
    
    # Verificar puertos
    ports_needed = [5000, 8000, 8501]
    ports_busy = []
    
    for port in ports_needed:
        if check_port(port):
            ports_busy.append(port)
    
    if ports_busy:
        print(f"\n⚠️  Puertos en uso: {ports_busy}")
        for port in ports_busy:
            print(f"\n   Puerto {port}:")
            kill_port(port)
        
        cont = input("\n¿Continuar de todas formas? (s/n): ").strip().lower()
        if cont != 's':
            return 1
    
    # Seleccionar componentes
    print_header("📦 COMPONENTES DISPONIBLES")
    print("""
Seleccione los componentes a iniciar:

1. Sistema Edge Completo (Recomendado)
   • Server Simulator (puerto 5000)
   • Pi Simulator (cámara + detección)
   
2. Backend API + Dashboard
   • Backend FastAPI (puerto 8000)  
   • Dashboard Streamlit (puerto 8501)
   
3. Sistema Completo (Todos)
   • Server Simulator
   • Pi Simulator
   • Backend API
   • Dashboard Streamlit
   • Report Generator
   
4. Solo Backend API
5. Solo Dashboard
6. Salir
    """)
    
    opcion = input("Seleccione opción (1-6) [3]: ").strip()
    if not opcion:
        opcion = '3'
    
    if opcion == '6':
        print("👋 Saliendo...")
        return 0
    
    processes = []
    
    try:
        print_header("🚀 INICIANDO COMPONENTES")
        
        # Opción 1: Sistema Edge
        if opcion in ['1', '3']:
            print("\n1️⃣  Iniciando Server Simulator (puerto 5000)...")
            
            if sys.platform == 'win32':
                proc = subprocess.Popen(
                    ['start', 'cmd', '/k', 'python', 'server_simulator.py'],
                    shell=True
                )
            else:
                proc = subprocess.Popen(
                    [sys.executable, 'server_simulator.py'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            
            processes.append(('Server Simulator', proc))
            time.sleep(3)
            print("   ✅ Server Simulator iniciado")
            
            print("\n2️⃣  Iniciando Pi Simulator...")
            
            if sys.platform == 'win32':
                proc = subprocess.Popen(
                    ['start', 'cmd', '/k', 'python', 'pi_simulator.py'],
                    shell=True
                )
            else:
                proc = subprocess.Popen(
                    [sys.executable, 'pi_simulator.py']
                )
            
            processes.append(('Pi Simulator', proc))
            time.sleep(2)
            print("   ✅ Pi Simulator iniciado")
        
        # Opción 2 o 3: Backend API
        if opcion in ['2', '3', '4']:
            print("\n3️⃣  Iniciando Backend API (puerto 8000)...")
            
            if sys.platform == 'win32':
                proc = subprocess.Popen(
                    ['start', 'cmd', '/k', 'python', 'backend_api.py'],
                    shell=True
                )
            else:
                proc = subprocess.Popen(
                    [sys.executable, 'backend_api.py'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            
            processes.append(('Backend API', proc))
            time.sleep(3)
            print("   ✅ Backend API iniciado")
            print("   📄 Docs: http://localhost:8000/api/docs")
        
        # Opción 2, 3 o 5: Dashboard
        if opcion in ['2', '3', '5']:
            print("\n4️⃣  Iniciando Dashboard Streamlit (puerto 8501)...")
            
            if sys.platform == 'win32':
                proc = subprocess.Popen(
                    ['start', 'cmd', '/k', 'streamlit', 'run', 'main.py'],
                    shell=True
                )
            else:
                proc = subprocess.Popen(
                    [sys.executable, '-m', 'streamlit', 'run', 'main.py'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            
            processes.append(('Dashboard', proc))
            time.sleep(3)
            print("   ✅ Dashboard iniciado")
            print("   🌐 URL: http://localhost:8501")
        
        # Opción 3: Report Generator
        if opcion == '3':
            print("\n5️⃣  Iniciando Report Generator...")
            
            if sys.platform == 'win32':
                proc = subprocess.Popen(
                    ['start', 'cmd', '/k', 'python', '-c', 
                     '"from report_generator import ReportGenerator; import time; gen = ReportGenerator(); gen.start(15); time.sleep(999999)"'],
                    shell=True
                )
            else:
                proc = subprocess.Popen(
                    [sys.executable, 'report_generator.py'],
                    stdin=subprocess.PIPE
                )
                # Enviar opción '2' para iniciar automático
                time.sleep(1)
                try:
                    proc.stdin.write(b'2\n')
                    proc.stdin.flush()
                except:
                    pass
            
            processes.append(('Report Generator', proc))
            print("   ✅ Report Generator iniciado (reportes cada 15 min)")
        
        # Resumen
        print_header("✅ SISTEMA INICIADO")
        
        print("\n📊 Componentes activos:")
        for name, proc in processes:
            print(f"   • {name}")
        
        print("\n🌐 URLs:")
        if opcion in ['1', '3']:
            print("   • Server Simulator: http://localhost:5000/stats")
        if opcion in ['2', '3', '4']:
            print("   • Backend API: http://localhost:8000")
            print("   • API Docs: http://localhost:8000/api/docs")
        if opcion in ['2', '3', '5']:
            print("   • Dashboard: http://localhost:8501")
        
        print("\n💡 Para detener:")
        print("   • Presione Ctrl+C aquí")
        print("   • O cierre las ventanas abiertas")
        print("   • O presione 'Q' en ventana del simulador")
        
        print("\n⏳ Sistema en ejecución. Presione Ctrl+C para detener...\n")
        print("="*80 + "\n")
        
        # Mantener vivo
        try:
            while True:
                time.sleep(5)
                
                # Verificar que los procesos sigan vivos (opcional)
                # for name, proc in processes:
                #     if proc.poll() is not None:
                #         print(f"\n⚠️  {name} se detuvo inesperadamente")
        
        except KeyboardInterrupt:
            print("\n\n⏹️  Deteniendo sistema...")
            
            # Intentar detener procesos gracefully
            for name, proc in processes:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                    print(f"   ✓ {name} detenido")
                except:
                    try:
                        proc.kill()
                    except:
                        pass
            
            print("\n✅ Sistema detenido")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "="*80)
    print("👋 ¡Hasta pronto!")
    print("="*80 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())





