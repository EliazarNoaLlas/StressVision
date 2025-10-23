"""
Script de Inicio Rápido - Stress Vision
Guía interactiva para configurar el sistema paso a paso

Autor: Gloria S.A.
"""

import os
import sys
import subprocess


def print_header(text):
    """Imprime un encabezado formateado."""
    print("\n" + "="*80)
    print(text)
    print("="*80 + "\n")


def print_step(number, text):
    """Imprime un paso numerado."""
    print(f"\n{'='*80}")
    print(f"📍 PASO {number}: {text}")
    print('='*80 + "\n")


def check_python_version():
    """Verifica la versión de Python."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (se requiere 3.8+)")
        return False


def check_file_exists(filepath):
    """Verifica si un archivo existe."""
    return os.path.exists(filepath)


def check_dependencies():
    """Verifica si las dependencias están instaladas."""
    required = ['cv2', 'torch', 'numpy', 'streamlit', 'facenet_pytorch']
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing.append(package)
    
    return len(missing) == 0, missing


def main():
    """Función principal del script de inicio rápido."""
    
    print_header("🚀 INICIO RÁPIDO - STRESS VISION")
    print("Gloria S.A. - Sistema de Detección de Estrés Laboral")
    print("\nEste script te guiará paso a paso para configurar el sistema.\n")
    
    # Paso 1: Verificar Python
    print_step(1, "Verificación de Python")
    if not check_python_version():
        print("\n❌ Por favor, instala Python 3.8 o superior")
        print("   Descarga: https://www.python.org/downloads/")
        return
    
    # Paso 2: Verificar/Instalar dependencias
    print_step(2, "Verificación de Dependencias")
    
    deps_ok, missing = check_dependencies()
    
    if not deps_ok:
        print(f"\n⚠️  Faltan {len(missing)} dependencias")
        install = input("\n¿Desea instalarlas ahora? (s/n): ").strip().lower()
        
        if install == 's':
            print("\n⏳ Instalando dependencias...")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
                ])
                print("\n✅ Dependencias instaladas exitosamente")
            except subprocess.CalledProcessError:
                print("\n❌ Error al instalar dependencias")
                print("   Intente manualmente: pip install -r requirements.txt")
                return
        else:
            print("\n⚠️  Instale las dependencias manualmente:")
            print("   pip install -r requirements.txt")
            return
    else:
        print("\n✅ Todas las dependencias están instaladas")
    
    # Paso 3: Crear Base de Datos
    print_step(3, "Creación de Base de Datos")
    
    if check_file_exists("gloria_stress_system.db"):
        print("⚠️  La base de datos ya existe: gloria_stress_system.db")
        recreate = input("¿Desea recrearla? (s/n): ").strip().lower()
        
        if recreate == 's':
            os.remove("gloria_stress_system.db")
            print("🗑️  Base de datos anterior eliminada")
        else:
            print("ℹ️  Usando base de datos existente")
    
    if not check_file_exists("gloria_stress_system.db"):
        print("\n⏳ Creando base de datos...")
        try:
            subprocess.check_call([sys.executable, "init_database.py"])
            print("\n✅ Base de datos creada exitosamente")
        except subprocess.CalledProcessError:
            print("\n❌ Error al crear base de datos")
            return
    
    # Paso 4: Enrollment
    print_step(4, "Enrollment de Empleados")
    
    enrollment_files = [f for f in os.listdir("enrollments") if f.endswith("_embedding.json")] if os.path.exists("enrollments") else []
    
    if enrollment_files:
        print(f"✅ Se encontraron {len(enrollment_files)} enrollments existentes")
        redo = input("¿Desea realizar más enrollments? (s/n): ").strip().lower()
        
        if redo != 's':
            print("ℹ️  Saltando enrollment")
        else:
            print("\n💡 Ejecute manualmente: python enrollment.py")
            input("\nPresione ENTER cuando termine el enrollment...")
    else:
        print("⚠️  No se encontraron enrollments")
        do_enrollment = input("¿Desea realizar el enrollment ahora? (s/n): ").strip().lower()
        
        if do_enrollment == 's':
            print("\n⏳ Iniciando enrollment...")
            try:
                subprocess.check_call([sys.executable, "enrollment.py"])
            except subprocess.CalledProcessError:
                print("\n❌ Error en enrollment")
                return
        else:
            print("\n💡 Puede hacerlo más tarde con: python enrollment.py")
    
    # Paso 5: Cargar Enrollments
    print_step(5, "Carga de Enrollments a Base de Datos")
    
    enrollment_files = [f for f in os.listdir("enrollments") if f.endswith("_embedding.json")] if os.path.exists("enrollments") else []
    
    if enrollment_files:
        print(f"✅ {len(enrollment_files)} enrollments disponibles para cargar")
        load = input("¿Desea cargarlos a la base de datos? (s/n): ").strip().lower()
        
        if load == 's':
            print("\n⏳ Cargando enrollments...")
            try:
                # Ejecutar load_enrollments con opción automática
                subprocess.check_call([sys.executable, "load_enrollments.py"])
            except subprocess.CalledProcessError:
                print("\n❌ Error al cargar enrollments")
                return
        else:
            print("\n💡 Puede hacerlo más tarde con: python load_enrollments.py")
    else:
        print("⚠️  No hay enrollments para cargar")
        print("   Complete primero el Paso 4")
    
    # Resumen Final
    print_header("✅ CONFIGURACIÓN COMPLETADA")
    
    print("📊 Estado del Sistema:")
    print(f"  • Base de datos: {'✅' if check_file_exists('gloria_stress_system.db') else '❌'}")
    print(f"  • Enrollments: {len(enrollment_files) if enrollment_files else 0}")
    
    print("\n🚀 Próximos Pasos:\n")
    
    if not enrollment_files:
        print("1. Ejecutar enrollment:")
        print("   python enrollment.py")
        print("\n2. Cargar enrollments:")
        print("   python load_enrollments.py")
        print("\n3. Iniciar aplicación:")
        print("   streamlit run main.py")
    else:
        print("1. Iniciar aplicación:")
        print("   streamlit run main.py")
        print("\n2. Abrir navegador en:")
        print("   http://localhost:8501")
    
    print("\n📖 Documentación:")
    print("   Ver archivo: INSTRUCCIONES_ENROLLMENT.md")
    
    print("\n" + "="*80)
    print("¡Éxito en la implementación! 🎉")
    print("="*80 + "\n")
    
    # Opción de iniciar la aplicación
    if enrollment_files:
        start_app = input("¿Desea iniciar la aplicación ahora? (s/n): ").strip().lower()
        
        if start_app == 's':
            print("\n⏳ Iniciando Streamlit...")
            try:
                subprocess.check_call([sys.executable, "-m", "streamlit", "run", "main.py"])
            except subprocess.CalledProcessError:
                print("\n❌ Error al iniciar aplicación")
            except KeyboardInterrupt:
                print("\n\n⏹️  Aplicación detenida")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Proceso cancelado por el usuario")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")




