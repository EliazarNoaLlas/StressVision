#!/usr/bin/env python3
"""
Script de Verificación de Imports - Post Migración
Encuentra todos los archivos Python con imports que necesitan actualización
"""

import os
from pathlib import Path
from collections import defaultdict
import re


def find_python_files(directory, exclude_dirs=None):
    """Encuentra todos los archivos Python en el directorio"""
    if exclude_dirs is None:
        exclude_dirs = ['__pycache__', 'venv', 'env', '.git', 'backup_*', 'build', 'dist']
    
    for root, dirs, files in os.walk(directory):
        # Filtrar directorios a excluir
        dirs[:] = [d for d in dirs if not any(
            d == excl or d.startswith(excl.rstrip('*')) 
            for excl in exclude_dirs
        )]
        
        for file in files:
            if file.endswith('.py'):
                yield Path(root) / file


def check_old_imports(file_path):
    """Verifica si un archivo tiene imports que podrían necesitar actualización"""
    # Módulos que han sido movidos
    old_modules = {
        'backend_api': 'backend/app/backend_api.py',
        'pi_simulator': 'edge/src/pi_simulator.py',
        'pi_config': 'edge/config/pi_config.py',
        'model_architecture': 'models/training/scripts/model_architecture.py',
        'model_trainer': 'models/training/scripts/model_trainer.py',
        'data_preparation': 'models/training/scripts/data_preparation.py',
        'evaluate_model': 'models/training/scripts/evaluate_model.py',
        'train_model': 'models/training/scripts/train_model.py',
        'convert_to_tflite': 'models/training/scripts/convert_to_tflite.py',
        'enrollment': 'enrollment/enrollment.py',
        'load_enrollments': 'enrollment/load_enrollments.py',
        'report_generator': 'reporting/report_generator.py',
        'init_database': 'database/init_database.py',
        'quick_start': 'scripts/testing/quick_start.py',
        'server_simulator': 'backend/scripts/server_simulator.py',
        'start_complete_system': 'backend/scripts/start_complete_system.py',
        'start_pi_system': 'edge/scripts/start_pi_system.py',
        'test_pi_system': 'edge/tests/test_pi_system.py',
        'test_system': 'edge/tests/test_system.py',
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"⚠️  Error leyendo {file_path}: {e}")
        return []
    
    found_imports = []
    
    # Buscar imports directos: import module
    for module, new_location in old_modules.items():
        # Patrón para "import module" o "import module as ..."
        pattern1 = rf'\bimport\s+{re.escape(module)}(\s+as\s+\w+)?(\s|$|,)'
        # Patrón para "from module import ..."
        pattern2 = rf'\bfrom\s+{re.escape(module)}\s+import'
        
        if re.search(pattern1, content) or re.search(pattern2, content):
            found_imports.append({
                'module': module,
                'new_location': new_location
            })
    
    return found_imports


def check_old_paths(file_path):
    """Verifica si hay rutas hardcodeadas que necesitan actualización"""
    old_paths = {
        'gloria_stress_system.db': 'database/gloria_stress_system.db',
        'enrollments/': 'enrollment/data/enrollments/',
        '"enrollments': 'enrollment/data/enrollments',
        "'enrollments": 'enrollment/data/enrollments',
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        return []
    
    found_paths = []
    for old_path, new_path in old_paths.items():
        if old_path in content:
            # Contar ocurrencias
            count = content.count(old_path)
            found_paths.append({
                'old_path': old_path,
                'new_path': new_path,
                'count': count
            })
    
    return found_paths


def main():
    project_root = Path('.')
    print("=" * 70)
    print("  VERIFICACIÓN DE IMPORTS - POST MIGRACIÓN")
    print("=" * 70)
    print()
    print("Buscando archivos Python con imports antiguos...\n")
    
    issues_by_category = defaultdict(list)
    path_issues_by_file = {}
    total_files_with_issues = 0
    
    for py_file in find_python_files(project_root):
        # Verificar imports
        old_imports = check_old_imports(py_file)
        
        # Verificar paths
        old_paths = check_old_paths(py_file)
        
        if old_imports or old_paths:
            total_files_with_issues += 1
            
            # Categorizar por directorio principal
            parts = py_file.parts
            if len(parts) > 1:
                category = parts[0]
            else:
                category = 'root'
            
            issues_by_category[category].append({
                'file': py_file,
                'imports': old_imports,
                'paths': old_paths
            })
    
    # Mostrar resultados por categoría
    if total_files_with_issues == 0:
        print("✅ ¡Excelente! No se encontraron imports o paths antiguos que necesiten actualización.")
        return
    
    print(f"⚠️  Se encontraron {total_files_with_issues} archivo(s) con posibles problemas:\n")
    
    for category in sorted(issues_by_category.keys()):
        issues = issues_by_category[category]
        print(f"📁 {category.upper()}/")
        print("-" * 70)
        
        for issue in issues:
            file_path = issue['file']
            imports = issue['imports']
            paths = issue['paths']
            
            print(f"\n  📄 {file_path}")
            
            if imports:
                print(f"     Imports a actualizar:")
                for imp in imports:
                    print(f"       • import {imp['module']}")
                    print(f"         → Nueva ubicación: {imp['new_location']}")
            
            if paths:
                print(f"     Rutas hardcodeadas a actualizar:")
                for path in paths:
                    print(f"       • '{path['old_path']}' ({path['count']} ocurrencia(s))")
                    print(f"         → Usar: {path['new_path']}")
        
        print()
    
    # Resumen y recomendaciones
    print("=" * 70)
    print("RESUMEN")
    print("=" * 70)
    print(f"\nTotal de archivos con problemas: {total_files_with_issues}")
    print(f"Categorías afectadas: {len(issues_by_category)}")
    
    print("\n" + "=" * 70)
    print("PRÓXIMOS PASOS")
    print("=" * 70)
    print("""
1. Revisa cada archivo listado arriba
2. Actualiza los imports según la GUIA_ACTUALIZACION_IMPORTS.md
3. Actualiza las rutas hardcodeadas a usar Path
4. Ejecuta este script nuevamente para verificar
5. Prueba cada módulo después de actualizar

Ejemplo de actualización de imports:

  # ANTES
  import backend_api
  
  # DESPUÉS  
  import sys
  from pathlib import Path
  sys.path.insert(0, str(Path(__file__).parent.parent))
  from backend.app import backend_api

Ejemplo de actualización de rutas:

  # ANTES
  db_path = "gloria_stress_system.db"
  
  # DESPUÉS
  from pathlib import Path
  project_root = Path(__file__).parent.parent
  db_path = project_root / "database" / "gloria_stress_system.db"

Ver GUIA_ACTUALIZACION_IMPORTS.md para más ejemplos.
""")


if __name__ == "__main__":
    main()


