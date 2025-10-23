# Guía de Actualización de Imports - Post Migración

Esta guía te ayudará a actualizar todos los imports en tu código Python después de reorganizar la estructura de carpetas.

## Cambios de Imports Necesarios

### 1. Archivos en `backend/`

#### `backend/main.py`
```python
# ANTES
from backend_api import app

# DESPUÉS
from app.backend_api import app
```

#### `backend/scripts/server_simulator.py`
```python
# ANTES
import backend_api

# DESPUÉS  
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from app import backend_api
```

#### `backend/scripts/start_complete_system.py`
```python
# ANTES
import backend_api
import pi_simulator

# DESPUÉS
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from app import backend_api
# Para pi_simulator, necesitarás importarlo desde edge
```

### 2. Archivos en `models/training/scripts/`

Todos los scripts de entrenamiento necesitarán ajustar sus imports si hacen referencia a otros módulos:

#### `models/training/scripts/train_model.py`
```python
# ANTES
from model_architecture import create_model
from model_trainer import ModelTrainer
from data_preparation import prepare_data

# DESPUÉS
from model_architecture import create_model  # Si están en la misma carpeta
from model_trainer import ModelTrainer
from data_preparation import prepare_data

# O si necesitas importar desde la raíz del proyecto:
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
```

#### `models/training/scripts/evaluate_model.py`
```python
# Similar al anterior, ajustar rutas relativas
```

### 3. Archivos en `edge/`

#### `edge/scripts/start_pi_system.py`
```python
# ANTES
from pi_simulator import PiSimulator
from pi_config import PI_CONFIG

# DESPUÉS
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.pi_simulator import PiSimulator
from config.pi_config import PI_CONFIG
```

#### `edge/tests/test_pi_system.py`
```python
# ANTES
import pi_simulator
import pi_config

# DESPUÉS
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src import pi_simulator
from config import pi_config
```

### 4. Archivos en `enrollment/`

#### `enrollment/enrollment.py`

```python
# Si usa la base de datos o configuración
# ANTES
from database.init_database import get_db_connection

# DESPUÉS
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from database.init_database import get_db_connection
```

#### `enrollment/load_enrollments.py`
```python
# Similar ajuste de paths
# DESPUÉS
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from database.init_database import get_db_connection
```

### 5. Archivos en `reporting/`

#### `reporting/report_generator.py`
```python
# ANTES
from backend_api import get_detections_from_db

# DESPUÉS
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.app.backend_api import get_detections_from_db
```

### 6. Archivos en `scripts/`

#### `scripts/testing/quick_start.py`
```python
# ANTES
import backend_api
import pi_simulator
import enrollment

# DESPUÉS
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.app import backend_api
from edge.src import pi_simulator
from enrollment import enrollment
```

## Actualización de Rutas de Archivos

### Base de datos

```python
# ANTES
DB_PATH = "gloria_stress_system.db"

# DESPUÉS
from pathlib import Path
project_root = Path(__file__).parent.parent  # Ajustar según ubicación
DB_PATH = project_root / "database" / "gloria_stress_system.db"
```

### Modelos ML

```python
# ANTES
MODEL_PATH = "emotion_detector.h5"

# DESPUÉS
from pathlib import Path
project_root = Path(__file__).parent.parent.parent  # Ajustar según ubicación
MODEL_PATH = project_root / "models" / "trained" / "emotion_detector.h5"
```

### Embeddings de Enrollment

```python
# ANTES
EMBEDDINGS_PATH = "enrollments/"

# DESPUÉS
from pathlib import Path
project_root = Path(__file__).parent.parent  # Ajustar según ubicación
EMBEDDINGS_PATH = project_root / "enrollment" / "data" / "enrollments"
```

### Logs

```python
# ANTES
LOG_FILE = "app.log"

# DESPUÉS
from pathlib import Path
project_root = Path(__file__).parent.parent  # Ajustar según ubicación
LOG_FILE = project_root / "logs" / "app" / "stressvision.log"
```

## Script de Actualización Automática de Imports

Puedes usar este script Python para ayudar a encontrar todos los imports que necesitan actualización:

```python
# check_imports.py
import os
from pathlib import Path

def find_python_files(directory):
    """Encuentra todos los archivos Python en el directorio"""
    for root, dirs, files in os.walk(directory):
        # Ignorar __pycache__ y venv
        dirs[:] = [d for d in dirs if d not in ['__pycache__', 'venv', 'env', '.git']]
        for file in files:
            if file.endswith('.py'):
                yield Path(root) / file

def check_old_imports(file_path):
    """Verifica si un archivo tiene imports que podrían necesitar actualización"""
    old_modules = [
        'backend_api',
        'pi_simulator', 
        'pi_config',
        'model_architecture',
        'model_trainer',
        'data_preparation',
        'evaluate_model',
        'train_model',
        'enrollment',
        'load_enrollments',
        'report_generator',
        'init_database',
        'quick_start'
    ]
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    found_imports = []
    for module in old_modules:
        if f'import {module}' in content or f'from {module}' in content:
            found_imports.append(module)
    
    return found_imports

def main():
    project_root = Path('.')
    print("Buscando archivos Python con imports antiguos...\n")
    
    issues_found = False
    for py_file in find_python_files(project_root):
        old_imports = check_old_imports(py_file)
        if old_imports:
            issues_found = True
            print(f"📄 {py_file}")
            for imp in old_imports:
                print(f"   ⚠️  import {imp}")
            print()
    
    if not issues_found:
        print("✓ No se encontraron imports antiguos")
    else:
        print("\nRevisa estos archivos y actualiza los imports según la guía.")

if __name__ == "__main__":
    main()
```

Guarda este script como `check_imports.py` en la raíz del proyecto y ejecútalo:

```bash
python check_imports.py
```

## Pasos de Verificación Post-Migración

1. **Ejecutar el script de verificación de imports**
   ```bash
   python check_imports.py
   ```

2. **Actualizar imports en archivos identificados**
   - Seguir las guías de esta documentación
   - Usar rutas relativas cuando sea posible
   - Agregar paths al sys.path cuando sea necesario

3. **Verificar rutas de archivos**
   - Base de datos
   - Modelos ML
   - Embeddings
   - Logs

4. **Probar cada módulo individualmente**
   ```bash
   # Backend
   cd backend
   python main.py
   
   # Edge
   cd edge
   python scripts/start_pi_system.py
   
   # Enrollment
   cd enrollment
   python enrollment.py --help
   
   # Training
   cd models/training/scripts
   python evaluate_model.py
   ```

5. **Ejecutar tests**
   ```bash
   python -m pytest tests/
   ```

6. **Probar el sistema completo**
   ```bash
   python backend/scripts/start_complete_system.py
   ```

## Alternativa: Configurar PYTHONPATH

En lugar de agregar `sys.path.insert()` en cada archivo, puedes configurar el PYTHONPATH:

### Linux/Mac
```bash
# En tu .bashrc o .zshrc
export PYTHONPATH="${PYTHONPATH}:/ruta/a/StressVision"

# O temporalmente
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python backend/main.py
```

### Windows
```powershell
# PowerShell
$env:PYTHONPATH = "$env:PYTHONPATH;C:\Users\unsaa\PycharmProjects\StressVision"
python backend/main.py

# CMD
set PYTHONPATH=%PYTHONPATH%;C:\Users\unsaa\PycharmProjects\StressVision
python backend/main.py
```

### PyCharm
1. File → Settings → Project → Project Structure
2. Marca la raíz del proyecto como "Sources Root"
3. PyCharm configurará automáticamente el PYTHONPATH

## Problemas Comunes y Soluciones

### Error: ModuleNotFoundError

```python
# Problema
ModuleNotFoundError: No module named 'backend_api'

# Solución
# Opción 1: Agregar proyecto root al path
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from backend.app import backend_api

# Opción 2: Usar imports relativos (si están en el mismo paquete)
from .backend_api import app

# Opción 3: Configurar PYTHONPATH
```

### Error: Archivo de base de datos no encontrado

```python
# Problema
sqlite3.OperationalError: unable to open database file

# Solución
from pathlib import Path

# Usar rutas absolutas basadas en la ubicación del script
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent  # Ajustar según profundidad
db_path = project_root / "database" / "gloria_stress_system.db"
```

### Error: Modelo no encontrado

```python
# Problema
FileNotFoundError: [Errno 2] No such file or directory: 'emotion_detector.h5'

# Solución  
from pathlib import Path

script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
model_path = project_root / "models" / "trained" / "emotion_detector.h5"
```

## Checklist de Migración

- [ ] Ejecutar script de migración de estructura
- [ ] Ejecutar `check_imports.py`
- [ ] Actualizar imports en archivos de backend/
- [ ] Actualizar imports en archivos de models/
- [ ] Actualizar imports en archivos de edge/
- [ ] Actualizar imports en archivos de enrollment/
- [ ] Actualizar imports en archivos de reporting/
- [ ] Actualizar rutas de base de datos
- [ ] Actualizar rutas de modelos ML
- [ ] Actualizar rutas de embeddings
- [ ] Actualizar rutas de logs
- [ ] Actualizar .gitignore
- [ ] Probar backend individualmente
- [ ] Probar edge individualmente
- [ ] Probar enrollment individualmente
- [ ] Probar sistema completo
- [ ] Ejecutar tests
- [ ] Commit de cambios

## Recursos Adicionales

- [Python Import System](https://docs.python.org/3/reference/import.html)
- [Python Packaging Guide](https://packaging.python.org/)
- [Structuring Your Project](https://docs.python-guide.org/writing/structure/)


