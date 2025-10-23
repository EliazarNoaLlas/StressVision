"""
Configuración del Simulador de Raspberry Pi
Archivo de configuración centralizado

Autor: Gloria S.A.
Fecha: 2024
"""

import os
from pathlib import Path


def get_config(device_id='pi-simulator-01'):
    """
    Retorna configuración del simulador.
    
    Args:
        device_id: ID único del dispositivo simulado
        
    Returns:
        dict: Configuración completa
    """
    
    # Detectar paths automáticamente
    base_dir = Path(__file__).parent
    
    # Buscar modelos TFLite
    models_dir = base_dir / "models"
    
    # Buscar modelo de emociones
    emotion_model_candidates = [
        models_dir / "tflite" / "model_int8.tflite",
        models_dir / "experiments" / "*" / "tflite" / "model_int8.tflite",
        models_dir / "stress_detector_quantized.tflite",
    ]
    
    emotion_model_path = None
    for candidate in emotion_model_candidates:
        if "*" in str(candidate):
            # Glob pattern
            import glob
            matches = glob.glob(str(candidate))
            if matches:
                emotion_model_path = matches[0]
                break
        elif Path(candidate).exists():
            emotion_model_path = str(candidate)
            break
    
    # Buscar modelo de embeddings faciales (opcional)
    face_model_path = None
    face_model_candidates = [
        models_dir / "facenet_mobilenet.tflite",
        models_dir / "face_embedder.tflite",
    ]
    
    for candidate in face_model_candidates:
        if Path(candidate).exists():
            face_model_path = str(candidate)
            break
    
    # Base de datos
    db_path = str(base_dir / "gloria_stress_system.db")
    
    config = {
        # Identificación del dispositivo
        'device_id': device_id,
        'location': 'Oficina Principal - Simulador',
        
        # Servidor (local simulado)
        'server_url': 'http://localhost:5000',
        
        # Cámara
        'camera_index': 0,  # 0 = cámara por defecto
        
        # Performance
        'frame_skip': 3,  # Procesar 1 de cada 3 frames (simular Pi)
        'show_preview': True,  # Mostrar ventana con detecciones
        
        # Modelos
        'emotion_model_path': emotion_model_path,
        'face_model_path': face_model_path,
        
        # Base de datos
        'db_path': db_path,
        
        # Reconocimiento facial
        'recognition_threshold': 0.6,  # Similitud mínima para match
        
        # Logging
        'log_detections': True,  # Guardar detecciones en archivos locales
        'log_dir': str(base_dir / "logs"),
        
        # Temporización
        'detection_cooldown': 2.0,  # Segundos entre detecciones del mismo empleado
    }
    
    return config


def print_config(config):
    """Imprime configuración de forma legible."""
    
    print("\n" + "="*80)
    print("⚙️  CONFIGURACIÓN DEL SIMULADOR")
    print("="*80)
    
    print(f"\n📍 Identificación:")
    print(f"   • Device ID: {config['device_id']}")
    print(f"   • Location: {config['location']}")
    
    print(f"\n🌐 Servidor:")
    print(f"   • URL: {config['server_url']}")
    
    print(f"\n📹 Cámara:")
    print(f"   • Index: {config['camera_index']}")
    print(f"   • Frame skip: {config['frame_skip']} (procesa 1 de cada {config['frame_skip']})")
    print(f"   • Preview: {'Sí' if config['show_preview'] else 'No'}")
    
    print(f"\n🤖 Modelos:")
    if config['emotion_model_path']:
        print(f"   • Emociones: ✅ {config['emotion_model_path']}")
    else:
        print(f"   • Emociones: ⚠️  No encontrado (usará mock)")
    
    if config['face_model_path']:
        print(f"   • Embeddings: ✅ {config['face_model_path']}")
    else:
        print(f"   • Embeddings: ⚠️  No encontrado (usará mock)")
    
    print(f"\n🗄️  Base de Datos:")
    if os.path.exists(config['db_path']):
        print(f"   • Path: ✅ {config['db_path']}")
    else:
        print(f"   • Path: ⚠️  {config['db_path']} (no existe)")
    
    print(f"\n🔧 Parámetros:")
    print(f"   • Recognition threshold: {config['recognition_threshold']}")
    print(f"   • Detection cooldown: {config['detection_cooldown']}s")
    print(f"   • Log detections: {'Sí' if config['log_detections'] else 'No'}")
    
    print("="*80 + "\n")


def create_default_config_file(filename='pi_config_custom.json'):
    """Crea archivo de configuración personalizado en JSON."""
    
    import json
    
    config = get_config()
    
    # Convertir Paths a strings para JSON
    config_json = {}
    for key, value in config.items():
        if isinstance(value, Path):
            config_json[key] = str(value)
        else:
            config_json[key] = value
    
    with open(filename, 'w') as f:
        json.dump(config_json, f, indent=2)
    
    print(f"✅ Configuración guardada en: {filename}")
    print(f"   Puedes editarla y cargarla con load_config('{filename}')")


def load_config_from_file(filename):
    """Carga configuración desde archivo JSON."""
    
    import json
    
    with open(filename, 'r') as f:
        config = json.load(f)
    
    return config


if __name__ == "__main__":
    # Test de configuración
    config = get_config()
    print_config(config)
    
    # Opción de crear archivo personalizado
    create = input("\n¿Crear archivo de configuración personalizado? (s/n): ").strip().lower()
    if create == 's':
        filename = input("Nombre del archivo [pi_config_custom.json]: ").strip()
        if not filename:
            filename = 'pi_config_custom.json'
        
        create_default_config_file(filename)


