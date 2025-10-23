"""
Script Principal de Entrenamiento - Stress Vision
Pipeline completo de entrenamiento del modelo de detección de estrés

Flujo:
1. Cargar datos procesados
2. Construir modelo
3. Entrenar con callbacks
4. Evaluar en test set
5. Guardar modelo final
6. Convertir a TFLite (opcional)

Autor: Gloria S.A.
Fecha: 2024
"""

import os
import numpy as np
import json
from datetime import datetime

# Importar módulos del proyecto
from model_architecture import StressDetectionModel
from model_trainer import ModelTrainer
from convert_to_tflite import TFLiteConverter


def load_processed_data(data_dir='data/processed'):
    """
    Carga datos procesados desde archivos NPZ.
    
    Args:
        data_dir: Directorio con datos procesados
        
    Returns:
        train_data, val_data, test_data, metadata
    """
    print("\n" + "="*80)
    print("📥 CARGANDO DATOS PROCESADOS")
    print("="*80)
    print(f"   Directorio: {data_dir}\n")
    
    # Verificar que existen los archivos
    required_files = ['train_data.npz', 'val_data.npz', 'test_data.npz', 'metadata.json']
    for filename in required_files:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Archivo requerido no encontrado: {filepath}")
    
    # Cargar datos
    train_data = np.load(os.path.join(data_dir, 'train_data.npz'))
    val_data = np.load(os.path.join(data_dir, 'val_data.npz'))
    test_data = np.load(os.path.join(data_dir, 'test_data.npz'))
    
    # Cargar metadata
    with open(os.path.join(data_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    print(f"✅ Datos cargados:")
    print(f"   • Train: {len(train_data['images'])} imágenes")
    print(f"   • Validation: {len(val_data['images'])} imágenes")
    print(f"   • Test: {len(test_data['images'])} imágenes")
    print(f"   • Clases: {metadata['target_emotions']}")
    print("="*80)
    
    return (
        (train_data['images'], train_data['labels']),
        (val_data['images'], val_data['labels']),
        (test_data['images'], test_data['labels']),
        metadata
    )


def create_experiment_config(backbone, learning_rate, epochs, batch_size, 
                             use_augmentation, dropout_rate):
    """
    Crea configuración del experimento.
    
    Returns:
        config: Dict con configuración
    """
    config = {
        'experiment_name': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'timestamp': datetime.now().isoformat(),
        'model': {
            'backbone': backbone,
            'dropout_rate': dropout_rate,
            'input_shape': [160, 160, 3],
            'num_classes': 5
        },
        'training': {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'use_augmentation': use_augmentation
        },
        'target_device': 'Raspberry Pi 5',
        'objective': 'Detección de estrés laboral',
        'kpis': {
            'accuracy_target': 0.84,
            'latency_target_ms': 200
        }
    }
    
    return config


def save_experiment_results(config, history, test_results, output_dir):
    """
    Guarda resultados del experimento.
    
    Args:
        config: Configuración del experimento
        history: Historial de entrenamiento
        test_results: Resultados en test set
        output_dir: Directorio de salida
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Agregar resultados a config
    config['results'] = {
        'final_train_accuracy': float(history.history['accuracy'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1]),
        'best_val_accuracy': float(max(history.history['val_accuracy'])),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'epochs_trained': len(history.history['loss']),
        'test_metrics': test_results
    }
    
    # Guardar configuración y resultados
    with open(os.path.join(output_dir, 'experiment_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"   ✓ Configuración guardada en: {output_dir}/experiment_config.json")
    
    # Guardar historial de entrenamiento
    history_dict = {key: [float(v) for v in values] 
                   for key, values in history.history.items()}
    
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    print(f"   ✓ Historial guardado en: {output_dir}/training_history.json")


def main():
    """Función principal de entrenamiento."""
    
    print("\n" + "="*100)
    print(" "*30 + "🤖 ENTRENAMIENTO DEL MODELO")
    print(" "*30 + "STRESS VISION - GLORIA S.A.")
    print("="*100)
    print("\nFase 4: Entrenamiento y Optimización del Modelo")
    print("Objetivo: Detectar estrés laboral con ≥84% accuracy y ≤200ms latencia")
    print("\n" + "="*100)
    
    # ========================================
    # 1. CARGAR DATOS
    # ========================================
    try:
        train_data, val_data, test_data, metadata = load_processed_data('data/processed')
        class_names = metadata['target_emotions']
        num_classes = len(class_names)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Primero debe preparar los datos:")
        print("   python data_preparation.py")
        return
    
    # ========================================
    # 2. CONFIGURAR ENTRENAMIENTO
    # ========================================
    print("\n" + "="*100)
    print("⚙️  CONFIGURACIÓN DEL ENTRENAMIENTO")
    print("="*100)
    
    print("\n📋 Opciones de arquitectura:")
    print("1. MobileNetV3-Small (Recomendado)")
    print("   • Transfer learning de ImageNet")
    print("   • ~4-6 MB")
    print("   • Alta precisión")
    print("\n2. Custom Light")
    print("   • Arquitectura custom ultra-ligera")
    print("   • ~1-2 MB")
    print("   • Máxima velocidad")
    
    backbone_opt = input("\nSeleccione arquitectura (1/2) [1]: ").strip()
    backbone = 'custom_light' if backbone_opt == '2' else 'mobilenetv3'
    
    # Hiperparámetros
    print("\n📊 Hiperparámetros:")
    
    epochs = input("Épocas máximas [50]: ").strip()
    epochs = int(epochs) if epochs else 50
    
    batch_size = input("Batch size [32]: ").strip()
    batch_size = int(batch_size) if batch_size else 32
    
    lr = input("Learning rate [0.001]: ").strip()
    learning_rate = float(lr) if lr else 0.001
    
    dropout = input("Dropout rate [0.3]: ").strip()
    dropout_rate = float(dropout) if dropout else 0.3
    
    aug = input("\n¿Usar data augmentation en training? (s/n) [s]: ").strip().lower()
    use_augmentation = aug != 'n'
    
    # Crear configuración
    config = create_experiment_config(
        backbone, learning_rate, epochs, batch_size, 
        use_augmentation, dropout_rate
    )
    
    experiment_name = config['experiment_name']
    
    print(f"\n✅ Configuración completa")
    print(f"   • Experimento: {experiment_name}")
    print(f"   • Backbone: {backbone}")
    print(f"   • Épocas: {epochs}")
    print(f"   • Batch size: {batch_size}")
    print(f"   • Learning rate: {learning_rate}")
    print(f"   • Dropout: {dropout_rate}")
    print(f"   • Data augmentation: {use_augmentation}")
    
    # ========================================
    # 3. CONSTRUIR MODELO
    # ========================================
    print("\n" + "="*100)
    print("🏗️  CONSTRUCCIÓN DEL MODELO")
    print("="*100)
    
    model_builder = StressDetectionModel(
        num_classes=num_classes,
        input_shape=(160, 160, 3)
    )
    
    model = model_builder.build_model(
        backbone=backbone,
        dropout_rate=dropout_rate
    )
    
    model = model_builder.compile_model(
        model,
        learning_rate=learning_rate
    )
    
    # ========================================
    # 4. ENTRENAR MODELO
    # ========================================
    print("\n" + "="*100)
    print("🚀 ENTRENAMIENTO")
    print("="*100)
    
    trainer = ModelTrainer(
        model=model,
        train_data=train_data,
        val_data=val_data,
        class_names=class_names
    )
    
    history = trainer.train(
        epochs=epochs,
        batch_size=batch_size,
        experiment_name=experiment_name,
        use_data_augmentation=use_augmentation
    )
    
    # ========================================
    # 5. EVALUAR EN TEST SET
    # ========================================
    print("\n" + "="*100)
    print("📊 EVALUACIÓN FINAL")
    print("="*100)
    
    test_results = trainer.evaluate(test_data, plot_results=True)
    
    # Verificar KPIs
    print("\n" + "="*100)
    print("🎯 VERIFICACIÓN DE KPIs")
    print("="*100)
    
    test_accuracy = test_results['accuracy']
    target_accuracy = 0.84
    
    print(f"\n📈 Accuracy:")
    print(f"   • Test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"   • Target: {target_accuracy:.4f} ({target_accuracy*100:.2f}%)")
    
    if test_accuracy >= target_accuracy:
        print(f"   ✅ CUMPLE KPI de accuracy")
    else:
        diff = (target_accuracy - test_accuracy) * 100
        print(f"   ⚠️  Falta {diff:.2f}% para cumplir KPI")
        print(f"   💡 Sugerencias:")
        print(f"      • Aumentar dataset (data augmentation)")
        print(f"      • Ajustar learning rate")
        print(f"      • Probar MobileNetV3 si usó Custom Light")
        print(f"      • Aumentar epochs")
    
    # ========================================
    # 6. GUARDAR MODELO Y RESULTADOS
    # ========================================
    print("\n" + "="*100)
    print("💾 GUARDANDO RESULTADOS")
    print("="*100)
    
    output_dir = f"models/experiments/{experiment_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar modelo final
    final_model_path = f"{output_dir}/final_model.keras"
    model.save(final_model_path)
    print(f"   ✓ Modelo guardado: {final_model_path}")
    
    # Guardar configuración del modelo
    model_builder.save_model_config(model, f"{output_dir}/model_config.json")
    
    # Guardar resultados del experimento
    save_experiment_results(config, history, test_results, output_dir)
    
    # Graficar historial
    plot_path = f"{output_dir}/training_history.png"
    trainer.plot_training_history(save_path=plot_path)
    
    # ========================================
    # 7. CONVERSIÓN A TFLITE (OPCIONAL)
    # ========================================
    print("\n" + "="*100)
    convert_opt = input("\n¿Convertir a TensorFlow Lite ahora? (s/n) [s]: ").strip().lower()
    
    if convert_opt != 'n':
        print("\n🔄 CONVERSIÓN A TENSORFLOW LITE")
        print("="*100)
        
        converter = TFLiteConverter(final_model_path)
        
        tflite_dir = f"{output_dir}/tflite"
        os.makedirs(tflite_dir, exist_ok=True)
        
        # Convertir a INT8 (recomendado para Raspberry Pi)
        print("\nConvirtiendo a INT8 quantized...")
        tflite_path = converter.convert_int8(
            output_path=f"{tflite_dir}/model_int8.tflite"
        )
        
        if tflite_path:
            # Benchmark
            print("\n⚡ Ejecutando benchmark...")
            latency_stats = converter.benchmark_latency(tflite_path, num_runs=100)
            
            # Guardar stats
            with open(f"{tflite_dir}/latency_stats.json", 'w') as f:
                json.dump(latency_stats, f, indent=2)
            
            print(f"\n✅ Modelo TFLite guardado en: {tflite_dir}/")
    
    # ========================================
    # 8. RESUMEN FINAL
    # ========================================
    print("\n" + "="*100)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("="*100)
    
    print(f"\n📊 Resumen del Experimento:")
    print(f"   • Experimento ID: {experiment_name}")
    print(f"   • Modelo: {backbone}")
    print(f"   • Épocas entrenadas: {len(history.history['loss'])}")
    print(f"   • Mejor val accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"   • Test accuracy: {test_accuracy:.4f}")
    print(f"   • Test loss: {test_results['loss']:.4f}")
    
    if 'precision' in test_results:
        print(f"   • Precision: {test_results['precision']:.4f}")
        print(f"   • Recall: {test_results['recall']:.4f}")
    
    print(f"\n📁 Archivos guardados en: {output_dir}/")
    print(f"   • final_model.keras")
    print(f"   • model_config.json")
    print(f"   • experiment_config.json")
    print(f"   • training_history.json")
    print(f"   • training_history.png")
    if convert_opt != 'n':
        print(f"   • tflite/model_int8.tflite")
        print(f"   • tflite/latency_stats.json")
    
    print(f"\n💡 Próximos pasos:")
    print(f"   1. Revisar métricas en TensorBoard:")
    print(f"      tensorboard --logdir logs/{experiment_name}")
    print(f"   2. Evaluar modelo: python evaluate_model.py")
    print(f"   3. Desplegar en Raspberry Pi")
    
    print("\n" + "="*100 + "\n")


if __name__ == "__main__":
    main()


