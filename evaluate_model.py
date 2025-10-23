"""
Evaluación del Modelo - Stress Vision
Script para evaluar modelos entrenados y generar reportes detallados

Autor: Gloria S.A.
Fecha: 2024
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json


def load_model_and_metadata(experiment_dir):
    """Carga modelo y metadata del experimento."""
    model_path = os.path.join(experiment_dir, 'final_model.keras')
    config_path = os.path.join(experiment_dir, 'experiment_config.json')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    
    model = tf.keras.models.load_model(model_path)
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = None
    
    return model, config


def evaluate_comprehensive(model, test_data, class_names, output_dir):
    """Evaluación comprehensiva del modelo."""
    X_test, y_test = test_data
    
    print("\n" + "="*80)
    print("📊 EVALUACIÓN COMPREHENSIVA")
    print("="*80)
    
    # 1. Métricas globales
    print("\n1️⃣  Métricas Globales")
    results = model.evaluate(X_test, y_test, verbose=1)
    metric_names = model.metrics_names
    
    metrics_dict = {}
    for name, value in zip(metric_names, results):
        metrics_dict[name] = float(value)
        print(f"   • {name}: {value:.4f}")
    
    # 2. Predicciones
    print("\n2️⃣  Generando predicciones...")
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # 3. Classification Report
    print("\n3️⃣  Classification Report por Clase:")
    print("-" * 80)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)
    
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # 4. Confusion Matrix
    print("\n4️⃣  Matriz de Confusión")
    cm = confusion_matrix(y_true, y_pred)
    
    # Plotear
    plt.figure(figsize=(10, 8))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Recall'}, square=True)
    
    plt.title('Matriz de Confusión', fontsize=14, fontweight='bold')
    plt.ylabel('Clase Real')
    plt.xlabel('Clase Predicha')
    plt.tight_layout()
    
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Guardada en: {cm_path}")
    plt.close()
    
    # 5. Análisis de errores
    print("\n5️⃣  Análisis de Errores")
    errors = np.where(y_pred != y_true)[0]
    print(f"   • Total errores: {len(errors)} ({len(errors)/len(y_test)*100:.2f}%)")
    
    # Errores más comunes
    error_pairs = []
    for idx in errors:
        error_pairs.append((class_names[y_true[idx]], class_names[y_pred[idx]]))
    
    from collections import Counter
    top_errors = Counter(error_pairs).most_common(5)
    
    print("   • Top 5 confusiones:")
    for (true_class, pred_class), count in top_errors:
        print(f"     - {true_class} → {pred_class}: {count} veces")
    
    # Guardar resultados
    results_dict = {
        'global_metrics': metrics_dict,
        'classification_report': report_dict,
        'confusion_matrix': cm.tolist(),
        'total_errors': int(len(errors)),
        'error_rate': float(len(errors)/len(y_test)),
        'top_errors': [(true_c, pred_c, int(cnt)) for (true_c, pred_c), cnt in top_errors]
    }
    
    results_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n💾 Resultados guardados en: {results_path}")
    
    return results_dict


def main():
    """Función principal."""
    print("\n" + "="*80)
    print("📊 EVALUACIÓN DE MODELO - STRESS VISION")
    print("="*80)
    
    # Seleccionar experimento
    experiments_dir = "models/experiments"
    
    if not os.path.exists(experiments_dir):
        print(f"\n❌ Directorio no encontrado: {experiments_dir}")
        return
    
    experiments = [d for d in os.listdir(experiments_dir) 
                  if os.path.isdir(os.path.join(experiments_dir, d))]
    
    if not experiments:
        print(f"\n❌ No hay experimentos en: {experiments_dir}")
        return
    
    print(f"\n📋 Experimentos disponibles:")
    for i, exp in enumerate(experiments, 1):
        print(f"   {i}. {exp}")
    
    choice = input("\nSeleccione experimento (número): ").strip()
    try:
        exp_idx = int(choice) - 1
        experiment_name = experiments[exp_idx]
    except:
        print("❌ Selección inválida")
        return
    
    experiment_dir = os.path.join(experiments_dir, experiment_name)
    
    # Cargar modelo
    print(f"\n📥 Cargando experimento: {experiment_name}")
    model, config = load_model_and_metadata(experiment_dir)
    
    # Cargar datos de test
    print("\n📥 Cargando datos de test...")
    test_data = np.load('data/processed/test_data.npz')
    X_test = test_data['images']
    y_test = test_data['labels']
    
    # Cargar metadata
    with open('data/processed/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    class_names = metadata['target_emotions']
    
    print(f"   • Test samples: {len(X_test)}")
    print(f"   • Clases: {class_names}")
    
    # Evaluar
    output_dir = f"{experiment_dir}/evaluation"
    os.makedirs(output_dir, exist_ok=True)
    
    results = evaluate_comprehensive(
        model, (X_test, y_test), class_names, output_dir
    )
    
    print("\n" + "="*80)
    print("✅ EVALUACIÓN COMPLETADA")
    print("="*80)
    print(f"\n📁 Resultados en: {output_dir}/")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()


