"""
Sistema de Entrenamiento - Stress Vision
Entrenamiento de modelos con callbacks, early stopping y métricas avanzadas

Autor: Gloria S.A.
Fecha: 2024
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json
from sklearn.metrics import confusion_matrix, classification_report


class ModelTrainer:
    def __init__(self, model, train_data, val_data, class_names):
        """
        Inicializa el sistema de entrenamiento.
        
        Args:
            model: Modelo de Keras compilado
            train_data: (X_train, y_train)
            val_data: (X_val, y_val)
            class_names: Lista de nombres de clases
        """
        self.model = model
        self.X_train, self.y_train = train_data
        self.X_val, self.y_val = val_data
        self.class_names = class_names
        self.history = None
        
        print(f"🎓 Trainer inicializado:")
        print(f"   • Train samples: {len(self.X_train)}")
        print(f"   • Val samples: {len(self.X_val)}")
        print(f"   • Clases: {class_names}")
    
    def create_callbacks(self, experiment_name=None):
        """
        Crea callbacks para entrenamiento.
        
        Args:
            experiment_name: Nombre del experimento (si None, usa timestamp)
            
        Returns:
            Lista de callbacks
        """
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Directorios
        checkpoint_dir = f"models/checkpoints/{experiment_name}"
        log_dir = f"logs/{experiment_name}"
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        print(f"\n📁 Directorios de experimento:")
        print(f"   • Checkpoints: {checkpoint_dir}")
        print(f"   • Logs: {log_dir}")
        
        callbacks = [
            # Early Stopping
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1,
                mode='min'
            ),
            
            # Model Checkpoint (guardar mejor modelo)
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"{checkpoint_dir}/best_model.keras",
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1,
                mode='max'
            ),
            
            # Model Checkpoint por época (backup)
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"{checkpoint_dir}/model_epoch_{{epoch:02d}}_acc_{{val_accuracy:.4f}}.keras",
                save_freq='epoch',
                save_best_only=False,
                verbose=0
            ),
            
            # Reduce Learning Rate on Plateau
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1,
                mode='min'
            ),
            
            # TensorBoard
            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True,
                write_images=False,
                update_freq='epoch',
                profile_batch=0
            ),
            
            # CSV Logger
            tf.keras.callbacks.CSVLogger(
                filename=f"{log_dir}/training_log.csv",
                separator=',',
                append=False
            ),
            
            # Callback custom para métricas adicionales
            CustomMetricsCallback(
                validation_data=(self.X_val, self.y_val),
                class_names=self.class_names,
                log_dir=log_dir
            )
        ]
        
        print(f"   ✓ {len(callbacks)} callbacks configurados")
        
        return callbacks
    
    def train(self, epochs=50, batch_size=32, experiment_name=None, 
              use_data_augmentation=True):
        """
        Entrena el modelo.
        
        Args:
            epochs: Número máximo de épocas
            batch_size: Tamaño del batch
            experiment_name: Nombre del experimento
            use_data_augmentation: Si aplicar data augmentation en training
            
        Returns:
            history: Historial de entrenamiento
        """
        print("\n" + "="*80)
        print(f"🚀 INICIANDO ENTRENAMIENTO")
        print("="*80)
        print(f"   • Épocas: {epochs}")
        print(f"   • Batch size: {batch_size}")
        print(f"   • Data augmentation: {use_data_augmentation}")
        print("="*80 + "\n")
        
        # Crear callbacks
        callbacks = self.create_callbacks(experiment_name)
        
        # Data augmentation (solo en training)
        if use_data_augmentation:
            print("🔄 Aplicando data augmentation en training...")
            
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            
            train_datagen = ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                brightness_range=[0.9, 1.1],
                zoom_range=0.05,
                fill_mode='nearest'
            )
            
            train_generator = train_datagen.flow(
                self.X_train, self.y_train,
                batch_size=batch_size,
                shuffle=True
            )
            
            steps_per_epoch = len(self.X_train) // batch_size
            
        else:
            train_generator = None
            steps_per_epoch = None
        
        # Entrenar
        start_time = datetime.now()
        print(f"⏰ Inicio: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        if train_generator:
            self.history = self.model.fit(
                train_generator,
                steps_per_epoch=steps_per_epoch,
                validation_data=(self.X_val, self.y_val),
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
        else:
            self.history = self.model.fit(
                self.X_train, self.y_train,
                validation_data=(self.X_val, self.y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n⏰ Fin: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  Duración: {duration}")
        
        return self.history
    
    def evaluate(self, test_data, plot_results=True):
        """
        Evalúa el modelo en set de prueba.
        
        Args:
            test_data: (X_test, y_test)
            plot_results: Si graficar resultados
            
        Returns:
            results: Dict con métricas
        """
        X_test, y_test = test_data
        
        print("\n" + "="*80)
        print("📊 EVALUACIÓN EN TEST SET")
        print("="*80)
        print(f"   • Test samples: {len(X_test)}")
        
        # Evaluar con métricas de Keras
        results = self.model.evaluate(X_test, y_test, verbose=1)
        
        # Extraer métricas
        metric_names = self.model.metrics_names
        results_dict = dict(zip(metric_names, results))
        
        print("\n📈 Métricas finales:")
        for name, value in results_dict.items():
            print(f"   • {name}: {value:.4f}")
        
        # Predicciones
        print("\n🔮 Generando predicciones...")
        y_pred_probs = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Classification report
        print("\n" + "="*80)
        print("CLASSIFICATION REPORT")
        print("="*80)
        print(classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            digits=4
        ))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        if plot_results:
            self.plot_confusion_matrix(cm, self.class_names)
        
        # Agregar métricas adicionales
        results_dict['confusion_matrix'] = cm.tolist()
        results_dict['classification_report'] = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            output_dict=True
        )
        
        return results_dict
    
    def plot_training_history(self, save_path=None):
        """
        Grafica historial de entrenamiento.
        
        Args:
            save_path: Path para guardar figura (si None, solo muestra)
        """
        if self.history is None:
            print("⚠️  No hay historial de entrenamiento disponible")
            return
        
        history = self.history.history
        
        # Crear figura con subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Historial de Entrenamiento', fontsize=16, fontweight='bold')
        
        # Loss
        ax = axes[0, 0]
        ax.plot(history['loss'], label='Train Loss', linewidth=2)
        ax.plot(history['val_loss'], label='Val Loss', linewidth=2)
        ax.set_xlabel('Época')
        ax.set_ylabel('Loss')
        ax.set_title('Loss durante Entrenamiento')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Accuracy
        ax = axes[0, 1]
        ax.plot(history['accuracy'], label='Train Accuracy', linewidth=2)
        ax.plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
        ax.set_xlabel('Época')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy durante Entrenamiento')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Precision
        if 'precision' in history:
            ax = axes[1, 0]
            ax.plot(history['precision'], label='Train Precision', linewidth=2)
            ax.plot(history['val_precision'], label='Val Precision', linewidth=2)
            ax.set_xlabel('Época')
            ax.set_ylabel('Precision')
            ax.set_title('Precision durante Entrenamiento')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Recall
        if 'recall' in history:
            ax = axes[1, 1]
            ax.plot(history['recall'], label='Train Recall', linewidth=2)
            ax.plot(history['val_recall'], label='Val Recall', linewidth=2)
            ax.set_xlabel('Época')
            ax.set_ylabel('Recall')
            ax.set_title('Recall durante Entrenamiento')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✓ Figura guardada en: {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, cm, class_names, save_path=None):
        """
        Grafica matriz de confusión.
        
        Args:
            cm: Matriz de confusión
            class_names: Nombres de las clases
            save_path: Path para guardar figura
        """
        plt.figure(figsize=(10, 8))
        
        # Normalizar por fila (recall)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Crear heatmap
        sns.heatmap(
            cm_norm,
            annot=cm,  # Mostrar counts
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Recall normalizado'},
            square=True
        )
        
        plt.title('Matriz de Confusión', fontsize=16, fontweight='bold')
        plt.ylabel('Clase Real', fontsize=12)
        plt.xlabel('Clase Predicha', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✓ Matriz de confusión guardada en: {save_path}")
        
        plt.show()


class CustomMetricsCallback(tf.keras.callbacks.Callback):
    """Callback para métricas custom durante entrenamiento."""
    
    def __init__(self, validation_data, class_names, log_dir):
        super().__init__()
        self.X_val, self.y_val = validation_data
        self.class_names = class_names
        self.log_dir = log_dir
        self.best_val_acc = 0.0
    
    def on_epoch_end(self, epoch, logs=None):
        """Al terminar cada época."""
        logs = logs or {}
        
        # Obtener accuracy de validación
        val_acc = logs.get('val_accuracy', 0)
        
        # Si es el mejor hasta ahora, guardar métricas adicionales
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            
            # Generar predicciones
            y_pred_probs = self.model.predict(self.X_val, verbose=0)
            y_pred = np.argmax(y_pred_probs, axis=1)
            y_true = np.argmax(self.y_val, axis=1)
            
            # Calcular métricas por clase
            from sklearn.metrics import precision_recall_fscore_support
            
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, average=None, labels=range(len(self.class_names))
            )
            
            # Guardar métricas
            metrics_per_class = {}
            for i, class_name in enumerate(self.class_names):
                metrics_per_class[class_name] = {
                    'precision': float(precision[i]),
                    'recall': float(recall[i]),
                    'f1': float(f1[i]),
                    'support': int(support[i])
                }
            
            # Guardar en JSON
            with open(f"{self.log_dir}/best_metrics_per_class.json", 'w') as f:
                json.dump(metrics_per_class, f, indent=2)
            
            print(f"\n   ✓ Nuevo mejor modelo (val_acc: {val_acc:.4f})")


def main():
    """Función principal de ejemplo."""
    
    print("\n" + "="*80)
    print("🎓 SISTEMA DE ENTRENAMIENTO - STRESS VISION")
    print("="*80)
    print("\nEste módulo debe ser importado desde train_model.py")
    print("o usado directamente con datos preparados.\n")
    print("Ejemplo de uso:")
    print("""
    from model_trainer import ModelTrainer
    from model_architecture import StressDetectionModel
    
    # Cargar datos
    train_data = np.load('data/processed/train_data.npz')
    val_data = np.load('data/processed/val_data.npz')
    
    # Construir modelo
    builder = StressDetectionModel()
    model = builder.build_model('mobilenetv3')
    model = builder.compile_model(model)
    
    # Entrenar
    trainer = ModelTrainer(
        model, 
        (train_data['images'], train_data['labels']),
        (val_data['images'], val_data['labels']),
        class_names=['neutral', 'stress', 'sad', 'happy', 'fatigue']
    )
    
    history = trainer.train(epochs=50, batch_size=32)
    trainer.plot_training_history()
    """)
    print("="*80 + "\n")


if __name__ == "__main__":
    main()


