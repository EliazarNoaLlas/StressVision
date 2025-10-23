"""
Arquitectura del Modelo - Stress Vision
Definición de arquitecturas optimizadas para detección de estrés

Modelos disponibles:
- MobileNetV3-Small: Transfer learning, alta precisión
- Custom Light: Ultra-ligero, máximo rendimiento en Raspberry Pi

Autor: Gloria S.A.
Fecha: 2024
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV3Small
import json


class StressDetectionModel:
    def __init__(self, num_classes=5, input_shape=(160, 160, 3)):
        """
        Inicializa el constructor de modelos.
        
        Args:
            num_classes: Número de clases de emoción (default: 5)
            input_shape: Forma del input (height, width, channels)
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        
        print(f"🏗️  Configuración del modelo:")
        print(f"   • Input shape: {input_shape}")
        print(f"   • Número de clases: {num_classes}")
        print(f"   • Target device: Raspberry Pi 5")
    
    def build_model(self, backbone='mobilenetv3', dropout_rate=0.3):
        """
        Construye modelo según backbone especificado.
        
        Args:
            backbone: 'mobilenetv3' o 'custom_light'
            dropout_rate: Tasa de dropout para regularización
            
        Returns:
            model: Modelo compilado de Keras
        """
        print(f"\n🔨 Construyendo modelo con backbone: {backbone}")
        
        if backbone == 'mobilenetv3':
            model = self._build_mobilenetv3(dropout_rate)
        elif backbone == 'custom_light':
            model = self._build_custom_light(dropout_rate)
        else:
            raise ValueError(f"Backbone desconocido: {backbone}")
        
        # Mostrar resumen
        print("\n📊 Resumen del modelo:")
        model.summary()
        
        # Calcular parámetros
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        print(f"\n📈 Parámetros:")
        print(f"   • Total: {total_params:,}")
        print(f"   • Entrenables: {trainable_params:,}")
        print(f"   • No entrenables: {non_trainable_params:,}")
        print(f"   • Tamaño estimado: {(total_params * 4) / (1024**2):.2f} MB (FP32)")
        
        return model
    
    def _build_mobilenetv3(self, dropout_rate=0.3):
        """
        Construye modelo basado en MobileNetV3-Small con transfer learning.
        
        Ventajas:
        - Pre-entrenado en ImageNet
        - Optimizado para dispositivos móviles
        - Buena precisión
        
        Args:
            dropout_rate: Tasa de dropout
            
        Returns:
            model: Modelo de Keras
        """
        print("   📥 Cargando MobileNetV3-Small pre-entrenado...")
        
        # Base pre-entrenada (sin top)
        base_model = MobileNetV3Small(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet',
            alpha=0.75,  # Reducir ancho para menor latencia
            minimalistic=False  # Usar squeeze-and-excitation para mejor precisión
        )
        
        # Fine-tuning: Congelar primeras capas
        print(f"   🔒 Congelando primeras capas (fine-tuning)...")
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        
        # Contar capas entrenables
        trainable_layers = sum([1 for layer in base_model.layers if layer.trainable])
        print(f"   ✓ Capas entrenables: {trainable_layers}/{len(base_model.layers)}")
        
        # Construir modelo completo
        inputs = layers.Input(shape=self.input_shape, name='input_image')
        
        # Preprocesamiento (normalización a [-1, 1])
        x = layers.Rescaling(1./127.5, offset=-1, name='preprocessing')(inputs)
        
        # Backbone
        x = base_model(x, training=False)
        
        # Custom head para detección de estrés
        x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
        
        # Dense layers con regularización
        x = layers.Dropout(dropout_rate, name='dropout_1')(x)
        x = layers.Dense(
            256, 
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            name='dense_1'
        )(x)
        x = layers.BatchNormalization(name='bn_1')(x)
        
        x = layers.Dropout(dropout_rate * 0.7, name='dropout_2')(x)
        x = layers.Dense(
            128,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            name='dense_2'
        )(x)
        x = layers.BatchNormalization(name='bn_2')(x)
        
        # Output layer
        outputs = layers.Dense(
            self.num_classes,
            activation='softmax',
            name='emotion_output'
        )(x)
        
        model = models.Model(inputs=inputs, outputs=outputs, name='stress_detector_mobilenetv3')
        
        return model
    
    def _build_custom_light(self, dropout_rate=0.3):
        """
        Construye arquitectura ultra-ligera custom para máximo rendimiento.
        
        Ventajas:
        - Mínimo número de parámetros
        - Latencia muy baja
        - Ideal para Raspberry Pi sin GPU
        
        Args:
            dropout_rate: Tasa de dropout
            
        Returns:
            model: Modelo de Keras
        """
        print("   🚀 Construyendo arquitectura ligera custom...")
        
        inputs = layers.Input(shape=self.input_shape, name='input_image')
        
        # Preprocesamiento
        x = layers.Rescaling(1./255.0, name='preprocessing')(inputs)
        
        # Bloque 1: Reducción inicial
        x = layers.Conv2D(
            32, 3, strides=2, padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            name='block1_conv'
        )(x)
        x = layers.BatchNormalization(name='block1_bn')(x)
        x = layers.ReLU(name='block1_relu')(x)
        x = layers.MaxPooling2D(2, name='block1_pool')(x)
        
        # Bloque 2: Separable convolutions (MobileNet-style)
        x = layers.SeparableConv2D(
            64, 3, padding='same',
            depthwise_regularizer=tf.keras.regularizers.l2(0.0001),
            pointwise_regularizer=tf.keras.regularizers.l2(0.0001),
            name='block2_sepconv'
        )(x)
        x = layers.BatchNormalization(name='block2_bn')(x)
        x = layers.ReLU(name='block2_relu')(x)
        x = layers.MaxPooling2D(2, name='block2_pool')(x)
        
        # Bloque 3
        x = layers.SeparableConv2D(
            128, 3, padding='same',
            depthwise_regularizer=tf.keras.regularizers.l2(0.0001),
            pointwise_regularizer=tf.keras.regularizers.l2(0.0001),
            name='block3_sepconv'
        )(x)
        x = layers.BatchNormalization(name='block3_bn')(x)
        x = layers.ReLU(name='block3_relu')(x)
        x = layers.MaxPooling2D(2, name='block3_pool')(x)
        
        # Bloque 4
        x = layers.SeparableConv2D(
            256, 3, padding='same',
            depthwise_regularizer=tf.keras.regularizers.l2(0.0001),
            pointwise_regularizer=tf.keras.regularizers.l2(0.0001),
            name='block4_sepconv'
        )(x)
        x = layers.BatchNormalization(name='block4_bn')(x)
        x = layers.ReLU(name='block4_relu')(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
        
        # Classifier
        x = layers.Dropout(dropout_rate, name='dropout')(x)
        x = layers.Dense(
            128,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            name='dense'
        )(x)
        
        # Output
        outputs = layers.Dense(
            self.num_classes,
            activation='softmax',
            name='emotion_output'
        )(x)
        
        model = models.Model(inputs=inputs, outputs=outputs, name='stress_detector_light')
        
        return model
    
    def compile_model(self, model, learning_rate=0.001, metrics=None):
        """
        Compila modelo con optimizer y métricas.
        
        Args:
            model: Modelo de Keras sin compilar
            learning_rate: Learning rate inicial
            metrics: Lista de métricas (si None, usa defaults)
            
        Returns:
            model: Modelo compilado
        """
        print(f"\n⚙️  Compilando modelo...")
        print(f"   • Optimizer: Adam (lr={learning_rate})")
        print(f"   • Loss: Categorical Crossentropy")
        
        if metrics is None:
            metrics = [
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')
            ]
        
        print(f"   • Métricas: {[m if isinstance(m, str) else m.name for m in metrics]}")
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=metrics
        )
        
        print("   ✓ Modelo compilado")
        
        return model
    
    def save_model_config(self, model, output_path):
        """
        Guarda configuración del modelo en JSON.
        
        Args:
            model: Modelo de Keras
            output_path: Path al archivo JSON de salida
        """
        config = {
            'model_name': model.name,
            'input_shape': list(self.input_shape),
            'num_classes': self.num_classes,
            'total_params': int(model.count_params()),
            'trainable_params': int(sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])),
            'layers': []
        }
        
        # Agregar info de cada capa
        for layer in model.layers:
            layer_config = {
                'name': layer.name,
                'type': layer.__class__.__name__,
                'trainable': layer.trainable
            }
            
            # Agregar output shape si está disponible
            try:
                layer_config['output_shape'] = [int(d) if d is not None else None 
                                                for d in layer.output_shape]
            except:
                pass
            
            config['layers'].append(layer_config)
        
        # Guardar
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"   ✓ Configuración guardada en: {output_path}")


def main():
    """Función principal para probar arquitecturas."""
    
    print("\n" + "="*80)
    print("🏗️  ARQUITECTURA DEL MODELO - STRESS VISION")
    print("="*80)
    print("\nGloria S.A. - Sistema de Detección de Estrés Laboral")
    print("Fase 4: Entrenamiento y Optimización del Modelo\n")
    print("="*80)
    
    # Seleccionar backbone
    print("\nArquitecturas disponibles:")
    print("1. MobileNetV3-Small (Transfer Learning)")
    print("   • Precisión: Alta")
    print("   • Velocidad: Media")
    print("   • Tamaño: ~4-6 MB")
    print("   • Recomendado para: Raspberry Pi 5 con buen performance")
    print("\n2. Custom Light (Ultra-ligero)")
    print("   • Precisión: Media-Alta")
    print("   • Velocidad: Muy Alta")
    print("   • Tamaño: ~1-2 MB")
    print("   • Recomendado para: Máximo rendimiento, latencia mínima")
    
    opcion = input("\nSeleccione arquitectura (1/2) [1]: ").strip()
    
    if opcion == '2':
        backbone = 'custom_light'
    else:
        backbone = 'mobilenetv3'
    
    # Configurar dropout
    dropout = input("\nDropout rate [0.3]: ").strip()
    dropout_rate = float(dropout) if dropout else 0.3
    
    # Construir modelo
    builder = StressDetectionModel(num_classes=5, input_shape=(160, 160, 3))
    model = builder.build_model(backbone=backbone, dropout_rate=dropout_rate)
    
    # Compilar
    lr = input("\nLearning rate [0.001]: ").strip()
    learning_rate = float(lr) if lr else 0.001
    
    model = builder.compile_model(model, learning_rate=learning_rate)
    
    # Guardar configuración
    import os
    os.makedirs("models", exist_ok=True)
    
    config_path = f"models/{model.name}_config.json"
    builder.save_model_config(model, config_path)
    
    # Guardar arquitectura
    model_path = f"models/{model.name}_architecture.keras"
    model.save(model_path)
    print(f"   ✓ Arquitectura guardada en: {model_path}")
    
    print("\n" + "="*80)
    print("✅ MODELO CREADO EXITOSAMENTE")
    print("="*80)
    print("\n💡 Próximos pasos:")
    print("   1. Preparar datos: python data_preparation.py")
    print("   2. Entrenar modelo: python train_model.py")
    print("   3. Evaluar: python evaluate_model.py")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()


