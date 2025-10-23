"""
Preparación de Datos - Stress Vision
Sistema de carga, procesamiento y augmentación de datasets para entrenamiento

Datasets soportados:
- FER-2013 (Facial Expression Recognition)
- AffectNet (opcional)
- Dataset custom de Gloria S.A.

Autor: Gloria S.A.
Fecha: 2024
"""

import os
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import json


class StressDatasetBuilder:
    def __init__(self, base_datasets=None, custom_dataset_path=None):
        """
        Inicializa el constructor de dataset.
        
        Args:
            base_datasets: dict con paths a datasets públicos
                Ejemplo: {'fer2013': 'data/fer2013/fer2013.csv'}
            custom_dataset_path: path a imágenes propias de Gloria
        """
        self.base_datasets = base_datasets or {}
        self.custom_dataset_path = custom_dataset_path
        
        # Mapeo de emociones a categorías de estrés
        self.emotion_map = {
            'angry': 'stress',      # Enojo → Estrés
            'fear': 'stress',       # Miedo → Estrés
            'disgust': 'stress',    # Disgusto → Estrés
            'sad': 'sad',           # Tristeza
            'neutral': 'neutral',   # Neutral
            'happy': 'happy',       # Felicidad
            'surprise': 'neutral'   # Sorpresa → Neutral
        }
        
        # Categorías objetivo para el modelo
        self.target_emotions = ['neutral', 'stress', 'sad', 'happy', 'fatigue']
        
        # Estadísticas del dataset
        self.stats = {
            'total_images': 0,
            'per_class': {},
            'sources': {}
        }
        
    def load_fer2013(self, csv_path):
        """
        Carga dataset FER-2013.
        
        Formato esperado del CSV:
        emotion,pixels,Usage
        0,"1 2 3 ... 2304",Training
        
        Args:
            csv_path: Path al archivo fer2013.csv
            
        Returns:
            images: Array de imágenes (N, 160, 160, 3)
            labels: Array de etiquetas string
        """
        print(f"\n📥 Cargando FER-2013 desde: {csv_path}")
        
        if not os.path.exists(csv_path):
            print(f"⚠️  Archivo no encontrado: {csv_path}")
            print("   Descargue FER-2013 desde: https://www.kaggle.com/datasets/msambare/fer2013")
            return np.array([]), np.array([])
        
        try:
            df = pd.read_csv(csv_path)
            
            # Mapeo de índices FER-2013 a nombres de emociones
            fer_emotions = {
                0: 'angry',
                1: 'disgust',
                2: 'fear',
                3: 'happy',
                4: 'sad',
                5: 'surprise',
                6: 'neutral'
            }
            
            images = []
            labels = []
            
            print("⏳ Procesando imágenes...")
            for idx, row in df.iterrows():
                try:
                    # Convertir string de pixels a array
                    pixels = np.array(row['pixels'].split(), dtype=np.uint8)
                    
                    # Reshape a imagen 48x48
                    img = pixels.reshape(48, 48)
                    
                    # Redimensionar a 160x160 (input size del modelo)
                    img = cv2.resize(img, (160, 160))
                    
                    # Convertir a RGB (DeepFace requiere 3 canales)
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    
                    # Mapear emoción
                    fer_emotion = fer_emotions.get(row['emotion'], 'neutral')
                    mapped_emotion = self.emotion_map.get(fer_emotion, 'neutral')
                    
                    # Solo agregar si está en categorías objetivo
                    if mapped_emotion in self.target_emotions:
                        images.append(img)
                        labels.append(mapped_emotion)
                        
                except Exception as e:
                    if idx % 1000 == 0:
                        print(f"  ⚠️  Error en imagen {idx}: {e}")
                    continue
                
                # Progress
                if (idx + 1) % 5000 == 0:
                    print(f"  Procesadas: {idx + 1}/{len(df)}")
            
            images = np.array(images)
            labels = np.array(labels)
            
            print(f"✅ FER-2013 cargado: {len(images)} imágenes")
            
            # Actualizar estadísticas
            self.stats['sources']['fer2013'] = len(images)
            
            return images, labels
            
        except Exception as e:
            print(f"❌ Error al cargar FER-2013: {e}")
            return np.array([]), np.array([])
    
    def load_custom_dataset(self):
        """
        Carga dataset custom de Gloria S.A.
        
        Estructura esperada:
        custom_dataset/
        ├── neutral/
        │   ├── img1.jpg
        │   └── img2.jpg
        ├── stress/
        ├── sad/
        ├── happy/
        └── fatigue/
        
        Returns:
            images: Array de imágenes
            labels: Array de etiquetas
        """
        if not self.custom_dataset_path:
            return np.array([]), np.array([])
        
        print(f"\n📥 Cargando dataset custom desde: {self.custom_dataset_path}")
        
        if not os.path.exists(self.custom_dataset_path):
            print(f"⚠️  Directorio no encontrado: {self.custom_dataset_path}")
            return np.array([]), np.array([])
        
        images = []
        labels = []
        
        for emotion in self.target_emotions:
            emotion_path = os.path.join(self.custom_dataset_path, emotion)
            
            if not os.path.exists(emotion_path):
                print(f"  ⚠️  No encontrado: {emotion}/")
                continue
            
            # Obtener todas las imágenes
            image_files = list(Path(emotion_path).glob('*.jpg')) + \
                         list(Path(emotion_path).glob('*.png')) + \
                         list(Path(emotion_path).glob('*.jpeg'))
            
            print(f"  📁 {emotion}: {len(image_files)} imágenes")
            
            for img_path in image_files:
                try:
                    # Leer imagen
                    img = cv2.imread(str(img_path))
                    
                    if img is None:
                        continue
                    
                    # Redimensionar
                    img = cv2.resize(img, (160, 160))
                    
                    # Convertir BGR a RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    images.append(img)
                    labels.append(emotion)
                    
                except Exception as e:
                    print(f"    ⚠️  Error en {img_path.name}: {e}")
                    continue
        
        images = np.array(images)
        labels = np.array(labels)
        
        print(f"✅ Dataset custom cargado: {len(images)} imágenes")
        
        # Actualizar estadísticas
        if len(images) > 0:
            self.stats['sources']['custom'] = len(images)
        
        return images, labels
    
    def load_and_merge_datasets(self):
        """
        Carga y combina todos los datasets disponibles.
        
        Returns:
            images: Array combinado de imágenes
            labels: Array combinado de etiquetas
        """
        print("\n" + "="*80)
        print("📊 CARGANDO Y COMBINANDO DATASETS")
        print("="*80)
        
        all_images = []
        all_labels = []
        
        # Cargar FER-2013 si está disponible
        if 'fer2013' in self.base_datasets:
            fer_images, fer_labels = self.load_fer2013(self.base_datasets['fer2013'])
            if len(fer_images) > 0:
                all_images.append(fer_images)
                all_labels.append(fer_labels)
        
        # Cargar dataset custom
        custom_images, custom_labels = self.load_custom_dataset()
        if len(custom_images) > 0:
            all_images.append(custom_images)
            all_labels.append(custom_labels)
        
        if len(all_images) == 0:
            print("\n❌ No se cargaron imágenes de ningún dataset")
            return np.array([]), np.array([])
        
        # Combinar todos los datasets
        images = np.concatenate(all_images, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
        # Shuffle
        images, labels = shuffle(images, labels, random_state=42)
        
        # Estadísticas
        self.stats['total_images'] = len(images)
        
        print("\n" + "="*80)
        print("📊 ESTADÍSTICAS DEL DATASET COMBINADO")
        print("="*80)
        print(f"Total de imágenes: {len(images)}")
        print("\nDistribución por clase:")
        
        for emotion in self.target_emotions:
            count = np.sum(labels == emotion)
            percentage = (count / len(labels)) * 100
            self.stats['per_class'][emotion] = count
            print(f"  {emotion:10s}: {count:6d} ({percentage:5.1f}%)")
        
        print("\nFuentes de datos:")
        for source, count in self.stats['sources'].items():
            percentage = (count / len(images)) * 100
            print(f"  {source:10s}: {count:6d} ({percentage:5.1f}%)")
        
        print("="*80)
        
        return images, labels
    
    def augment_data(self, images, labels, target_per_class=5000):
        """
        Aplica data augmentation para balancear clases.
        
        Args:
            images: Array de imágenes
            labels: Array de etiquetas
            target_per_class: Número objetivo de imágenes por clase
            
        Returns:
            augmented_images: Array balanceado de imágenes
            augmented_labels: Array balanceado de etiquetas
        """
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        print("\n" + "="*80)
        print("🔄 APLICANDO DATA AUGMENTATION")
        print("="*80)
        print(f"Objetivo por clase: {target_per_class} imágenes\n")
        
        # Configurar generador de augmentation
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            zoom_range=0.1,
            fill_mode='nearest'
        )
        
        augmented_images = []
        augmented_labels = []
        
        for emotion in self.target_emotions:
            # Filtrar imágenes de esta emoción
            mask = (labels == emotion)
            emotion_images = images[mask]
            current_count = len(emotion_images)
            
            print(f"{emotion:10s}: {current_count:5d} → ", end="")
            
            # Agregar imágenes originales
            augmented_images.extend(emotion_images)
            augmented_labels.extend([emotion] * current_count)
            
            # Si necesita más, generar con augmentation
            if current_count < target_per_class:
                needed = target_per_class - current_count
                generated = 0
                
                # Generar imágenes augmentadas
                for batch in datagen.flow(
                    emotion_images, 
                    batch_size=32, 
                    shuffle=True
                ):
                    for img in batch:
                        augmented_images.append(img.astype(np.uint8))
                        augmented_labels.append(emotion)
                        generated += 1
                        
                        if generated >= needed:
                            break
                    
                    if generated >= needed:
                        break
                
                print(f"{target_per_class:5d} (✓ {generated} generadas)")
            else:
                print(f"{current_count:5d} (sin augmentation)")
        
        augmented_images = np.array(augmented_images)
        augmented_labels = np.array(augmented_labels)
        
        # Shuffle final
        augmented_images, augmented_labels = shuffle(
            augmented_images, augmented_labels, random_state=42
        )
        
        print("\n" + "="*80)
        print(f"✅ Dataset balanceado: {len(augmented_images)} imágenes totales")
        print("="*80)
        
        return augmented_images, augmented_labels
    
    def split_dataset(self, images, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Divide dataset en train, validation y test sets.
        
        Args:
            images: Array de imágenes
            labels: Array de etiquetas
            train_ratio: Proporción para entrenamiento (0.7 = 70%)
            val_ratio: Proporción para validación (0.15 = 15%)
            test_ratio: Proporción para prueba (0.15 = 15%)
            
        Returns:
            (X_train, y_train), (X_val, y_val), (X_test, y_test)
        """
        print("\n" + "="*80)
        print("✂️  DIVIDIENDO DATASET")
        print("="*80)
        
        # Verificar que las proporciones sumen 1
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, \
            "Las proporciones deben sumar 1.0"
        
        # Convertir etiquetas a one-hot encoding
        from tensorflow.keras.utils import to_categorical
        
        # Crear mapeo de etiquetas a índices
        label_to_idx = {label: idx for idx, label in enumerate(self.target_emotions)}
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        
        # Convertir labels string a índices
        y_indices = np.array([label_to_idx[label] for label in labels])
        
        # One-hot encoding
        y_categorical = to_categorical(y_indices, num_classes=len(self.target_emotions))
        
        # Primera división: train vs (val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            images, y_categorical,
            test_size=(val_ratio + test_ratio),
            random_state=42,
            stratify=y_indices
        )
        
        # Segunda división: val vs test
        val_test_ratio = test_ratio / (val_ratio + test_ratio)
        y_temp_indices = np.argmax(y_temp, axis=1)
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=val_test_ratio,
            random_state=42,
            stratify=y_temp_indices
        )
        
        print(f"Training set:   {len(X_train):6d} imágenes ({train_ratio*100:.0f}%)")
        print(f"Validation set: {len(X_val):6d} imágenes ({val_ratio*100:.0f}%)")
        print(f"Test set:       {len(X_test):6d} imágenes ({test_ratio*100:.0f}%)")
        print("="*80)
        
        # Guardar mapeo de etiquetas
        self.label_to_idx = label_to_idx
        self.idx_to_label = idx_to_label
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def save_dataset(self, output_dir, train_data, val_data, test_data):
        """
        Guarda dataset procesado en formato NPZ.
        
        Args:
            output_dir: Directorio de salida
            train_data: (X_train, y_train)
            val_data: (X_val, y_val)
            test_data: (X_test, y_test)
        """
        print("\n" + "="*80)
        print(f"💾 GUARDANDO DATASET EN: {output_dir}")
        print("="*80)
        
        os.makedirs(output_dir, exist_ok=True)
        
        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data
        
        # Guardar arrays
        np.savez_compressed(
            os.path.join(output_dir, 'train_data.npz'),
            images=X_train,
            labels=y_train
        )
        print(f"✓ train_data.npz: {len(X_train)} imágenes")
        
        np.savez_compressed(
            os.path.join(output_dir, 'val_data.npz'),
            images=X_val,
            labels=y_val
        )
        print(f"✓ val_data.npz: {len(X_val)} imágenes")
        
        np.savez_compressed(
            os.path.join(output_dir, 'test_data.npz'),
            images=X_test,
            labels=y_test
        )
        print(f"✓ test_data.npz: {len(X_test)} imágenes")
        
        # Guardar metadatos
        metadata = {
            'target_emotions': self.target_emotions,
            'label_to_idx': self.label_to_idx,
            'idx_to_label': {int(k): v for k, v in self.idx_to_label.items()},
            'stats': self.stats,
            'input_shape': (160, 160, 3),
            'num_classes': len(self.target_emotions)
        }
        
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ metadata.json")
        print("="*80)
        print("✅ Dataset guardado exitosamente")
        print("="*80)


def main():
    """Función principal para preparar el dataset."""
    
    print("\n" + "="*80)
    print("📊 PREPARACIÓN DE DATOS - STRESS VISION")
    print("="*80)
    print("\nGloria S.A. - Sistema de Detección de Estrés Laboral")
    print("Fase 4: Entrenamiento y Optimización del Modelo\n")
    print("="*80)
    
    # Configuración de datasets
    print("\n⚙️  Configuración:")
    print("\nDatasets disponibles:")
    print("1. FER-2013: Descargar de https://www.kaggle.com/datasets/msambare/fer2013")
    print("2. Dataset custom: Crear estructura en data/custom_dataset/")
    print("\nEstructura requerida:")
    print("  data/")
    print("  ├── fer2013/")
    print("  │   └── fer2013.csv")
    print("  └── custom_dataset/")
    print("      ├── neutral/")
    print("      ├── stress/")
    print("      ├── sad/")
    print("      ├── happy/")
    print("      └── fatigue/")
    
    # Preguntar al usuario
    print("\n" + "="*80)
    use_fer = input("\n¿Desea usar FER-2013? (s/n): ").strip().lower()
    use_custom = input("¿Desea usar dataset custom? (s/n): ").strip().lower()
    
    if use_fer != 's' and use_custom != 's':
        print("\n❌ Debe seleccionar al menos un dataset")
        return
    
    # Construir configuración
    base_datasets = {}
    if use_fer == 's':
        fer_path = input("\nPath a fer2013.csv [data/fer2013/fer2013.csv]: ").strip()
        if not fer_path:
            fer_path = "data/fer2013/fer2013.csv"
        base_datasets['fer2013'] = fer_path
    
    custom_path = None
    if use_custom == 's':
        custom_path = input("\nPath a dataset custom [data/custom_dataset]: ").strip()
        if not custom_path:
            custom_path = "data/custom_dataset"
    
    # Inicializar builder
    builder = StressDatasetBuilder(
        base_datasets=base_datasets,
        custom_dataset_path=custom_path
    )
    
    # Cargar y combinar datasets
    images, labels = builder.load_and_merge_datasets()
    
    if len(images) == 0:
        print("\n❌ No se pudieron cargar imágenes")
        return
    
    # Preguntar si aplicar augmentation
    print("\n" + "="*80)
    apply_aug = input("\n¿Aplicar data augmentation para balancear clases? (s/n): ").strip().lower()
    
    if apply_aug == 's':
        target = input("Número objetivo por clase [5000]: ").strip()
        target = int(target) if target else 5000
        
        images, labels = builder.augment_data(images, labels, target_per_class=target)
    
    # Dividir dataset
    train_data, val_data, test_data = builder.split_dataset(images, labels)
    
    # Guardar
    output_dir = input("\nDirectorio de salida [data/processed]: ").strip()
    if not output_dir:
        output_dir = "data/processed"
    
    builder.save_dataset(output_dir, train_data, val_data, test_data)
    
    print("\n" + "="*80)
    print("✅ PREPARACIÓN COMPLETADA")
    print("="*80)
    print("\n💡 Próximos pasos:")
    print("   1. Verificar archivos en:", output_dir)
    print("   2. Ejecutar: python train_model.py")
    print("   3. Entrenar el modelo")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()


