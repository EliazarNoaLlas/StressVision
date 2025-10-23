"""
Conversor a TensorFlow Lite - Stress Vision
Optimización y conversión de modelos para Raspberry Pi 5

Optimizaciones:
- Quantization INT8 (4x reducción de tamaño)
- Optimización de operaciones
- Benchmark de latencia

Autor: Gloria S.A.
Fecha: 2024
"""

import tensorflow as tf
import numpy as np
import time
import json
import os
from pathlib import Path


class TFLiteConverter:
    def __init__(self, model_path):
        """
        Inicializa el conversor.
        
        Args:
            model_path: Path al modelo Keras (.keras o .h5)
        """
        self.model_path = model_path
        
        print(f"📥 Cargando modelo desde: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
        
        self.model = tf.keras.models.load_model(model_path)
        
        print(f"✅ Modelo cargado: {self.model.name}")
        print(f"   • Input shape: {self.model.input_shape}")
        print(f"   • Output shape: {self.model.output_shape}")
        print(f"   • Parámetros: {self.model.count_params():,}")
    
    def convert_float32(self, output_path='model_float32.tflite'):
        """
        Convierte modelo a TFLite Float32 (sin quantization).
        
        Ventajas:
        - Máxima precisión
        - Conversión simple
        
        Desventajas:
        - Tamaño grande
        - Latencia alta
        
        Args:
            output_path: Path de salida
            
        Returns:
            output_path: Path al modelo convertido
        """
        print("\n" + "="*80)
        print("🔄 CONVERSIÓN A TFLITE FLOAT32")
        print("="*80)
        
        # Configurar converter
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Sin optimizaciones
        print("   • Tipo: Float32")
        print("   • Optimizaciones: Ninguna")
        
        # Convertir
        print("\n⏳ Convirtiendo...")
        tflite_model = converter.convert()
        
        # Guardar
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        # Estadísticas
        original_size = self.model.count_params() * 4 / (1024**2)  # MB
        tflite_size = len(tflite_model) / (1024**2)  # MB
        
        print(f"\n📊 Resultados:")
        print(f"   • Modelo original: {original_size:.2f} MB")
        print(f"   • Modelo TFLite: {tflite_size:.2f} MB")
        print(f"   • Guardado en: {output_path}")
        print("="*80)
        
        return output_path
    
    def convert_float16(self, output_path='model_float16.tflite'):
        """
        Convierte modelo a TFLite Float16.
        
        Ventajas:
        - Reducción 2x de tamaño
        - Pérdida mínima de precisión
        - Compatible con la mayoría de dispositivos
        
        Args:
            output_path: Path de salida
            
        Returns:
            output_path: Path al modelo convertido
        """
        print("\n" + "="*80)
        print("🔄 CONVERSIÓN A TFLITE FLOAT16")
        print("="*80)
        
        # Configurar converter
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Optimización Float16
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
        print("   • Tipo: Float16")
        print("   • Optimizaciones: DEFAULT")
        print("   • Reducción esperada: ~50%")
        
        # Convertir
        print("\n⏳ Convirtiendo...")
        tflite_model = converter.convert()
        
        # Guardar
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        # Estadísticas
        original_size = self.model.count_params() * 4 / (1024**2)
        tflite_size = len(tflite_model) / (1024**2)
        reduction = (1 - tflite_size/original_size) * 100
        
        print(f"\n📊 Resultados:")
        print(f"   • Modelo original: {original_size:.2f} MB")
        print(f"   • Modelo TFLite: {tflite_size:.2f} MB")
        print(f"   • Reducción: {reduction:.1f}%")
        print(f"   • Guardado en: {output_path}")
        print("="*80)
        
        return output_path
    
    def convert_int8(self, representative_dataset_generator=None, 
                     output_path='model_int8.tflite'):
        """
        Convierte modelo a TFLite INT8 con post-training quantization.
        
        Ventajas:
        - Reducción 4x de tamaño
        - Inferencia más rápida en CPU
        - Menor consumo de energía
        
        Requiere:
        - Dataset representativo para calibración
        
        Args:
            representative_dataset_generator: Generator de datos para calibración
            output_path: Path de salida
            
        Returns:
            output_path: Path al modelo convertido
        """
        print("\n" + "="*80)
        print("🔄 CONVERSIÓN A TFLITE INT8 (QUANTIZED)")
        print("="*80)
        
        if representative_dataset_generator is None:
            print("⚠️  No se proporcionó dataset representativo")
            print("   Usando datos sintéticos para calibración...")
            representative_dataset_generator = self._generate_representative_dataset()
        
        # Configurar converter
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Optimizaciones
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Quantization INT8
        converter.representative_dataset = representative_dataset_generator
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        
        print("   • Tipo: INT8")
        print("   • Optimizaciones: DEFAULT + INT8")
        print("   • Input/Output: UINT8")
        print("   • Reducción esperada: ~75%")
        
        # Convertir
        print("\n⏳ Convirtiendo (puede tomar varios minutos)...")
        try:
            tflite_model = converter.convert()
        except Exception as e:
            print(f"❌ Error en conversión: {e}")
            print("   Intente con Float16 o revise el modelo")
            return None
        
        # Guardar
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        # Estadísticas
        original_size = self.model.count_params() * 4 / (1024**2)
        tflite_size = len(tflite_model) / (1024**2)
        reduction = (1 - tflite_size/original_size) * 100
        
        print(f"\n📊 Resultados:")
        print(f"   • Modelo original: {original_size:.2f} MB")
        print(f"   • Modelo TFLite INT8: {tflite_size:.2f} MB")
        print(f"   • Reducción: {reduction:.1f}%")
        print(f"   • Guardado en: {output_path}")
        print("="*80)
        
        return output_path
    
    def _generate_representative_dataset(self, num_samples=100):
        """
        Genera dataset representativo sintético para calibración.
        
        Args:
            num_samples: Número de muestras
            
        Returns:
            generator: Generator de datos
        """
        input_shape = self.model.input_shape[1:]  # Sin batch dimension
        
        def representative_dataset_gen():
            for _ in range(num_samples):
                # Generar imagen aleatoria normalizada
                data = np.random.rand(1, *input_shape).astype(np.float32)
                yield [data]
        
        return representative_dataset_gen
    
    def benchmark_latency(self, tflite_path, num_runs=100, num_warmup=10):
        """
        Mide latencia de inferencia del modelo TFLite.
        
        Args:
            tflite_path: Path al modelo TFLite
            num_runs: Número de ejecuciones para benchmark
            num_warmup: Número de ejecuciones de calentamiento
            
        Returns:
            stats: Dict con estadísticas de latencia
        """
        print("\n" + "="*80)
        print(f"⚡ BENCHMARK DE LATENCIA")
        print("="*80)
        print(f"   • Modelo: {tflite_path}")
        print(f"   • Warmup runs: {num_warmup}")
        print(f"   • Benchmark runs: {num_runs}")
        
        # Cargar modelo TFLite
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Obtener detalles de input/output
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"   • Input shape: {input_details[0]['shape']}")
        print(f"   • Input dtype: {input_details[0]['dtype']}")
        print(f"   • Output shape: {output_details[0]['shape']}")
        
        # Preparar input dummy
        input_shape = input_details[0]['shape']
        input_dtype = input_details[0]['dtype']
        
        if input_dtype == np.uint8:
            input_data = np.random.randint(0, 256, input_shape, dtype=np.uint8)
        else:
            input_data = np.random.rand(*input_shape).astype(input_dtype)
        
        # Warmup
        print(f"\n⏳ Warmup...")
        for _ in range(num_warmup):
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
        
        # Benchmark
        print(f"⏳ Ejecutando benchmark...")
        times = []
        
        for i in range(num_runs):
            start = time.perf_counter()
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            end = time.perf_counter()
            
            times.append((end - start) * 1000)  # Convert to ms
            
            if (i + 1) % 25 == 0:
                print(f"   Completado: {i + 1}/{num_runs}")
        
        # Calcular estadísticas
        times = np.array(times)
        stats = {
            'mean_ms': float(np.mean(times)),
            'std_ms': float(np.std(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'p50_ms': float(np.percentile(times, 50)),
            'p95_ms': float(np.percentile(times, 95)),
            'p99_ms': float(np.percentile(times, 99)),
            'fps_theoretical': float(1000 / np.mean(times))
        }
        
        # Mostrar resultados
        print(f"\n📊 Resultados del Benchmark:")
        print(f"   • Latencia promedio: {stats['mean_ms']:.2f} ms")
        print(f"   • Desviación estándar: {stats['std_ms']:.2f} ms")
        print(f"   • Mínimo: {stats['min_ms']:.2f} ms")
        print(f"   • Máximo: {stats['max_ms']:.2f} ms")
        print(f"   • P50 (mediana): {stats['p50_ms']:.2f} ms")
        print(f"   • P95: {stats['p95_ms']:.2f} ms")
        print(f"   • P99: {stats['p99_ms']:.2f} ms")
        print(f"   • FPS teórico: {stats['fps_theoretical']:.1f}")
        
        # Verificar KPI (≤ 200ms)
        print(f"\n🎯 Verificación de KPIs:")
        if stats['mean_ms'] <= 200:
            print(f"   ✅ CUMPLE KPI de latencia (≤ 200ms)")
        else:
            print(f"   ❌ NO CUMPLE KPI de latencia")
            print(f"   📝 Sugerencias:")
            print(f"      • Reducir input size (ej: 120x120)")
            print(f"      • Usar arquitectura más ligera")
            print(f"      • Considerar Coral USB TPU")
        
        print("="*80)
        
        return stats
    
    def compare_models(self, model_paths, num_runs=100):
        """
        Compara latencia de múltiples modelos TFLite.
        
        Args:
            model_paths: Lista de paths a modelos TFLite
            num_runs: Número de ejecuciones para cada modelo
            
        Returns:
            comparison: Dict con comparación de modelos
        """
        print("\n" + "="*80)
        print("📊 COMPARACIÓN DE MODELOS")
        print("="*80)
        
        results = {}
        
        for model_path in model_paths:
            model_name = Path(model_path).stem
            print(f"\n🔍 Evaluando: {model_name}")
            
            stats = self.benchmark_latency(model_path, num_runs=num_runs, num_warmup=10)
            
            # Agregar tamaño del modelo
            size_mb = os.path.getsize(model_path) / (1024**2)
            stats['size_mb'] = size_mb
            
            results[model_name] = stats
        
        # Tabla comparativa
        print("\n" + "="*80)
        print("TABLA COMPARATIVA")
        print("="*80)
        print(f"{'Modelo':<25} {'Tamaño (MB)':<12} {'Latencia (ms)':<15} {'FPS':<10} {'KPI':<6}")
        print("-"*80)
        
        for name, stats in results.items():
            kpi_status = "✅" if stats['mean_ms'] <= 200 else "❌"
            print(f"{name:<25} {stats['size_mb']:<12.2f} {stats['mean_ms']:<15.2f} "
                  f"{stats['fps_theoretical']:<10.1f} {kpi_status}")
        
        print("="*80)
        
        return results


def main():
    """Función principal."""
    
    print("\n" + "="*80)
    print("🔄 CONVERSOR A TENSORFLOW LITE - STRESS VISION")
    print("="*80)
    print("\nGloria S.A. - Sistema de Detección de Estrés Laboral")
    print("Fase 4: Optimización para Raspberry Pi 5\n")
    print("="*80)
    
    # Solicitar modelo
    model_path = input("\nPath al modelo Keras (.keras) [models/best_model.keras]: ").strip()
    if not model_path:
        model_path = "models/best_model.keras"
    
    if not os.path.exists(model_path):
        print(f"\n❌ Modelo no encontrado: {model_path}")
        return
    
    # Inicializar conversor
    converter = TFLiteConverter(model_path)
    
    # Seleccionar tipo de conversión
    print("\n" + "="*80)
    print("TIPOS DE CONVERSIÓN DISPONIBLES:")
    print("="*80)
    print("1. Float32 (sin optimización)")
    print("   • Máxima precisión")
    print("   • Tamaño grande")
    print("\n2. Float16 (optimización media)")
    print("   • ~50% reducción de tamaño")
    print("   • Pérdida mínima de precisión")
    print("\n3. INT8 Quantized (máxima optimización)")
    print("   • ~75% reducción de tamaño")
    print("   • Inferencia más rápida")
    print("   • ⚠️ Requiere calibración")
    print("\n4. Todas las anteriores")
    
    opcion = input("\nSeleccione opción (1-4) [3]: ").strip()
    if not opcion:
        opcion = '3'
    
    output_dir = "models/tflite"
    os.makedirs(output_dir, exist_ok=True)
    
    model_paths = []
    
    if opcion in ['1', '4']:
        path = converter.convert_float32(f"{output_dir}/model_float32.tflite")
        if path:
            model_paths.append(path)
    
    if opcion in ['2', '4']:
        path = converter.convert_float16(f"{output_dir}/model_float16.tflite")
        if path:
            model_paths.append(path)
    
    if opcion in ['3', '4']:
        path = converter.convert_int8(output_path=f"{output_dir}/model_int8.tflite")
        if path:
            model_paths.append(path)
    
    # Benchmark
    if model_paths:
        print("\n" + "="*80)
        do_benchmark = input("\n¿Realizar benchmark de latencia? (s/n) [s]: ").strip().lower()
        
        if do_benchmark != 'n':
            num_runs = input("Número de ejecuciones [100]: ").strip()
            num_runs = int(num_runs) if num_runs else 100
            
            results = converter.compare_models(model_paths, num_runs=num_runs)
            
            # Guardar resultados
            results_path = f"{output_dir}/benchmark_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n💾 Resultados guardados en: {results_path}")
    
    print("\n" + "="*80)
    print("✅ CONVERSIÓN COMPLETADA")
    print("="*80)
    print(f"\n📁 Modelos guardados en: {output_dir}/")
    print("\n💡 Próximos pasos:")
    print("   1. Copiar modelo optimizado a Raspberry Pi")
    print("   2. Integrar en sistema de monitoreo")
    print("   3. Probar en producción")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()


