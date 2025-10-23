"""
Carga de Enrollments a Base de Datos - Stress Vision
Carga los embeddings faciales generados a la base de datos SQLite

Autor: Gloria S.A.
Fecha: 2024
"""

import sqlite3
import json
import glob
import os
from datetime import datetime


class EnrollmentLoader:
    def __init__(self, db_path="gloria_stress_system.db"):
        """
        Inicializa el cargador de enrollments.
        
        Args:
            db_path: Ruta a la base de datos SQLite
        """
        self.db_path = db_path
        
        if not os.path.exists(db_path):
            print(f"❌ Error: Base de datos no encontrada: {db_path}")
            print("   Por favor, ejecute primero: python init_database.py")
            raise FileNotFoundError(f"Base de datos no encontrada: {db_path}")
    
    def load_enrollments_from_directory(self, directory="enrollments"):
        """
        Carga todos los enrollments desde un directorio.
        
        Args:
            directory: Directorio que contiene los archivos JSON de enrollment
            
        Returns:
            int: Número de enrollments cargados exitosamente
        """
        if not os.path.exists(directory):
            print(f"❌ Error: Directorio no encontrado: {directory}")
            print("   Por favor, ejecute primero: python enrollment.py")
            return 0
        
        # Buscar archivos de enrollment
        enrollment_files = glob.glob(f"{directory}/*_embedding.json")
        
        if not enrollment_files:
            print(f"⚠️  No se encontraron archivos de enrollment en: {directory}")
            print("   Asegúrese de haber ejecutado el proceso de enrollment primero.")
            return 0
        
        print("\n" + "="*80)
        print(f"📥 CARGANDO ENROLLMENTS A BASE DE DATOS")
        print("="*80)
        print(f"  📁 Directorio: {directory}")
        print(f"  📊 Archivos encontrados: {len(enrollment_files)}")
        print(f"  🗄️  Base de datos: {self.db_path}")
        print("="*80 + "\n")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        loaded_count = 0
        updated_count = 0
        failed_count = 0
        
        for i, filepath in enumerate(enrollment_files, 1):
            try:
                # Leer archivo JSON
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                employee_code = data['employee_code']
                employee_name = data['employee_name']
                
                print(f"[{i}/{len(enrollment_files)}] Procesando: {employee_name} ({employee_code})")
                
                # Verificar si el empleado ya existe
                cursor.execute(
                    "SELECT id FROM employees WHERE employee_code = ?",
                    (employee_code,)
                )
                existing = cursor.fetchone()
                
                # Convertir embedding a JSON string
                embedding_json = json.dumps(data['mean_embedding'])
                
                if existing:
                    # Actualizar empleado existente
                    cursor.execute("""
                        UPDATE employees SET
                            full_name = ?,
                            department = ?,
                            shift = ?,
                            face_embedding = ?,
                            face_encoding_quality = ?,
                            enrollment_date = ?,
                            consent_given = ?,
                            consent_date = ?,
                            thumbnail_base64 = ?,
                            is_active = 1,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE employee_code = ?
                    """, (
                        employee_name,
                        data.get('department', ''),
                        data.get('shift', ''),
                        embedding_json,
                        data['quality_score'],
                        data['timestamp'],
                        1 if data.get('consent_given', True) else 0,
                        data.get('consent_date', data['timestamp']),
                        data.get('thumbnail_base64', ''),
                        employee_code
                    ))
                    
                    print(f"  ✓ Actualizado (ID: {existing[0]})")
                    updated_count += 1
                else:
                    # Insertar nuevo empleado
                    cursor.execute("""
                        INSERT INTO employees (
                            employee_code,
                            full_name,
                            department,
                            shift,
                            face_embedding,
                            face_encoding_quality,
                            enrollment_date,
                            consent_given,
                            consent_date,
                            thumbnail_base64,
                            is_active
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                    """, (
                        employee_code,
                        employee_name,
                        data.get('department', ''),
                        data.get('shift', ''),
                        embedding_json,
                        data['quality_score'],
                        data['timestamp'],
                        1 if data.get('consent_given', True) else 0,
                        data.get('consent_date', data['timestamp']),
                        data.get('thumbnail_base64', '')
                    ))
                    
                    print(f"  ✓ Insertado (ID: {cursor.lastrowid})")
                    loaded_count += 1
                
                # Información adicional
                print(f"    • Calidad: {data['quality_score']:.2f}")
                print(f"    • Muestras: {data['num_samples']}")
                print(f"    • Departamento: {data.get('department', 'N/A')}")
                print()
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                print()
                failed_count += 1
                continue
        
        # Commit de cambios
        conn.commit()
        
        # Verificar registros cargados
        cursor.execute("SELECT COUNT(*) FROM employees WHERE is_active = 1")
        total_active = cursor.fetchone()[0]
        
        conn.close()
        
        # Resumen
        print("\n" + "="*80)
        print("📊 RESUMEN DE CARGA")
        print("="*80)
        print(f"  ✅ Nuevos empleados: {loaded_count}")
        print(f"  🔄 Empleados actualizados: {updated_count}")
        print(f"  ❌ Fallos: {failed_count}")
        print(f"  📈 Total procesados: {loaded_count + updated_count + failed_count}")
        print(f"  👥 Total empleados activos en BD: {total_active}")
        print("="*80 + "\n")
        
        if failed_count > 0:
            print(f"⚠️  Se encontraron {failed_count} errores. Revise los mensajes anteriores.")
        
        return loaded_count + updated_count
    
    def load_single_enrollment(self, filepath):
        """
        Carga un único archivo de enrollment.
        
        Args:
            filepath: Ruta al archivo JSON de enrollment
            
        Returns:
            bool: True si se cargó exitosamente
        """
        if not os.path.exists(filepath):
            print(f"❌ Error: Archivo no encontrado: {filepath}")
            return False
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            employee_code = data['employee_code']
            employee_name = data['employee_name']
            
            print(f"\n📥 Cargando: {employee_name} ({employee_code})")
            
            # Verificar si existe
            cursor.execute(
                "SELECT id FROM employees WHERE employee_code = ?",
                (employee_code,)
            )
            existing = cursor.fetchone()
            
            embedding_json = json.dumps(data['mean_embedding'])
            
            if existing:
                cursor.execute("""
                    UPDATE employees SET
                        full_name = ?,
                        department = ?,
                        shift = ?,
                        face_embedding = ?,
                        face_encoding_quality = ?,
                        enrollment_date = ?,
                        consent_given = ?,
                        consent_date = ?,
                        thumbnail_base64 = ?,
                        is_active = 1,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE employee_code = ?
                """, (
                    employee_name,
                    data.get('department', ''),
                    data.get('shift', ''),
                    embedding_json,
                    data['quality_score'],
                    data['timestamp'],
                    1 if data.get('consent_given', True) else 0,
                    data.get('consent_date', data['timestamp']),
                    data.get('thumbnail_base64', ''),
                    employee_code
                ))
                print(f"✅ Empleado actualizado (ID: {existing[0]})")
            else:
                cursor.execute("""
                    INSERT INTO employees (
                        employee_code,
                        full_name,
                        department,
                        shift,
                        face_embedding,
                        face_encoding_quality,
                        enrollment_date,
                        consent_given,
                        consent_date,
                        thumbnail_base64,
                        is_active
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """, (
                    employee_code,
                    employee_name,
                    data.get('department', ''),
                    data.get('shift', ''),
                    embedding_json,
                    data['quality_score'],
                    data['timestamp'],
                    1 if data.get('consent_given', True) else 0,
                    data.get('consent_date', data['timestamp']),
                    data.get('thumbnail_base64', '')
                ))
                print(f"✅ Empleado insertado (ID: {cursor.lastrowid})")
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            print(f"❌ Error al cargar enrollment: {e}")
            return False
    
    def list_employees(self):
        """Lista todos los empleados registrados en la base de datos."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                id,
                employee_code,
                full_name,
                department,
                shift,
                face_encoding_quality,
                enrollment_date,
                is_active
            FROM employees
            ORDER BY employee_code
        """)
        
        employees = cursor.fetchall()
        conn.close()
        
        if not employees:
            print("\n⚠️  No hay empleados registrados en la base de datos")
            return
        
        print("\n" + "="*120)
        print(f"👥 EMPLEADOS REGISTRADOS ({len(employees)} total)")
        print("="*120)
        print(f"{'ID':<5} {'Código':<10} {'Nombre':<30} {'Departamento':<20} {'Turno':<12} {'Calidad':<8} {'Estado':<8}")
        print("-"*120)
        
        for emp in employees:
            emp_id, code, name, dept, shift, quality, enroll_date, active = emp
            
            status = "Activo" if active else "Inactivo"
            dept_display = dept if dept else "N/A"
            shift_display = shift if shift else "N/A"
            quality_display = f"{quality:.2f}" if quality else "N/A"
            
            print(f"{emp_id:<5} {code:<10} {name:<30} {dept_display:<20} {shift_display:<12} {quality_display:<8} {status:<8}")
        
        print("="*120 + "\n")
        
        # Estadísticas
        cursor = sqlite3.connect(self.db_path).cursor()
        
        cursor.execute("SELECT COUNT(*) FROM employees WHERE is_active = 1")
        active_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(face_encoding_quality) FROM employees WHERE face_encoding_quality IS NOT NULL")
        avg_quality = cursor.fetchone()[0]
        
        cursor.execute("SELECT department, COUNT(*) FROM employees WHERE department IS NOT NULL GROUP BY department")
        dept_stats = cursor.fetchall()
        
        print("📊 Estadísticas:")
        print(f"  • Empleados activos: {active_count}/{len(employees)}")
        if avg_quality:
            print(f"  • Calidad promedio: {avg_quality:.2f}/1.0")
        
        if dept_stats:
            print(f"  • Por departamento:")
            for dept, count in dept_stats:
                print(f"    - {dept}: {count}")
        
        print()
    
    def verify_embeddings(self):
        """Verifica que todos los embeddings estén correctamente cargados."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                employee_code,
                full_name,
                face_embedding,
                face_encoding_quality
            FROM employees
            WHERE is_active = 1
        """)
        
        employees = cursor.fetchall()
        conn.close()
        
        print("\n" + "="*80)
        print("🔍 VERIFICACIÓN DE EMBEDDINGS")
        print("="*80 + "\n")
        
        valid_count = 0
        invalid_count = 0
        
        for code, name, embedding_json, quality in employees:
            try:
                if not embedding_json:
                    print(f"❌ {code} - {name}: Sin embedding")
                    invalid_count += 1
                    continue
                
                embedding = json.loads(embedding_json)
                
                if not isinstance(embedding, list):
                    print(f"❌ {code} - {name}: Formato inválido")
                    invalid_count += 1
                    continue
                
                if len(embedding) != 512:
                    print(f"❌ {code} - {name}: Dimensión incorrecta ({len(embedding)}, esperado: 512)")
                    invalid_count += 1
                    continue
                
                print(f"✅ {code} - {name}: OK (dim: {len(embedding)}, calidad: {quality:.2f})")
                valid_count += 1
                
            except Exception as e:
                print(f"❌ {code} - {name}: Error - {e}")
                invalid_count += 1
        
        print("\n" + "="*80)
        print("📊 RESULTADO DE VERIFICACIÓN")
        print("="*80)
        print(f"  ✅ Válidos: {valid_count}")
        print(f"  ❌ Inválidos: {invalid_count}")
        print(f"  📈 Total: {valid_count + invalid_count}")
        print("="*80 + "\n")
        
        return invalid_count == 0


def main():
    """Función principal para cargar enrollments."""
    
    print("\n" + "="*80)
    print("📥 CARGADOR DE ENROLLMENTS - STRESS VISION")
    print("="*80)
    print("\nGloria S.A. - Carga de Embeddings a Base de Datos")
    print("\n" + "="*80)
    
    try:
        loader = EnrollmentLoader("gloria_stress_system.db")
    except FileNotFoundError:
        print("\n❌ No se pudo inicializar el cargador.")
        print("   Ejecute primero: python init_database.py")
        return
    
    # Opciones
    print("\nOPCIONES:")
    print("1. Cargar todos los enrollments desde directorio")
    print("2. Cargar enrollment individual")
    print("3. Listar empleados registrados")
    print("4. Verificar embeddings")
    print("5. Salir")
    print("="*80)
    
    opcion = input("\nSeleccione una opción (1-5): ").strip()
    
    if opcion == "1":
        # Cargar todos
        directory = input("\nDirectorio de enrollments [enrollments]: ").strip()
        if not directory:
            directory = "enrollments"
        
        count = loader.load_enrollments_from_directory(directory)
        
        if count > 0:
            print(f"\n✅ Se cargaron {count} enrollments exitosamente")
            print("\n💡 Próximo paso:")
            print("   Ejecutar: streamlit run main.py")
            print("   Para iniciar el sistema de monitoreo")
    
    elif opcion == "2":
        # Cargar individual
        filepath = input("\nRuta al archivo de enrollment: ").strip()
        
        if loader.load_single_enrollment(filepath):
            print("\n✅ Enrollment cargado exitosamente")
        else:
            print("\n❌ Error al cargar enrollment")
    
    elif opcion == "3":
        # Listar empleados
        loader.list_employees()
    
    elif opcion == "4":
        # Verificar embeddings
        if loader.verify_embeddings():
            print("✅ Todos los embeddings son válidos")
        else:
            print("⚠️  Se encontraron embeddings inválidos")
    
    else:
        print("\n👋 Saliendo...")
    
    print("\n" + "="*80)
    print("✅ PROCESO FINALIZADO")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()




