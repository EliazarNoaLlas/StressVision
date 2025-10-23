"""
Generador de Reportes Automáticos - Stress Vision
Sistema de reportes periódicos cada 15 minutos (sin Celery, usa APScheduler)

Autor: Gloria S.A.
Fecha: 2024
"""

from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
import sqlite3
import json
from collections import defaultdict
import time


class ReportGenerator:
    def __init__(self, db_path='gloria_stress_system.db'):
        """
        Inicializa el generador de reportes.
        
        Args:
            db_path: Path a la base de datos SQLite
        """
        self.db_path = db_path
        self.scheduler = BackgroundScheduler()
        self.report_count = 0
        
        print("📊 Report Generator inicializado")
    
    def start(self, interval_minutes=15):
        """
        Inicia el scheduler de reportes.
        
        Args:
            interval_minutes: Intervalo en minutos (default: 15)
        """
        print(f"\n⏰ Programando reportes cada {interval_minutes} minutos...")
        
        # Programar tarea
        self.scheduler.add_job(
            func=self.generate_report,
            trigger='interval',
            minutes=interval_minutes,
            id='generate_15min_report',
            name=f'Reporte cada {interval_minutes} minutos',
            replace_existing=True
        )
        
        # Iniciar scheduler
        self.scheduler.start()
        
        print(f"✅ Scheduler iniciado")
        print(f"   • Próximo reporte: {self.scheduler.get_jobs()[0].next_run_time}")
    
    def stop(self):
        """Detiene el scheduler."""
        self.scheduler.shutdown()
        print("\n⏹️  Report Generator detenido")
    
    def generate_report(self):
        """Genera reporte de los últimos 15 minutos."""
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=15)
        
        print(f"\n{'='*80}")
        print(f"📊 GENERANDO REPORTE AUTOMÁTICO #{self.report_count + 1}")
        print(f"{'='*80}")
        print(f"   Período: {start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')}")
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Consultar detecciones en el período
            cursor.execute("""
                SELECT 
                    employee_id,
                    emotion,
                    emotion_confidence,
                    device_id,
                    timestamp
                FROM detection_events
                WHERE timestamp BETWEEN ? AND ?
            """, (start_time.isoformat(), end_time.isoformat()))
            
            detections = cursor.fetchall()
            
            if not detections:
                print("   ℹ️  No hay detecciones en este período")
                conn.close()
                return
            
            # Procesar datos
            total_detections = len(detections)
            per_employee = defaultdict(lambda: {
                'counts': defaultdict(int),
                'total': 0,
                'confidences': [],
                'devices': set()
            })
            
            emotion_totals = defaultdict(int)
            devices_used = set()
            
            for det in detections:
                emp_id = det['employee_id']
                emotion = det['emotion']
                conf = det['emotion_confidence']
                device = det['device_id']
                
                if emp_id:
                    per_employee[emp_id]['counts'][emotion] += 1
                    per_employee[emp_id]['total'] += 1
                    per_employee[emp_id]['confidences'].append(conf)
                    per_employee[emp_id]['devices'].add(device)
                
                emotion_totals[emotion] += 1
                devices_used.add(device)
            
            # Calcular métricas por empleado
            per_employee_summary = {}
            
            for emp_id, data in per_employee.items():
                # Obtener información del empleado
                cursor.execute("""
                    SELECT full_name, employee_code, department
                    FROM employees WHERE id = ?
                """, (emp_id,))
                
                emp_info = cursor.fetchone()
                
                if emp_info:
                    # Calcular porcentaje de estrés
                    stress_related = (data['counts']['stress'] + 
                                    data['counts']['sad'])
                    stress_pct = (stress_related / data['total']) * 100
                    
                    # Confianza promedio
                    avg_conf = sum(data['confidences']) / len(data['confidences'])
                    
                    per_employee_summary[str(emp_id)] = {
                        'employee_id': emp_id,
                        'name': emp_info['full_name'],
                        'code': emp_info['employee_code'],
                        'department': emp_info['department'],
                        'stress_pct': round(stress_pct, 2),
                        'counts': dict(data['counts']),
                        'avg_confidence': round(avg_conf, 2),
                        'total_detections': data['total']
                    }
            
            # Calcular estrés general
            total_stress = emotion_totals['stress'] + emotion_totals.get('sad', 0)
            overall_stress_pct = (total_stress / total_detections) * 100
            
            # Detectar alertas a generar
            alerts_to_create = []
            for emp_id, summary in per_employee_summary.items():
                if summary['stress_pct'] > 50:  # Más de 50% en estrés
                    alerts_to_create.append({
                        'employee_id': int(emp_id),
                        'stress_pct': summary['stress_pct'],
                        'severity': 'high' if summary['stress_pct'] > 70 else 'medium'
                    })
            
            # Guardar reporte en BD
            cursor.execute("""
                INSERT INTO reports_15min (
                    start_timestamp, end_timestamp,
                    total_detections, total_employees_detected,
                    overall_stress_percentage,
                    per_employee_summary,
                    alerts_triggered,
                    generated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                start_time.isoformat(),
                end_time.isoformat(),
                total_detections,
                len(per_employee),
                round(overall_stress_pct, 2),
                json.dumps(per_employee_summary),
                len(alerts_to_create),
                datetime.utcnow().isoformat()
            ))
            
            report_id = cursor.lastrowid
            conn.commit()
            
            # Mostrar resumen
            print(f"\n   ✅ Reporte #{report_id} generado:")
            print(f"      • Detecciones totales: {total_detections}")
            print(f"      • Empleados detectados: {len(per_employee)}")
            print(f"      • Estrés general: {overall_stress_pct:.2f}%")
            print(f"      • Dispositivos activos: {len(devices_used)}")
            
            if per_employee_summary:
                print(f"\n   👥 Top 5 empleados por detecciones:")
                sorted_emps = sorted(per_employee_summary.items(), 
                                    key=lambda x: x[1]['total_detections'], 
                                    reverse=True)[:5]
                
                for emp_id, summary in sorted_emps:
                    print(f"      • {summary['name']:30s} - {summary['total_detections']:3d} detecciones, "
                          f"{summary['stress_pct']:5.1f}% estrés")
            
            if alerts_to_create:
                print(f"\n   ⚠️  {len(alerts_to_create)} empleado(s) requieren atención:")
                for alert in alerts_to_create:
                    emp_id = alert['employee_id']
                    emp_summary = per_employee_summary[str(emp_id)]
                    print(f"      • {emp_summary['name']:30s} - {alert['stress_pct']:.1f}% estrés")
            
            print(f"{'='*80}\n")
            
            self.report_count += 1
            
            conn.close()
            
        except Exception as e:
            print(f"   ❌ Error generando reporte: {e}")
            import traceback
            traceback.print_exc()
    
    def generate_report_now(self):
        """Genera un reporte inmediatamente (manual)."""
        print("\n📊 Generando reporte manual...")
        self.generate_report()


def main():
    """Función principal para testing."""
    
    print("\n" + "="*80)
    print("📊 GENERADOR DE REPORTES AUTOMÁTICOS")
    print("="*80)
    print("\nGloria S.A. - Sistema de Detección de Estrés Laboral")
    print("\n" + "="*80)
    
    # Inicializar generador
    generator = ReportGenerator()
    
    # Opciones
    print("\nOpciones:")
    print("1. Generar reporte ahora (manual)")
    print("2. Iniciar reportes automáticos cada 15 minutos")
    print("3. Iniciar con intervalo personalizado")
    print("4. Salir")
    
    opcion = input("\nSeleccione opción (1-4): ").strip()
    
    if opcion == '1':
        # Reporte manual
        generator.generate_report_now()
    
    elif opcion == '2':
        # Automático cada 15 min
        generator.start(interval_minutes=15)
        
        print("\n💡 Reportes automáticos activados")
        print("   Presione Ctrl+C para detener\n")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n⏹️  Deteniendo...")
            generator.stop()
    
    elif opcion == '3':
        # Intervalo personalizado
        interval = input("\nIntervalo en minutos [15]: ").strip()
        interval = int(interval) if interval else 15
        
        generator.start(interval_minutes=interval)
        
        print(f"\n💡 Reportes cada {interval} minutos activados")
        print("   Presione Ctrl+C para detener\n")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n⏹️  Deteniendo...")
            generator.stop()
    
    else:
        print("👋 Saliendo...")
    
    print("\n" + "="*80)
    print("✅ PROCESO FINALIZADO")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()




