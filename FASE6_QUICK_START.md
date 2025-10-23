# ⚡ Quick Start - Fase 6: Backend y Sistema Completo

## 🚀 Inicio Super Rápido (1 Comando)

```bash
python start_complete_system.py
```

Seleccionar **Opción 3: Sistema Completo**

¡Listo! El sistema completo se iniciará automáticamente.

---

## 📊 ¿Qué se Inicia?

### 5 Componentes:

1. **Server Simulator** (Puerto 5000)
   - Recibe detecciones del Pi
   - Guarda en base de datos
   
2. **Pi Simulator** (Cámara + Detección)
   - Captura video
   - Detecta rostros y emociones
   - Envía al servidor

3. **Backend API** (Puerto 8000)
   - REST API con 15+ endpoints
   - WebSocket para tiempo real
   - http://localhost:8000/api/docs

4. **Dashboard** (Puerto 8501)
   - Interfaz web Streamlit
   - http://localhost:8501

5. **Report Generator**
   - Reportes automáticos cada 15 min
   - Corre en background

---

## 🌐 URLs Importantes

| Servicio | URL |
|----------|-----|
| Dashboard Streamlit | http://localhost:8501 |
| Backend API | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/api/docs |
| Server Stats | http://localhost:5000/stats |
| Health Check | http://localhost:8000/health |

---

## 📋 Opciones del Launcher

```
1. Sistema Edge Completo
   • Server Simulator + Pi Simulator
   
2. Backend API + Dashboard
   • Backend API + Dashboard Streamlit
   
3. Sistema Completo (RECOMENDADO)
   • Todos los componentes
   
4. Solo Backend API
5. Solo Dashboard
```

---

## 🧪 Verificación Rápida

### 1. Todos los componentes iniciaron:

```bash
# Windows
tasklist | findstr python

# Linux/Mac
ps aux | grep python
```

Deberías ver 5 procesos de Python activos.

### 2. API responde:

```bash
curl http://localhost:8000/health
```

Debería retornar:
```json
{"status": "ok"}
```

### 3. Dashboard carga:

Abrir en navegador: http://localhost:8501

### 4. Detecciones se guardan:

```bash
sqlite3 gloria_stress_system.db "SELECT COUNT(*) FROM detection_events;"
```

El número debería ir aumentando.

---

## 📊 Endpoints Esenciales

### Ver empleados:
```bash
curl http://localhost:8000/api/employees
```

### Ver overview del dashboard:
```bash
curl http://localhost:8000/api/dashboard/overview
```

### Ver alertas:
```bash
curl http://localhost:8000/api/alerts
```

### Ver reportes:
```bash
curl http://localhost:8000/api/reports
```

### Ver estadísticas:
```bash
curl http://localhost:8000/api/dashboard/stats
```

---

## 🔧 Solución Rápida de Problemas

### Puerto ocupado:

```bash
# Windows - Cerrar todo Python
taskkill /F /IM python.exe

# Linux/Mac
pkill python

# Reiniciar
python start_complete_system.py
```

### Base de datos bloqueada:

```bash
# Cerrar todos los procesos de Python
# Luego reiniciar
python start_complete_system.py
```

### Cámara no funciona:

```bash
# Verificar cámara disponible
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"

# Si False, verificar permisos y que no esté en uso
```

---

## ⏱️ Timeline

| Actividad | Tiempo |
|-----------|--------|
| Instalar deps nuevas | 2-3 min |
| Iniciar sistema | 30 seg |
| Verificar funcionamiento | 2-3 min |
| Prueba completa | 15-30 min |
| **TOTAL** | **20-35 min** |

---

## 💡 Tips Pro

### 1. Monitoreo Multi-Pantalla

- **Pantalla 1:** Dashboard (http://localhost:8501)
- **Pantalla 2:** API Docs (http://localhost:8000/api/docs)
- **Pantalla 3:** Preview del Pi Simulator

### 2. Probar Alertas

```bash
# Simular estrés prolongado (manual):
# 1. Modificar pi_simulator.py temporalmente para siempre retornar 'stress'
# 2. Dejar corriendo 5 minutos
# 3. Verificar alertas:
curl http://localhost:8000/api/alerts
```

### 3. Exportar Datos

```bash
# Exportar todas las detecciones
curl "http://localhost:8000/api/export/detections" > detections.json

# Exportar solo de un empleado
curl "http://localhost:8000/api/export/detections?employee_id=1" > emp1_detections.json

# Exportar solo estrés
curl "http://localhost:8000/api/export/detections?emotion=stress" > stress_detections.json
```

---

## 🎯 Próximos Pasos

Después de probar el sistema completo:

1. **Dejar corriendo 1-2 horas** para acumular datos

2. **Analizar reportes generados:**
   ```bash
   curl http://localhost:8000/api/reports
   ```

3. **Revisar alertas (si se generaron):**
   ```bash
   curl http://localhost:8000/api/alerts
   ```

4. **Exportar datos para análisis:**
   ```bash
   curl http://localhost:8000/api/export/detections > data.json
   ```

5. **Personalizar el dashboard** en `main.py` para usar el Backend API

---

## ✅ Checklist Rápido

### Antes de iniciar:
- [ ] `pip install -r requirements.txt` (nuevas deps)
- [ ] Base de datos existe
- [ ] Al menos 1 empleado enrollado (recomendado)

### Después de iniciar:
- [ ] 5 ventanas/procesos de Python activos
- [ ] http://localhost:8000/health retorna "ok"
- [ ] http://localhost:8501 carga dashboard
- [ ] Preview del simulador muestra tu rostro
- [ ] Detecciones aumentan en BD

---

**¡Sistema completo en 1 comando! 🚀**

Gloria S.A. - Stress Vision





