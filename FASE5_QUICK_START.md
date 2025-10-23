# ⚡ Quick Start - Fase 5: Simulador de Raspberry Pi

## 🚀 Inicio Rápido (3 Pasos)

### Paso 1: Verificar Requisitos (1 min)

```bash
# Ejecutar suite de pruebas
python test_pi_system.py
```

Deberías ver:
```
✅ PASS  Dependencias
✅ PASS  Base de Datos
✅ PASS  Cámara
✅ PASS  Servidor
✅ PASS  Configuración
✅ PASS  Pipeline Completo

Total: 6/6 pruebas exitosas (100%)
```

---

### Paso 2: Iniciar Sistema (1 comando)

```bash
python start_pi_system.py
```

El sistema abrirá automáticamente:
- **Ventana 1:** Servidor (estadísticas cada 30s)
- **Ventana 2:** Simulador de Pi (preview de cámara)

---

### Paso 3: ¡Usar el Sistema!

1. **Siéntate frente a la cámara**
2. **Observa la ventana de preview**
   - Verás rectángulos alrededor de tu rostro
   - Color según emoción detectada
   - Tu nombre si estás enrollado

3. **Monitorea las estadísticas**
   - Abre en navegador: `http://localhost:5000/stats`
   - O revisa la ventana del servidor

4. **Para detener:**
   - Presiona `Q` en la ventana del simulador
   - O `Ctrl+C` en las terminales

---

## 📊 ¿Qué Esperar?

### En la Ventana del Simulador:

```
┌─────────────────────────────────────────────┐
│ FPS: 10.3                                   │
│ Rostros: 1                                  │
│ Detecciones: 15                             │
│ Device: pi-simulator-01                     │
├─────────────────────────────────────────────┤
│                                             │
│     ┌──────────────────┐                   │
│     │ Juan Pérez       │                   │
│     │ neutral (0.92)   │                   │
│     └──────────────────┘                   │
│           [Tu rostro]                       │
│                                             │
│                                             │
│ ESPACIO: Capturar | Q: Salir               │
└─────────────────────────────────────────────┘
```

### En el Servidor:

```
📊 ESTADÍSTICAS DEL SERVIDOR
========================================
   Total detecciones: 25
   Empleados únicos: 1
   Sesiones activas: 1

   Por dispositivo:
      • pi-simulator-01: 25

   Por emoción:
      • neutral: 20
      • happy: 3
      • stress: 2
```

### En la Base de Datos:

```bash
sqlite3 gloria_stress_system.db

SELECT COUNT(*) FROM detection_events;
# Debería ir aumentando mientras el sistema corre
```

---

## 🎯 Casos de Uso Rápidos

### Caso 1: Probar Reconocimiento Facial

```bash
# 1. Asegúrate de estar enrollado
python load_enrollments.py  # Opción 3: Listar

# 2. Inicia sistema
python start_pi_system.py

# 3. Mira a la cámara
# → Deberías ver tu nombre en el preview
```

### Caso 2: Probar Sin Enrollments

```bash
# Funciona igual, pero mostrará "Track#ID" en lugar de nombre
python start_pi_system.py
```

### Caso 3: Simular Detección de Estrés

```bash
# 1. Inicia sistema
python start_pi_system.py

# 2. Haz expresiones faciales
# → El sistema (en modo mock) detectará emociones aleatorias
# → Con modelo real, detectará tus emociones reales

# 3. Revisa estadísticas
curl http://localhost:5000/stats
```

---

## 🔧 Solución de Problemas Comunes

### Problema: "Address already in use"

```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Linux/Mac
kill -9 $(lsof -ti:5000)
```

### Problema: No aparece mi nombre

**Posibles causas:**
1. No estás enrollado → `python enrollment.py`
2. Threshold muy alto → Editar `pi_config.py`: `'recognition_threshold': 0.5`
3. Mala iluminación → Mejorar luz
4. Mala calidad del enrollment → Re-enrollar

### Problema: Cámara no funciona

```bash
# Verificar índice correcto
python -c "import cv2; print([i for i in range(4) if cv2.VideoCapture(i).isOpened()])"

# Cambiar en pi_config.py
'camera_index': 1,  # Probar con otro índice
```

---

## 📋 Comandos Útiles

### Iniciar sistema completo:
```bash
python start_pi_system.py
```

### Iniciar componentes por separado:
```bash
# Terminal 1
python server_simulator.py

# Terminal 2
python pi_simulator.py
```

### Ver estadísticas:
```bash
curl http://localhost:5000/stats
```

### Ver configuración actual:
```bash
python pi_config.py
```

### Ver últimas detecciones en BD:
```bash
sqlite3 gloria_stress_system.db "SELECT timestamp, emotion, employee_id FROM detection_events ORDER BY timestamp DESC LIMIT 10;"
```

### Limpiar detecciones de prueba:
```bash
sqlite3 gloria_stress_system.db "DELETE FROM detection_events WHERE device_id LIKE '%simulator%';"
```

---

## ⏱️ Timeline

| Actividad | Tiempo |
|-----------|--------|
| Verificar requisitos | 1 min |
| Iniciar sistema | 30 seg |
| Prueba básica | 2-5 min |
| Testing completo | 10-15 min |
| **TOTAL** | **15-20 min** |

---

## 💡 Tips

### 1. Usa dos monitores

- Monitor 1: Ventanas del sistema (servidor + simulador)
- Monitor 2: Navegador con estadísticas

### 2. Logging

- Las detecciones se guardan en `logs/detections/`
- Útil para análisis posterior
- Un archivo JSONL por día

### 3. Testing incremental

- Primero prueba servidor solo: `python server_simulator.py`
- Verifica que responda: `curl http://localhost:5000/health`
- Luego inicia simulador: `python pi_simulator.py`

### 4. Performance

- Si va lento, aumenta `frame_skip` a 5 en `pi_config.py`
- Si quieres más FPS, reduce a 2
- Si no necesitas preview, `show_preview=False`

---

## 🎯 Resultado Esperado

Después de ejecutar, deberías tener:

✅ Servidor corriendo en `http://localhost:5000`  
✅ Simulador capturando video en tiempo real  
✅ Detecciones guardándose en base de datos  
✅ Reconocimiento facial funcionando (si tienes enrollments)  
✅ Estadísticas actualizándose en tiempo real  
✅ Preview mostrando detecciones con colores  

---

## 📞 Siguiente Paso

Una vez que el sistema funcione correctamente:

### Fase 6: Dashboard Avanzado
- Integrar detecciones en `main.py`
- Vista por empleado
- Reportes automáticos

---

**¡Listo para simular un Raspberry Pi! 🚀**

Gloria S.A. - Stress Vision





