import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from collections import deque
import time
import json

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Detección de Estrés - Gloria S.A.",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1e3a8a;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .alert-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    .alert-danger {
        background-color: #fee2e2;
        border-color: #dc2626;
        color: #991b1b;
    }
    .alert-warning {
        background-color: #fef3c7;
        border-color: #f59e0b;
        color: #92400e;
    }
    .alert-success {
        background-color: #d1fae5;
        border-color: #10b981;
        color: #065f46;
    }
    .stats-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Inicialización del estado de la sesión
if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = deque(maxlen=100)
if 'stress_events' not in st.session_state:
    st.session_state.stress_events = []
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = {
        'total_detections': 0,
        'negative_count': 0,
        'neutral_count': 0,
        'positive_count': 0,
        'stress_alerts': 0
    }
if 'employee_records' not in st.session_state:
    st.session_state.employee_records = {}

# Configuración de emociones negativas
NEGATIVE_EMOTIONS = ['angry', 'fear', 'sad', 'disgust']
NEUTRAL_EMOTIONS = ['neutral']
POSITIVE_EMOTIONS = ['happy', 'surprise']

EMOTION_COLORS = {
    'angry': '#dc2626',
    'fear': '#ea580c',
    'sad': '#6366f1',
    'disgust': '#7c3aed',
    'neutral': '#64748b',
    'happy': '#10b981',
    'surprise': '#f59e0b'
}

EMOTION_LABELS_ES = {
    'angry': 'Enojado',
    'fear': 'Miedo',
    'sad': 'Triste',
    'disgust': 'Disgusto',
    'neutral': 'Neutral',
    'happy': 'Feliz',
    'surprise': 'Sorpresa'
}


def calculate_stress_index(emotion_history, window=30):
    """Calcula el índice de estrés basado en el historial de emociones"""
    if len(emotion_history) < 5:
        return 0

    recent = list(emotion_history)[-window:]
    negative_count = sum(1 for e in recent if e in NEGATIVE_EMOTIONS)
    stress_index = (negative_count / len(recent)) * 100
    return stress_index


def analyze_frame(frame, detector='retinaface'):
    """Analiza un frame y retorna las emociones detectadas"""
    try:
        results = DeepFace.analyze(
            frame,
            actions=['emotion', 'age', 'gender'],
            detector_backend=detector,
            enforce_detection=False
        )
        return results
    except Exception as e:
        return None


def draw_analysis_overlay(frame, results):
    """Dibuja información del análisis sobre el frame"""
    if results is None or len(results) == 0:
        return frame

    result = results[0] if isinstance(results, list) else results

    # Extraer región facial
    if 'region' in result:
        region = result['region']
        x, y, w, h = region['x'], region['y'], region['w'], region['h']

        # Dibujar rectángulo facial
        emotion = result['dominant_emotion']
        color = (0, 0, 255) if emotion in NEGATIVE_EMOTIONS else (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

        # Información de overlay
        emotion_es = EMOTION_LABELS_ES.get(emotion, emotion)
        age = result.get('age', 'N/A')
        gender = result.get('dominant_gender', 'N/A')

        # Fondo para texto
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y - 80), (x + w, y), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # Textos
        cv2.putText(frame, f"Emocion: {emotion_es}", (x + 5, y - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Edad: {age}", (x + 5, y - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Genero: {gender}", (x + 5, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


# Header
st.markdown('<h1 class="main-header">🧠 Sistema de Detección de Estrés Laboral</h1>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align: center; color: #64748b; font-size: 1.2rem;">Gloria S.A. - Monitoreo de Bienestar Organizacional</p>',
    unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/organization.png", width=80)
    st.markdown("### ⚙️ Configuración")

    mode = st.radio(
        "Modo de Operación:",
        ["📹 Análisis en Tiempo Real", "📁 Análisis de Imagen", "📊 Dashboard de Reportes"],
        index=0
    )

    st.markdown("---")
    st.markdown("### 🎯 Parámetros")

    detector = st.selectbox(
        "Detector Facial:",
        ['retinaface', 'mtcnn', 'opencv', 'ssd'],
        index=0
    )

    stress_threshold = st.slider(
        "Umbral de Estrés (%):",
        min_value=20,
        max_value=80,
        value=40,
        step=5
    )

    alert_frequency = st.slider(
        "Ventana de Análisis (frames):",
        min_value=10,
        max_value=50,
        value=30,
        step=5
    )

    st.markdown("---")
    st.markdown("### 📋 Emociones Negativas")
    st.markdown("""
    - 😠 Enojado
    - 😰 Miedo
    - 😢 Triste
    - 🤢 Disgusto
    """)

    st.markdown("---")
    if st.button("🗑️ Limpiar Historial"):
        st.session_state.emotion_history.clear()
        st.session_state.stress_events.clear()
        st.session_state.analysis_data = {
            'total_detections': 0,
            'negative_count': 0,
            'neutral_count': 0,
            'positive_count': 0,
            'stress_alerts': 0
        }
        st.success("Historial limpiado")

# Métricas principales
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Detecciones",
        st.session_state.analysis_data['total_detections'],
        delta=None
    )

with col2:
    negative_pct = (st.session_state.analysis_data['negative_count'] /
                    max(st.session_state.analysis_data['total_detections'], 1)) * 100
    st.metric(
        "Emociones Negativas",
        f"{negative_pct:.1f}%",
        delta=f"{st.session_state.analysis_data['negative_count']}"
    )

with col3:
    current_stress = calculate_stress_index(st.session_state.emotion_history, alert_frequency)
    st.metric(
        "Índice de Estrés",
        f"{current_stress:.1f}%",
        delta="⚠️ Alto" if current_stress > stress_threshold else "✅ Normal"
    )

with col4:
    st.metric(
        "Alertas Generadas",
        st.session_state.analysis_data['stress_alerts'],
        delta=None
    )

st.markdown("---")

# Contenido principal según modo seleccionado
if mode == "📹 Análisis en Tiempo Real":
    st.markdown("### 📹 Análisis en Tiempo Real")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        run_camera = st.checkbox("▶️ Iniciar Cámara", value=False)
        video_placeholder = st.empty()

        if run_camera:
            cap = cv2.VideoCapture(0)
            frame_counter = 0

            while run_camera:
                ret, frame = cap.read()
                if not ret:
                    st.error("No se pudo acceder a la cámara")
                    break

                frame_counter += 1

                # Analizar cada N frames para optimizar rendimiento
                if frame_counter % 5 == 0:
                    results = analyze_frame(frame, detector)

                    if results:
                        result = results[0] if isinstance(results, list) else results
                        emotion = result['dominant_emotion']

                        # Actualizar historial
                        st.session_state.emotion_history.append(emotion)
                        st.session_state.analysis_data['total_detections'] += 1

                        if emotion in NEGATIVE_EMOTIONS:
                            st.session_state.analysis_data['negative_count'] += 1
                        elif emotion in NEUTRAL_EMOTIONS:
                            st.session_state.analysis_data['neutral_count'] += 1
                        else:
                            st.session_state.analysis_data['positive_count'] += 1

                        # Calcular índice de estrés
                        stress_idx = calculate_stress_index(st.session_state.emotion_history, alert_frequency)

                        if stress_idx > stress_threshold:
                            st.session_state.analysis_data['stress_alerts'] += 1
                            st.session_state.stress_events.append({
                                'timestamp': datetime.now(),
                                'stress_index': stress_idx,
                                'emotion': emotion
                            })

                        # Dibujar overlay
                        frame = draw_analysis_overlay(frame, results)

                # Mostrar frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                time.sleep(0.03)

            cap.release()

    with col_right:
        st.markdown("### 📊 Análisis Actual")

        if len(st.session_state.emotion_history) > 0:
            last_emotion = st.session_state.emotion_history[-1]
            emotion_es = EMOTION_LABELS_ES.get(last_emotion, last_emotion)

            if last_emotion in NEGATIVE_EMOTIONS:
                st.markdown(
                    f'<div class="alert-box alert-danger">⚠️ <strong>Emoción Negativa Detectada:</strong> {emotion_es}</div>',
                    unsafe_allow_html=True)
            elif last_emotion in POSITIVE_EMOTIONS:
                st.markdown(
                    f'<div class="alert-box alert-success">✅ <strong>Emoción Positiva:</strong> {emotion_es}</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="alert-box alert-warning">ℹ️ <strong>Estado Neutral:</strong> {emotion_es}</div>',
                    unsafe_allow_html=True)

            # Gráfico de emociones recientes
            if len(st.session_state.emotion_history) > 5:
                recent_emotions = list(st.session_state.emotion_history)[-30:]
                emotion_counts = pd.Series(recent_emotions).value_counts()

                fig = px.pie(
                    values=emotion_counts.values,
                    names=[EMOTION_LABELS_ES.get(e, e) for e in emotion_counts.index],
                    title="Distribución de Emociones (últimos 30 frames)",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

elif mode == "📁 Análisis de Imagen":
    st.markdown("### 📁 Análisis de Imagen Estática")

    uploaded_file = st.file_uploader("Subir imagen del colaborador:", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Leer imagen
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Imagen Original")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col2:
            with st.spinner("Analizando imagen..."):
                results = analyze_frame(image, detector)

                if results:
                    result = results[0] if isinstance(results, list) else results

                    # Dibujar análisis
                    annotated_image = draw_analysis_overlay(image.copy(), results)
                    st.markdown("#### Análisis Detectado")
                    st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), use_container_width=True)

                    # Mostrar detalles
                    st.markdown("### 📋 Resultados del Análisis")

                    emotion = result['dominant_emotion']
                    emotion_es = EMOTION_LABELS_ES.get(emotion, emotion)

                    col_a, col_b, col_c = st.columns(3)

                    with col_a:
                        st.metric("Emoción Dominante", emotion_es)
                    with col_b:
                        st.metric("Edad Estimada", f"{result.get('age', 'N/A')} años")
                    with col_c:
                        st.metric("Género", result.get('dominant_gender', 'N/A'))

                    # Distribución de emociones
                    st.markdown("#### 📊 Distribución de Probabilidades")
                    emotions_dict = result.get('emotion', {})
                    emotions_df = pd.DataFrame({
                        'Emoción': [EMOTION_LABELS_ES.get(k, k) for k in emotions_dict.keys()],
                        'Probabilidad': list(emotions_dict.values())
                    })
                    emotions_df = emotions_df.sort_values('Probabilidad', ascending=False)

                    fig = px.bar(
                        emotions_df,
                        x='Probabilidad',
                        y='Emoción',
                        orientation='h',
                        color='Probabilidad',
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

                    # Alerta de estrés
                    if emotion in NEGATIVE_EMOTIONS:
                        st.markdown(
                            f'<div class="alert-box alert-danger">⚠️ <strong>ALERTA:</strong> Se detectó una emoción negativa ({emotion_es}). Se recomienda seguimiento por parte de RRHH.</div>',
                            unsafe_allow_html=True)
                else:
                    st.error("No se pudo detectar ningún rostro en la imagen.")

else:  # Dashboard de Reportes
    st.markdown("### 📊 Dashboard de Reportes y Estadísticas")

    if len(st.session_state.emotion_history) < 10:
        st.info("⚠️ No hay suficientes datos para generar reportes. Ejecuta el análisis en tiempo real primero.")
    else:
        # Convertir historial a DataFrame
        emotions_list = list(st.session_state.emotion_history)
        df = pd.DataFrame({
            'timestamp': [datetime.now() - timedelta(seconds=i) for i in range(len(emotions_list) - 1, -1, -1)],
            'emotion': emotions_list
        })
        df['emotion_es'] = df['emotion'].map(EMOTION_LABELS_ES)
        df['category'] = df['emotion'].apply(
            lambda x: 'Negativa' if x in NEGATIVE_EMOTIONS
            else ('Positiva' if x in POSITIVE_EMOTIONS else 'Neutral')
        )

        # Gráfico de línea temporal
        st.markdown("#### 📈 Evolución Temporal de Emociones")
        fig_timeline = px.scatter(
            df,
            x='timestamp',
            y='emotion_es',
            color='category',
            color_discrete_map={'Negativa': '#dc2626', 'Neutral': '#64748b', 'Positiva': '#10b981'},
            title="Línea Temporal de Emociones Detectadas"
        )
        fig_timeline.update_layout(height=400)
        st.plotly_chart(fig_timeline, use_container_width=True)

        # Estadísticas generales
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🎯 Distribución Global de Emociones")
            emotion_counts = df['emotion_es'].value_counts()
            fig_pie = px.pie(
                values=emotion_counts.values,
                names=emotion_counts.index,
                hole=0.4
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            st.markdown("#### 📊 Categorías de Emociones")
            category_counts = df['category'].value_counts()
            fig_bar = px.bar(
                x=category_counts.index,
                y=category_counts.values,
                color=category_counts.index,
                color_discrete_map={'Negativa': '#dc2626', 'Neutral': '#64748b', 'Positiva': '#10b981'},
                labels={'x': 'Categoría', 'y': 'Cantidad'}
            )
            fig_bar.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        # Tabla de eventos de estrés
        if st.session_state.stress_events:
            st.markdown("#### ⚠️ Registro de Eventos de Estrés")
            events_df = pd.DataFrame(st.session_state.stress_events)
            events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])
            events_df['emotion'] = events_df['emotion'].map(EMOTION_LABELS_ES)
            events_df = events_df.sort_values('timestamp', ascending=False)

            st.dataframe(
                events_df.style.background_gradient(subset=['stress_index'], cmap='Reds'),
                use_container_width=True,
                height=300
            )

            # Botón de exportación
            csv = events_df.to_csv(index=False)
            st.download_button(
                label="📥 Descargar Reporte de Estrés (CSV)",
                data=csv,
                file_name=f"reporte_estres_gloria_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

        # Recomendaciones
        st.markdown("#### 💡 Recomendaciones")

        negative_pct = (st.session_state.analysis_data['negative_count'] /
                        max(st.session_state.analysis_data['total_detections'], 1)) * 100

        if negative_pct > 50:
            st.error(
                "🚨 **Alerta Crítica:** Más del 50% de emociones negativas detectadas. Se recomienda intervención inmediata de RRHH.")
        elif negative_pct > 30:
            st.warning(
                "⚠️ **Precaución:** Niveles elevados de emociones negativas. Considerar implementar pausas activas y seguimiento.")
        else:
            st.success("✅ **Estado Normal:** Los niveles de estrés se encuentran dentro de parámetros aceptables.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 2rem;'>
    <p><strong>Sistema de Detección de Estrés Laboral v1.0</strong></p>
    <p>Gloria S.A. | Área de Recursos Humanos | Salud Ocupacional</p>
    <p style='font-size: 0.9rem;'>⚡ Powered by DeepFace | 🔒 Datos procesados localmente | 📊 Dashboard en Tiempo Real</p>
</div>
""", unsafe_allow_html=True)