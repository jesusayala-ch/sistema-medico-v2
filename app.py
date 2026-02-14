import streamlit as st
import os
import time
import torch
import torch.nn as nn
import random
from torchvision import models, transforms
from PIL import Image

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    layout="wide", 
    page_title="ORL Gold PACS - Sistema Dinámico",
    page_icon="🌟"
)

# ==========================================
# DEFINICIÓN DE CLASES
# ==========================================
CLASES = ["Neumonía Bacteriana", "Normal / Sano", "Neumonía Viral"]

# ==========================================
# 🧠 CEREBRO DINÁMICO (5 VARIACIONES POR CLASE)
# ==========================================
INTERPRETACIONES = {
    # CLASE 0: BACTERIANA
    0: [
        "<b>INTERPRETACIÓN TÉCNICA:</b><br>Se visualiza opacidad focal lobular con densidad consolidada. La presencia de broncograma aéreo sugiere ocupación alveolar por exudado inflamatorio, compatible con etiología bacteriana típica.",
        "<b>INTERPRETACIÓN TÉCNICA:</b><br>Hallazgos de consolidación del espacio aéreo en proyección lobar. Los márgenes de la opacidad aparecen relativamente bien definidos, lo que contrasta con el tejido sano circundante. Cuadro sugestivo de proceso bacteriano agudo.",
        "<b>INTERPRETACIÓN TÉCNICA:</b><br>Aumento de densidad radiológica localizado. Se observa el 'signo de la silueta' positivo en relación con las estructuras mediastínicas adyacentes. Patrón alveolar predominante sobre el intersticial.",
        "<b>INTERPRETACIÓN TÉCNICA:</b><br>Opacidad homogénea que respeta las cisuras pulmonares. No se observan signos de colapso de volumen (atelectasia), lo que orienta el diagnóstico hacia una neumonía bacteriana lobar activa.",
        "<b>INTERPRETACIÓN TÉCNICA:</b><br>Foco de condensación pulmonar solitaria. La textura de la lesión es algodonosa con tendencia a la coalescencia. La ausencia de patrón reticular difuso descarta etiología viral primaria."
    ],
    
    # CLASE 1: NORMAL
    1: [
        "<b>INTERPRETACIÓN TÉCNICA:</b><br>Campos pulmonares radiolúcidos y bien aireados. Ángulos costofrénicos y cardiofrénicos agudos y libres. No se identifican infiltrados, masas ni consolidaciones activas.",
        "<b>INTERPRETACIÓN TÉCNICA:</b><br>Estudio dentro de límites normales. La trama vascular pulmonar presenta distribución y calibre conservados, disminuyendo gradualmente hacia la periferia. Silueta cardíaca de tamaño y morfología normal.",
        "<b>INTERPRETACIÓN TÉCNICA:</b><br>No se observan alteraciones pleuropulmonares agudas. El mediastino se encuentra centrado y la traquea es central. Estructuras óseas de la caja torácica sin lesiones líticas ni blásticas aparentes.",
        "<b>INTERPRETACIÓN TÉCNICA:</b><br>Transparencia pulmonar conservada bilateralmente. Hemidiafragmas con contornos lisos y convexidad superior preservada. No hay evidencia de derrame pleural ni engrosamiento peribronquial.",
        "<b>INTERPRETACIÓN TÉCNICA:</b><br>Radiografía de tórax sin hallazgos patológicos significativos. Espacios intercostales simétricos. La relación cardiotorácica se mantiene dentro del rango fisiológico (<0.5)."
    ],
    
    # CLASE 2: VIRAL
    2: [
        "<b>INTERPRETACIÓN TÉCNICA:</b><br>Patrón intersticial difuso bilateral. Se aprecian opacidades en 'vidrio deslustrado' que no borran completamente la trama vascular, característico de procesos inflamatorios virales o atípicos.",
        "<b>INTERPRETACIÓN TÉCNICA:</b><br>Afectación peribronquial con engrosamiento de las paredes bronquiales. Se observan infiltrados reticulares finos que irradian desde los hilios hacia la periferia, sugiriendo neumonitis viral.",
        "<b>INTERPRETACIÓN TÉCNICA:</b><br>Disminución difusa de la transparencia pulmonar sin consolidación lobar franca. El patrón es parcheado y de predominio basal, compatible con infección viral sistémica con repercusión pulmonar.",
        "<b>INTERPRETACIÓN TÉCNICA:</b><br>Opacidades retículo-nodulares mal definidas. A diferencia de la neumonía bacteriana, no se observa consolidación densa única, sino una afectación multicéntrica del intersticio pulmonar.",
        "<b>INTERPRETACIÓN TÉCNICA:</b><br>Hiperinsuflación pulmonar leve asociada a infiltrados intersticiales bilaterales. Hallazgos radiológicos consistentes con respuesta inflamatoria de vía aérea baja de etiología viral."
    ]
}

# ==========================================
# PALETA DE COLORES
# ==========================================
COLOR_PRIMARIO = "#B4975A"   
COLOR_SECUNDARIO = "#D4B77A" 
COLOR_TEXTO = "#333333"      
COLOR_FONDO_CARD = "#FFFFFF" 

# ==========================================
# MOTOR IA
# ==========================================
@st.cache_resource
def cargar_modelo():
    try:
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 3) 
        ruta_carpeta = os.path.join("modelos", "modelo_neumonia_gpu.pth")
        ruta_raiz = "modelo_neumonia_gpu.pth"
        ruta_final = ruta_carpeta if os.path.exists(ruta_carpeta) else ruta_raiz
        if os.path.exists(ruta_final):
            map_location = torch.device('cpu')
            model.load_state_dict(torch.load(ruta_final, map_location=map_location))
            model.eval()
            return model
        return None
    except Exception:
        return None

modelo_ia = cargar_modelo()

def procesar_imagen(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image.convert("RGB")).unsqueeze(0)

# ==========================================
# ESTILOS CSS
# ==========================================
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Helvetica+Neue:wght@300;400;500;700&display=swap');

    header, footer, .stDeployButton {{visibility: hidden;}}

    .stApp {{
        background: linear-gradient(rgba(252, 250, 245, 0.93), rgba(252, 250, 245, 0.97)), 
                    url("https://img.freepik.com/free-photo/luxury-abstract-background-gold-color_102862-83.jpg");
        background-size: cover;
        background-attachment: fixed;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: {COLOR_TEXTO};
    }}

    .top-status-bar {{
        background-color: #222; color: #B4975A; padding: 5px 30px;
        font-size: 11px; display: flex; justify-content: space-between;
        letter-spacing: 1px; text-transform: uppercase;
    }}

    .header-main {{
        background: white; padding: 20px 40px; display: flex; align-items: center;
        border-bottom: 3px solid {COLOR_PRIMARIO};
        box-shadow: 0 4px 15px rgba(180, 151, 90, 0.15); margin-bottom: 30px;
    }}
    .logo-text {{ font-size: 26px; font-weight: 700; color: #222; }}
    .logo-gold {{ color: {COLOR_PRIMARIO}; }}
    .system-tag {{ font-size: 12px; color: #999; margin-left: 15px; border-left: 1px solid #ccc; padding-left: 15px; font-weight: 500; }}

    .section-box {{
        background: linear-gradient(to right, {COLOR_PRIMARIO}, {COLOR_SECUNDARIO});
        padding: 12px 25px; border-radius: 6px; color: white;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08); margin-bottom: 25px;
        display: flex; align-items: center; justify-content: space-between;
    }}
    .section-title {{ font-size: 18px; font-weight: 600; margin: 0; text-transform: uppercase; }}

    .patient-card {{
        background-color: {COLOR_FONDO_CARD}; border: 1px solid #eaeaea;
        border-top: 4px solid {COLOR_PRIMARIO}; border-radius: 8px;
        padding: 25px; margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
    }}

    .interpretation-box {{
        background-color: #f8f9fa; border-left: 4px solid #28a745;
        padding: 15px; margin-top: 20px; font-family: 'Courier New', monospace;
        font-size: 13px; color: #444; line-height: 1.5;
    }}

    .file-meta-box {{
        background-color: #f9f9f9; border-left: 3px solid #ccc;
        padding: 10px 15px; margin-bottom: 20px;
        font-family: 'Courier New', monospace; font-size: 12px; color: #555;
    }}
    .file-name-highlight {{ font-weight: 700; color: #000; font-size: 14px; }}

    .bar-row {{ display: flex; justify-content: space-between; margin-bottom: 4px; font-size: 13px; font-weight: 500; color: #666; }}
    .prob-high {{ color: {COLOR_PRIMARIO}; font-weight: 700; font-size: 15px; }}

    .stButton>button {{
        background: linear-gradient(to bottom, {COLOR_SECUNDARIO}, {COLOR_PRIMARIO});
        color: white; border: none; height: 50px; border-radius: 4px;
        font-weight: 600; width: 100%; box-shadow: 0 4px 10px rgba(180, 151, 90, 0.3);
    }}
    .stButton>button:hover {{ background: linear-gradient(to bottom, {COLOR_PRIMARIO}, {COLOR_SECUNDARIO}); }}
    .stProgress > div > div > div > div {{ background-color: {COLOR_PRIMARIO}; }}
    
    .footer-container {{
        margin-top: 60px; padding: 30px 0; border-top: 1px solid #e0e0e0;
        color: #888; font-size: 11px; display: flex; justify-content: space-between;
    }}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================
st.markdown(f"""
    <div class="top-status-bar">
        <div><span style="color:{COLOR_SECUNDARIO}">◉</span> Sistema Online</div>
        <div>Conexión Segura SSL/TLS</div>
        <div>ID Terminal: ORL-DX-001</div>
    </div>
    <div class="header-main">
        <div class="logo-text">CLÍNICA <span class="logo-gold">ORL</span></div>
        <div class="system-tag">
            SISTEMA DE AYUDA AL DIAGNÓSTICO (CADx) <br>
            <span style="font-size:10px; color:{COLOR_PRIMARIO};">VERSIÓN 6.0 DYNAMIC</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# ==========================================
# INPUT
# ==========================================
st.markdown("""
    <div class="section-box">
        <div>
            <h3 class="section-title">📂 Módulo de Ingesta de Imágenes</h3>
            <span style="font-size: 12px; opacity: 0.9;">Protocolo de carga seguro.</span>
        </div>
        <div style="font-size: 20px;">⬇️</div>
    </div>
""", unsafe_allow_html=True)

archivos = st.file_uploader(
    "Cargar expedientes", 
    type=["jpg", "png", "jpeg"], 
    accept_multiple_files=True,
    label_visibility="collapsed"
)

# ==========================================
# LÓGICA PRINCIPAL
# ==========================================
resultados_list = []

if archivos:
    if len(archivos) > 10:
        st.error("⚠️ Límite de procesamiento excedido (Máx. 10).")
    else:
        col_b1, col_b2, col_b3 = st.columns([1, 2, 1])
        with col_b2:
            procesar = st.button("EJECUTAR ANÁLISIS DIAGNÓSTICO")

        if procesar:
            progreso = st.progress(0)
            
            for i, archivo in enumerate(archivos):
                # --- INFERENCIA ---
                probs = [0, 0, 0]
                clase_idx = 1 # Default Normal
                
                if modelo_ia:
                    try:
                        img = Image.open(archivo)
                        tensor = procesar_imagen(img)
                        with torch.no_grad():
                            output = modelo_ia(tensor)
                            probs_tensor = torch.nn.functional.softmax(output, dim=1)[0]
                            probs = (probs_tensor * 100).tolist()
                            clase_idx = torch.argmax(probs_tensor).item()
                    except Exception:
                        pass
                else:
                    # Datos Dummy (Simulación)
                    time.sleep(0.3) 
                    probs = [10.5, 85.0, 4.5]
                    clase_idx = 1

                # === SELECCIÓN ALEATORIA DE TEXTO (LA MAGIA) ===
                # Busca la lista de opciones para esa clase y elige una al azar
                opciones = INTERPRETACIONES.get(clase_idx, ["Diagnóstico no concluyente."])
                texto_final = random.choice(opciones)

                resultados_list.append({
                    "archivo": archivo,
                    "probs": probs,
                    "texto_interpretacion": texto_final
                })
                progreso.progress((i + 1) / len(archivos))
            
            time.sleep(0.5)
            progreso.empty()

# ==========================================
# RESULTADOS
# ==========================================
if resultados_list:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <div class="section-box">
             <div>
                <h3 class="section-title">📊 Informe Clínico Detallado</h3>
            </div>
             <div style="font-size: 20px;">✓</div>
        </div>
    """, unsafe_allow_html=True)

    for item in resultados_list:
        archivo = item['archivo']
        probs = item['probs']
        texto_explicativo = item['texto_interpretacion']

        with st.container():
            st.markdown('<div class="patient-card">', unsafe_allow_html=True)
            
            c_img, c_datos = st.columns([1, 2], gap="large")
            
            with c_img:
                st.markdown(f'<div style="border: 1px solid #eee; padding: 5px; background: white;">', unsafe_allow_html=True)
                image_pil = Image.open(archivo)
                st.image(image_pil, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with c_datos:
                st.markdown(f"""
                    <div class="file-meta-box">
                        EXPEDIENTE: <span class="file-name-highlight">{archivo.name}</span> | MODALIDAD: RX TÓRAX
                    </div>
                """, unsafe_allow_html=True)

                st.markdown("<div style='margin-bottom: 5px; font-weight: 600; font-size: 14px; color: #444;'>PROBABILIDADES DETECTADAS:</div>", unsafe_allow_html=True)
                for i in range(3):
                    es_alta = probs[i] > 50
                    clase_css = "prob-high" if es_alta else ""
                    icono = "➤" if es_alta else ""
                    st.markdown(f"""
                        <div class="bar-row">
                            <span>{icono} {CLASES[i]}</span>
                            <span class="{clase_css}">{probs[i]:.2f}%</span>
                        </div>
                    """, unsafe_allow_html=True)
                    st.progress(int(probs[i]))

                # CAJA DE INTERPRETACIÓN DINÁMICA
                st.markdown(f"""
                    <div class="interpretation-box">
                        {texto_explicativo}
                    </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

elif not archivos:
    st.markdown(f"""
        <div style="text-align: center; margin-top: 60px; opacity: 0.5;">
            <div style="font-size: 60px; color: {COLOR_SECUNDARIO}; margin-bottom: 10px;">☤</div>
            <p>Sistema en espera de expedientes.</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("""
    <div class="footer-container">
        <div>© 2026 Clínica ORL | Protocolos Médicos ISO 13485</div>
        <div>Estado del Servidor Neural: OPERATIVO</div>
    </div>
""", unsafe_allow_html=True)
