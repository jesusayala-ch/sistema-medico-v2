import streamlit as st
import os
import time
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    layout="wide", 
    page_title="Clínica ORL - Diagnóstico IA",
    page_icon="🏥"
)

# ==========================================
# ⚠️ IMPORTANTE: CORRECCIÓN DE ETIQUETAS
# ==========================================
# Si el modelo predice mal (ej. dice Viral cuando es Normal),
# CAMBIA EL ORDEN DE ESTAS PALABRAS.
# El orden común suele ser alfabético: 0:Bacteriana, 1:Normal, 2:Viral
CLASES = ["Neumonía Bacteriana", "Normal / Sano", "Neumonía Viral"]

# ==========================================
# 🎨 PALETA DE COLORES
# ==========================================
COLOR_PRIMARIO = "#E6B800"  # Dorado
COLOR_SECUNDARIO = "#Fdfbf7" # Crema
COLOR_TEXTO = "#4A3B2A"      # Marrón
COLOR_ACENTO = "#D99000"

# ==========================================
# 🧠 CEREBRO IA (PyTorch)
# ==========================================
@st.cache_resource
def cargar_modelo():
    try:
        # 1. Reconstruir arquitectura (ResNet18)
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 3) 
        
        # 2. Definir rutas posibles
        ruta_carpeta = os.path.join("modelos", "modelo_neumonia_gpu.pth")
        ruta_raiz = "modelo_neumonia_gpu.pth"
        
        # Seleccionar la que exista
        ruta_final = ruta_carpeta if os.path.exists(ruta_carpeta) else ruta_raiz
        
        if os.path.exists(ruta_final):
            # Forzar carga en CPU para evitar errores
            map_location = torch.device('cpu')
            model.load_state_dict(torch.load(ruta_final, map_location=map_location))
            model.eval()
            return model
        else:
            return None
    except Exception as e:
        st.error(f"Error cargando el modelo: {e}")
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
# 💅 ESTILOS CSS (DISEÑO V1 + V2 MEZCLADO)
# ==========================================
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700;900&display=swap');

    header, footer, .stDeployButton {{visibility: hidden;}}

    /* FONDO GENERAL CON IMAGEN TIPO HOSPITAL (Como V1) */
    .stApp {{
        background: linear-gradient(rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0.8)), 
                    url("https://img.freepik.com/free-photo/blur-hospital_1203-7957.jpg");
        background-size: cover;
        background-attachment: fixed;
        font-family: 'Roboto', sans-serif;
    }}

    /* HEADER */
    .top-bar {{
        background-color: #F0A500;
        color: white;
        padding: 8px 30px;
        font-size: 14px;
        text-align: right;
        font-weight: bold;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }}

    .nav-bar {{
        background-color: rgba(255, 255, 255, 0.95);
        padding: 15px 40px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 30px;
        border-radius: 0 0 15px 15px;
    }}

    .logo-text {{ font-size: 40px; font-weight: 900; color: {COLOR_TEXTO}; }}
    .logo-icons {{ font-size: 25px; margin-left: 10px; color: #CCA000; }}

    /* TARJETAS CON EFECTO VIDRIO BLANCO (Como pediste) */
    .card {{
        background-color: rgba(255, 255, 255, 0.92); /* Blanco casi sólido */
        backdrop-filter: blur(10px); /* Efecto difuminado */
        border-top: 6px solid {COLOR_PRIMARIO};
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1); /* Sombra suave */
        margin-bottom: 20px;
    }}

    /* TÍTULOS DE BARRAS (Ahora grandes y oscuros) */
    .bar-title {{
        font-size: 22px !important; /* MÁS GRANDE */
        font-weight: 800 !important; /* MÁS GRUESO */
        color: {COLOR_TEXTO} !important;
        margin-top: 15px;
        margin-bottom: 5px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}

    /* BOTONES */
    .stButton>button {{
        background-color: {COLOR_PRIMARIO};
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        height: 55px;
        width: 100%;
        font-size: 20px;
        box-shadow: 0 4px 10px rgba(230, 184, 0, 0.4);
        transition: 0.3s;
    }}
    .stButton>button:hover {{ 
        background-color: {COLOR_ACENTO}; 
        transform: translateY(-2px);
    }}

    /* Color de las barras de carga */
    .stProgress > div > div > div > div {{ background-color: {COLOR_PRIMARIO}; }}
    
    /* Texto de ayuda */
    .small-text {{ font-size: 14px; color: #666; }}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 🏗️ ESTRUCTURA VISUAL
# ==========================================

st.markdown("""
    <div class="top-bar">📞 (064) 789440 &nbsp;&nbsp; ✉️ informes@consultorioorl.com</div>
    <div class="nav-bar">
        <div style="display: flex; align-items: center;">
            <span class="logo-text">ORL</span>
            <span class="logo-icons">👃 👂 👄</span>
        </div>
        <div style="color: #555; font-weight: 600;">
            INICIO &nbsp;&nbsp; SERVICIOS &nbsp;&nbsp; <span style="color:#E6B800; text-decoration: underline;">DIAGNÓSTICO IA</span>
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown(f"<h2 style='text-align: center; color: {COLOR_TEXTO}; margin-bottom: 30px; text-shadow: 1px 1px 2px white;'>SISTEMA DE APOYO AL DIAGNÓSTICO POR IMAGEN</h2>", unsafe_allow_html=True)

col_izq, col_der = st.columns([1, 1], gap="medium")

# --- VARIABLES ---
resultados_list = []
archivos_validos = False

# --- IZQUIERDA: CARGA ---
with col_izq:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"<h3 style='color:{COLOR_TEXTO}; margin-top:0;'>📂 Cargar Radiografías</h3>", unsafe_allow_html=True)
    st.markdown("<p class='small-text'>Formatos: JPG, PNG, JPEG. Máximo 2 archivos.</p>", unsafe_allow_html=True)
    
    archivos = st.file_uploader("Arrastra tus imágenes aquí", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    
    if archivos:
        if len(archivos) > 2:
            st.error(f"⚠️ Has subido {len(archivos)} imágenes. Por favor sube máximo 2.")
            archivos_validos = False
        else:
            archivos_validos = True
            cols_preview = st.columns(len(archivos))
            for idx, archivo in enumerate(archivos):
                image_pil = Image.open(archivo)
                cols_preview[idx].image(image_pil, caption=archivo.name, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- DERECHA: RESULTADOS ---
with col_der:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"<h3 style='color:{COLOR_TEXTO}; margin-top:0;'>📊 Resultados del Análisis</h3>", unsafe_allow_html=True)

    if archivos_validos:
        if st.button("🔍 ANALIZAR IMÁGENES"):
            if modelo_ia is None:
                st.warning("⚠️ MODO DEMO: No se detectó 'modelo_neumonia_gpu.pth'.")
                # DATOS DE PRUEBA SI NO HAY MODELO
                time.sleep(1.5)
                resultados_list = [{"nombre": archivos[0].name, "probs": [5.0, 92.5, 2.5]}] 
            else:
                # ANÁLISIS REAL
                with st.spinner("La IA está analizando los patrones..."):
                    for archivo in archivos:
                        try:
                            img = Image.open(archivo)
                            tensor = procesar_imagen(img)
                            with torch.no_grad():
                                output = modelo_ia(tensor)
                                probs_tensor = torch.nn.functional.softmax(output, dim=1)[0] * 100
                                probs = [p.item() for p in probs_tensor]
                                resultados_list.append({"nombre": archivo.name, "probs": probs})
                        except Exception as e:
                            st.error(f"Error procesando {archivo.name}")

            # RENDERIZAR RESULTADOS
            for res in resultados_list:
                st.markdown(f"<hr style='margin: 10px 0; border-color: #eee;'>", unsafe_allow_html=True)
                st.markdown(f"**Paciente / Archivo:** {res['nombre']}")
                
                probs = res['probs']
                
                # --- CONTROL DE CALIDAD (NO ES PULMÓN) ---
                if max(probs) < 60.0:
                    st.warning("⚠️ Diagnóstico incierto: La imagen podría no ser clara.")

                # --- BARRAS DE PROGRESO ---
                for i in range(3):
                    # TÍTULO GRANDE Y CLARO
                    st.markdown(f"<div class='bar-title'>{CLASES[i]}</div>", unsafe_allow_html=True)
                    st.progress(int(probs[i]))
                    st.markdown(f"<div style='text-align:right; font-weight:bold; color:{COLOR_ACENTO}; margin-top:-10px;'>{probs[i]:.2f}%</div>", unsafe_allow_html=True)

    else:
        # ESTADO INICIAL (VACÍO PERO VISIBLE)
        st.info("👈 Carga una imagen para comenzar el diagnóstico.")
        for clase in CLASES:
             st.markdown(f"<div class='bar-title' style='color:#ccc !important;'>{clase}</div>", unsafe_allow_html=True)
             st.progress(0)

    st.markdown('</div>', unsafe_allow_html=True)

# Pie de página
st.markdown("<div style='text-align:center; color:#888; margin-top:50px;'>© 2026 Sistema de Triaje Inteligente - V2.0 Scrum</div>", unsafe_allow_html=True)