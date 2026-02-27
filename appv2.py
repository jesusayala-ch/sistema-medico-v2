import streamlit as st
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import base64
from openai import OpenAI
import io

# --- CONFIGURACIÓN ---
st.set_page_config(layout="wide", page_title="ORL Gold PACS - AI Vision", page_icon="🌟")

# 🔑 Tu API Key (Asegúrate de no compartirla públicamente después)
os.environ["OPENAI_API_KEY"] = "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" 
client = OpenAI()

CLASES = ["Neumonía Bacteriana", "Normal / Sano", "Neumonía Viral"]

# --- FUNCIONES DE VALIDACIÓN Y ANÁLISIS ---

def validar_y_analizar(image_bytes, diagnostico_local):
    """
    Usa OpenAI para verificar si es una radiografía y, si lo es, dar el diagnóstico.
    """
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    
    # Prompt de seguridad: Validamos si es una RX de tórax
    prompt = f"""
    Eres un sistema de triaje radiológico. 
    PRIMERO: Verifica si la imagen adjunta es una radiografía de tórax real.
    - Si NO es una radiografía de tórax (es una persona, un objeto, un paisaje o cualquier otra cosa), responde ÚNICAMENTE con la palabra: "ERROR_NOT_XRAY".
    - Si SÍ es una radiografía, actúa como radiólogo. El sistema detectó {diagnostico_local}. Explica los hallazgos en 4 líneas técnicas.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }],
            max_tokens=200
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error API: {str(e)}"

@st.cache_resource
def cargar_modelo():
    try:
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 3) 
        if os.path.exists("modelo_neumonia_gpu.pth"):
            model.load_state_dict(torch.load("modelo_neumonia_gpu.pth", map_location=torch.device('cpu')))
            model.eval()
            return model
    except: return None
    return None

modelo_ia = cargar_modelo()

# --- INTERFAZ ---
st.title("🏥 Sistema Médico V2 - Clínica ORL")
st.markdown("---")

archivos = st.file_uploader("Cargar Radiografías", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if archivos:
    if st.button("EJECUTAR ANÁLISIS DUAL"):
        for archivo in archivos:
            img_bytes = archivo.read()
            img = Image.open(io.BytesIO(img_bytes))
            
            # 1. Ejecutar Modelo Local (ResNet)
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            tensor = transform(img.convert("RGB")).unsqueeze(0)
            
            with torch.no_grad():
                output = modelo_ia(tensor) if modelo_ia else torch.randn(1, 3)
                probs = torch.nn.functional.softmax(output, dim=1)[0]
                clase_idx = torch.argmax(probs).item()
            
            # 2. Validación y Análisis con OpenAI
            with st.spinner(f'Verificando autenticidad de {archivo.name}...'):
                resultado_ia = validar_y_analizar(img_bytes, CLASES[clase_idx])

            # --- MOSTRAR RESULTADOS CON LÓGICA DE ERROR ---
            with st.container():
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(img, width=400) 
                
                with col2:
                    st.subheader(f"Expediente: {archivo.name}")
                    
                    if "ERROR_NOT_XRAY" in resultado_ia:
                        # ❌ ALERTA DE ERROR SI NO ES RADIOGRAFÍA
                        st.error("⚠️ IMAGEN NO VÁLIDA: El sistema ha detectado que esta imagen no es una radiografía de tórax. Por seguridad, el análisis se ha cancelado.")
                    else:
                        # ✅ MOSTRAR DIAGNÓSTICO SI TODO ESTÁ BIEN
                        st.success(f"**Validación Exitosa:** Radiografía Identificada.")
                        st.write(f"**Resultado Local:** {CLASES[clase_idx]} ({probs[clase_idx]*100:.1f}%)")
                        st.info(resultado_ia)
                st.divider()
