import streamlit as st
from PIL import Image
from pathlib import Path
import sys
import os

# El archivo inference.py ahora se encuentra en la misma carpeta que app.py
from inference import MultimodalPredictor

# ==========================================
# 0. CONFIGURACIÓN DE PÁGINA
# ==========================================
st.set_page_config(
    page_title="Clasificador de Riesgo de Autismo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 1. ESTILADO CSS
# ==========================================
STITCH_CSS = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    
    :root {
        --primary-purple: #5A4FCF;
        --bg-color: #0E1117;
        --card-bg: #1A1C23;
        --text-main: #E2E8F0;
        --text-muted: #94A3B8;
        --border-color: #2D3748;
    }
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: var(--text-main);
    }

    .upload-card {
        background-color: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    
    h1, h2, h3 {
        color: white !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em;
    }
    
    h1 {
        font-size: 2.2rem !important;
        margin-bottom: 0.2rem !important;
    }
    
    .subtitle {
        color: var(--text-muted);
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    .disclaimer-banner {
        background-color: rgba(217, 119, 6, 0.1);
        border-left: 4px solid #D97706;
        padding: 15px 20px;
        border-radius: 4px;
        color: #FCD34D;
        font-size: 0.9rem;
        margin-bottom: 30px;
    }
    /* Estilos para el botón principal (Generar Reporte) usando el selector nativo de Streamlit */
    button[data-testid="baseButton-primary"] {
        background-color: #2563EB !important; /* Blue */
        color: white !important;
        border: none !important;
        padding: 0.75rem 1rem !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    button[data-testid="baseButton-primary"]:hover {
        background-color: #1D4ED8 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4) !important;
    }

    .metric-value {
        font-size: 3rem;
        font-weight: 700;
        color: white;
        line-height: 1;
    }
    .metric-label {
        font-size: 1rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 5px;
    }
</style>
"""
st.markdown(STITCH_CSS, unsafe_allow_html=True)

# ==========================================
# 2. FUNCIONES BASE Y CACHÉ
# ==========================================
@st.cache_resource(show_spinner=False)
def load_predictor():
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    models_dir = current_dir.parent / "models"
    return MultimodalPredictor(str(models_dir))

# Manejo de estado para cargar ejemplos
if "demo_sag" not in st.session_state:
    st.session_state.demo_sag = None
if "demo_cor" not in st.session_state:
    st.session_state.demo_cor = None
if "demo_axi" not in st.session_state:
    st.session_state.demo_axi = None

def load_example(prefix):
    base_path = Path(__file__).parent / "assets"
    st.session_state.demo_sag = Image.open(base_path / f"{prefix}_sagittal.png")
    st.session_state.demo_cor = Image.open(base_path / f"{prefix}_coronal.png")
    st.session_state.demo_axi = Image.open(base_path / f"{prefix}_axial.png")

def clear_examples():
    st.session_state.demo_sag = None
    st.session_state.demo_cor = None
    st.session_state.demo_axi = None

def render_standard_image(img):
    """Fuerza la imagen a un tamaño cuadrado igual 256x256 para UI limpios."""
    if img is not None:
        return img.resize((256, 256))
    return None

# ==========================================
# 3. NAVEGACIÓN LATERAL
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2855/2855601.png", width=60)
    st.title("Navegación")
    seccion = st.radio(
        "Menú Principal:",
        ["🩺 Prueba del Modelo", "📘 Arquitectura e Infomación"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.caption("Ernesto Saniel Castro Lozano - 2026")


# ==========================================
# 4. SECCIÓN: PRUEBA DEL MODELO
# ==========================================
if seccion == "🩺 Prueba del Modelo":
    st.markdown("<h1>Clasificador por Fusión de Embeddings Multivista</h1>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Fusión temprana de embeddings extraídos de Resonancia Magnética Estructural (Sagital, Coronal y Axial).</div>", unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div class='disclaimer-banner'>
        <strong>Proyecto de Portafolio Académico.</strong> Este modelo de deep learning alcanzó un AUROC de 0.6738 sobre el conjunto 
        de validación puro de pruebas. Utiliza redes convolucionales y un MLP de fusión temprana modular, 
        pero <strong>NO</strong> está validado ni debe usarse para un diagnóstico médico real.
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Cargando arquitectura tensorial..."):
        predictor = load_predictor()

    # Controles de Ejemplos Integrados
    st.markdown("### Seleccionar Caso de Ejemplo")
    st.write("Si no tienes imágenes a mano, puedes cargar perfiles clínicos pre-extraídos:")
    ej1, ej2, ej3, ej4 = st.columns(4)
    with ej1:
        if st.button("🔴 Autismo (Alta prob)", use_container_width=True): load_example("autism1")
    with ej2:
        if st.button("🟠 Autismo (Moderada)", use_container_width=True): load_example("autism2")
    with ej3:
        if st.button("🟢 Neurotípico (Control)", use_container_width=True): load_example("control1")
    with ej4:
        if st.button("🧹 Limpiar Entradas", use_container_width=True): clear_examples()
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Contenedores de Carga
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<div class='upload-card'><h3>Plano Sagital</h3></div>", unsafe_allow_html=True)
        sag_file = st.file_uploader("Sube recorte Sagital", type=["png", "jpg", "jpeg"], key="sag")
        img_sag = render_standard_image(Image.open(sag_file) if sag_file else st.session_state.demo_sag)
        if img_sag:
            st.image(img_sag, use_container_width=True, caption="Entrada Sagital (256x256)")

    with col2:
        st.markdown("<div class='upload-card'><h3>Plano Coronal</h3></div>", unsafe_allow_html=True)
        cor_file = st.file_uploader("Sube recorte Coronal", type=["png", "jpg", "jpeg"], key="cor")
        img_cor = render_standard_image(Image.open(cor_file) if cor_file else st.session_state.demo_cor)
        if img_cor:
            st.image(img_cor, use_container_width=True, caption="Entrada Coronal (256x256)")

    with col3:
        st.markdown("<div class='upload-card'><h3>Plano Axial</h3></div>", unsafe_allow_html=True)
        axi_file = st.file_uploader("Sube recorte Axial", type=["png", "jpg", "jpeg"], key="axi")
        img_axi = render_standard_image(Image.open(axi_file) if axi_file else st.session_state.demo_axi)
        if img_axi:
            st.image(img_axi, use_container_width=True, caption="Entrada Axial (256x256)")

    st.markdown("---")

    # LOGICA DE INFERENCIA
    all_uploaded = (img_sag is not None) and (img_cor is not None) and (img_axi is not None)
    col_empty, col_button, col_empty2 = st.columns([1, 2, 1])

    with col_button:
        if st.button("Generar Reporte de Predicción", type="primary", disabled=not all_uploaded, use_container_width=True):
            
            with st.spinner("Fusionando representaciones tensoriales..."):
                prob = predictor.predict(img_sag, img_cor, img_axi)
                prob_percent = prob * 100.0
                
                st.markdown("<br>", unsafe_allow_html=True)
                res_col1, res_col2 = st.columns([1, 1])
                
                with res_col1:
                    if prob < 0.40:
                        color = "#10B981" # Green
                        risk_lvl = "Bajo Riesgo (Patrón Neurotípico)"
                    elif prob <= 0.60:
                        color = "#FCD34D" # Yellow
                        risk_lvl = "Indeterminado / Zona Limítrofe"
                    else:
                        color = "#EF4444" # Red
                        risk_lvl = "Alto Riesgo (Firma Estructural Asociada a TEA)"
                    
                    st.markdown(f"""
                    <div style="background-color: var(--card-bg); padding: 30px; border-radius: 12px; border: 1px solid var(--border-color); text-align: center;">
                        <div style="color: {color};" class="metric-value">{prob_percent:.1f}%</div>
                        <div class="metric-label">Probabilidad Predicha</div>
                        <div style="margin-top: 15px; color: {color}; font-weight: 600;">{risk_lvl}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with res_col2:
                    st.markdown("""
                    <div style="background-color: var(--card-bg); padding: 30px; border-radius: 12px; border: 1px solid var(--border-color); height: 100%;">
                        <h4 style="margin-top: 0; color: white;">Análisis del Perceptrón de Fusión</h4>
                        <p style="color: var(--text-muted); font-size: 0.95rem; line-height: 1.5;">
                            Los 3 backbones independientes (EfficientNetB0) extrajeron mapas de características proyectados a <b>256 dimensiones</b> cada uno.
                        </p>
                        <p style="color: var(--text-muted); font-size: 0.95rem; line-height: 1.5;">
                            Al concatenar los tres mapeos espaciales (Vector 768-d), el modelo identificó correlaciones morfológicas cruzadas que están asociadas estadísticamente con perfiles del espectro autista dentro del volumen cerebral.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)


    if not all_uploaded:
        st.info("💡 Sube imágenes o haz clic en los botones de pacientes para evaluar la red.", icon="🩺")

# ==========================================
# 5. SECCIÓN: INFORMACIÓN
# ==========================================
elif seccion == "📘 Arquitectura e Infomación":
    st.markdown("<h1>Arquitectura de Fusión Temprana</h1>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Detalles técnicos sobre la implementación del ensamblaje multimodal.</div>", unsafe_allow_html=True)
    
    st.write("El desafío central de este proyecto radica en consolidar información tridimensional estructural de cerebros a partir de cortes 2D dispersos geográficamente a gran escala.")

    # Uso de contenedores nativos para evitar HTML roto
    with st.container():
        st.markdown("### 1. Extractores de Características Vías Separadas (CNN)")
        st.info("""Se utilizaron 3 modelos `EfficientNetB0` entrenados exclusivamente para cada corte (Axial, Coronal y Sagital), 
        despojados de su cabeza de clasificación final. En su lugar, se insertó una capa densa que comprime las texturas visuales del MRI en **vectores latentes de 256 dimensiones**.""")

    with st.container():
        st.markdown("### 2. Concatenación Tensorial Espacial")
        st.info("""Los **3 tensores (1, 256)** que salen de las sub-redes convolucionales no se promedian, sino que se concatenan (fusión a nivel de vector temprano). 
        Esto produce un hipervector de **768 dimensiones**, obligando a la red superior a aprender cómo interaccionan geométricamente los planos, en vez de tratarlos como entes asilados.""")

    with st.container():
        st.markdown("### 3. Fusión vía Perceptrón Multicapa (MLP)")
        st.info("""Una diminuta red neuronal fully-connected se encarga de analizar el tensor de 768 dimensiones.
        Para evitar memorización en un dataset tan complejo, se emplearon robustas capas de `Dropout(0.5)`, condensando las señales latentes hasta calcular una salida lineal unificada pasada por Sigmoide.""")
        
    st.markdown("---")
    
    # Github Link section
    st.markdown("### 🔗 Repositorio y Referencias Clave")
    st.markdown("""
    * **Código fuente y Notebooks de Inferencia MLOps:** [Repositorio Oficial en GitHub](https://github.com/ErnestoSCL/app_prediccion_autismo_fusion_embeddings)
    * **Datos:** Los conjuntos de formación provienen del cruce estandarizado en repositorios multicéntricos globales de resonancias magnéticas, incluyendo metadatos preprocesados por _FreeSurfer_.
    """)
