import os
import platform
import traceback

import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="RAG • Chatea con tu PDF", page_icon="🔷", layout="centered")

# ------------------------------------------------------------------
# Estilos: azul profundo + cian eléctrico, rejilla tenue y brillo neón
# ------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;800&family=Exo+2:wght@300;400;600&display=swap');

:root {
    --bg: #040B1E;
    --panel: #0A1633;
    --panel-2: #0E1E45;
    --blue: #1E6BFF;
    --cyan: #22D3EE;
    --ice: #E6F1FF;
    --muted: #8FA8D6;
    --line: rgba(34, 211, 238, 0.28);
}

/* Fondo: degradado azul + rejilla */
.stApp {
    background:
        radial-gradient(1000px 500px at 85% -10%, rgba(30,107,255,0.30), transparent 60%),
        radial-gradient(800px 500px at -10% 110%, rgba(34,211,238,0.14), transparent 60%),
        linear-gradient(rgba(34,211,238,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(34,211,238,0.04) 1px, transparent 1px),
        var(--bg);
    background-size: auto, auto, 44px 44px, 44px 44px, auto;
    color: var(--ice);
    font-family: 'Exo 2', sans-serif;
}

[data-testid="stHeader"] { background: transparent; }

.block-container { padding-top: 2.5rem; max-width: 780px; }

/* Título con brillo */
h1 {
    font-family: 'Orbitron', sans-serif !important;
    font-weight: 800 !important;
    letter-spacing: 0.04em;
    line-height: 1.2 !important;
    background: linear-gradient(90deg, #FFFFFF 0%, var(--cyan) 55%, var(--blue) 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    filter: drop-shadow(0 0 14px rgba(34,211,238,0.45));
}

h2, h3 {
    font-family: 'Orbitron', sans-serif !important;
    font-weight: 600 !important;
    color: var(--ice) !important;
    letter-spacing: 0.03em;
}

p, label, li, span, .stMarkdown {
    font-family: 'Exo 2', sans-serif;
}

/* Insignia de versión de Python */
.py-badge {
    display: inline-block;
    padding: 4px 14px;
    border: 1px solid var(--line);
    border-radius: 999px;
    color: var(--cyan);
    background: rgba(10,22,51,0.7);
    font-size: 0.85rem;
    box-shadow: 0 0 12px rgba(34,211,238,0.18);
    margin-bottom: 1rem;
}

/* Imagen con marco luminoso */
[data-testid="stImage"] img {
    border-radius: 14px;
    border: 1px solid var(--line);
    box-shadow: 0 0 28px rgba(30,107,255,0.45);
}

/* Barra lateral */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #071029 0%, #0A1633 100%);
    border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] h3 { font-size: 1rem !important; color: var(--cyan) !important; }

/* Campos de texto */
.stTextInput input, .stTextArea textarea {
    background: var(--panel) !important;
    color: var(--ice) !important;
    border: 1px solid var(--line) !important;
    border-radius: 10px !important;
    font-family: 'Exo 2', sans-serif;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 0 1px var(--cyan), 0 0 18px rgba(34,211,238,0.35) !important;
}
.stTextArea textarea::placeholder, .stTextInput input::placeholder { color: var(--muted); }

/* Zona de carga de archivos */
[data-testid="stFileUploader"] section {
    background: var(--panel);
    border: 1.5px dashed var(--cyan);
    border-radius: 14px;
    transition: box-shadow .2s ease, background .2s ease;
}
[data-testid="stFileUploader"] section:hover {
    background: var(--panel-2);
    box-shadow: 0 0 22px rgba(34,211,238,0.28);
}
[data-testid="stFileUploader"] button {
    background: linear-gradient(90deg, var(--blue), var(--cyan));
    color: #031026;
    border: none;
    font-weight: 600;
}

/* Alertas (info, éxito, advertencia, error) */
[data-testid="stAlert"] {
    background: rgba(14,30,69,0.85);
    border: 1px solid var(--line);
    border-radius: 12px;
    color: var(--ice);
}

/* Bloque de respuesta (contenedor con borde) */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(135deg, rgba(14,30,69,0.95), rgba(10,22,51,0.95));
    border: 1px solid var(--line) !important;
    border-left: 4px solid var(--cyan) !important;
    border-radius: 12px !important;
    box-shadow: 0 0 26px rgba(30,107,255,0.28);
    line-height: 1.65;
}

/* Barra de desplazamiento */
::-webkit-scrollbar { width: 10px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--blue); border-radius: 8px; }

/* Respeta usuarios con movimiento reducido */
@media (prefers-reduced-motion: reduce) {
    * { transition: none !important; animation: none !important; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ------------------------------------------------------------------
# Encabezado
# ------------------------------------------------------------------
st.title("Generación Aumentada por Recuperación (RAG) 💬")
st.markdown(
    f'<span class="py-badge">Python {platform.python_version()}</span>',
    unsafe_allow_html=True,
)

# Imagen
try:
    image = Image.open("robot.jpg")
    st.image(image, width=350)
except Exception as e:
    st.warning(f"No se pudo cargar la imagen: {e}")

# Sidebar
with st.sidebar:
    st.subheader("Este Agente te ayudará a realizar análisis sobre el PDF cargado")

# API key
ke = st.text_input("Ingresa tu Clave de OpenAI", type="password")
if ke:
    os.environ["OPENAI_API_KEY"] = ke
else:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")

# Cargar PDF
pdf = st.file_uploader("Carga el archivo PDF", type="pdf")

if pdf is not None and ke:
    try:
        # Extraer texto
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += (page.extract_text() or "") + "\n"

        st.info(f"Texto extraído: {len(text)} caracteres")

        # Dividir en fragmentos
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=20,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)
        st.success(f"Documento dividido en {len(chunks)} fragmentos")

        # Embeddings y base de conocimiento
        embeddings = OpenAIEmbeddings(api_key=ke)
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # Pregunta del usuario
        st.subheader("Escribe qué quieres saber sobre el documento")
        user_question = st.text_area(
            "Tu pregunta",
            placeholder="Escribe tu pregunta aquí...",
            label_visibility="collapsed",
        )

        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            context = "\n\n".join(d.page_content for d in docs)

            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=ke)

            prompt = f"""Usa únicamente el siguiente contexto para responder la pregunta.
Si la respuesta no está en el contexto, di que no la sabes.

Contexto:
{context}

Pregunta: {user_question}

Respuesta:"""

            response = llm.invoke(prompt)

            st.markdown("### Respuesta")
            with st.container(border=True):
                st.markdown(response.content)

    except Exception as e:
        st.error(f"Error al procesar el PDF: {str(e)}")
        st.error(traceback.format_exc())
elif pdf is not None and not ke:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")
else:
    st.info("Por favor carga un archivo PDF para comenzar")
