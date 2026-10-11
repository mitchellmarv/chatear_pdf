import os
import platform
import traceback

import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

# Título y presentación
st.title('Generación Aumentada por Recuperación (RAG) 💬')
st.write("Versión de Python:", platform.python_version())

# Cargar imagen
try:
    image = Image.open('Chat_pdf.png')
    st.image(image, width=350)
except Exception as e:
    st.warning(f"No se pudo cargar la imagen: {e}")

# Sidebar
with st.sidebar:
    st.subheader("Este Agente te ayudará a realizar análisis sobre el PDF cargado")

# API key
ke = st.text_input('Ingresa tu Clave de OpenAI', type="password")
if ke:
    os.environ['OPENAI_API_KEY'] = ke
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
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        st.success(f"Documento dividido en {len(chunks)} fragmentos")

        # Embeddings y base de conocimiento
        embeddings = OpenAIEmbeddings(api_key=ke)
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # Pregunta del usuario
        st.subheader("Escribe qué quieres saber sobre el documento")
        user_question = st.text_area(" ", placeholder="Escribe tu pregunta aquí...")

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

            st.markdown("### Respuesta:")
            st.markdown(response.content)

    except Exception as e:
        st.error(f"Error al procesar el PDF: {str(e)}")
        st.error(traceback.format_exc())
elif pdf is not None and not ke:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")
else:
    st.info("Por favor carga un archivo PDF para comenzar")
