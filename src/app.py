# src/app.py
import streamlit as st
from rag_pipeline import crear_qa_chain, cargar_documentos, crear_vectorstore, crear_llm
from config import HF_TOKEN

st.title("GUIA - Asistente UNLaR")

# Inicializar sistema
if "qa_chain" not in st.session_state:
    with st.spinner("Cargando documentos y modelo..."):
        chunks = cargar_documentos()
        vectorstore = crear_vectorstore(chunks)
        llm = crear_llm()
        st.session_state.qa_chain = crear_qa_chain(vectorstore, llm)

# Input de usuario
pregunta = st.text_input("Pregunta sobre la UNLaR:")

if pregunta:
    respuesta = st.session_state.qa_chain.invoke({"question": pregunta})
    st.write("**Respuesta:**", respuesta["answer"])
    
    # Mostrar fuentes
    with st.expander("Ver fuentes"):
        for doc in respuesta["source_documents"]:
            st.write(doc.page_content[:500] + "...")
    
    # 🔎 Mostrar historial de conversación
    st.write("### Historial")
    st.write(st.session_state.qa_chain.memory.chat_memory.messages)
