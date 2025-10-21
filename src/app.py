import streamlit as st
from rag_pipeline import crear_qa_chain, cargar_documentos, crear_vectorstore, crear_llm
from config import HF_TOKEN

st.set_page_config(page_title="GUIA - Asistente UNLaR", page_icon="🎓")
st.title("🎓 GUIA - Asistente UNLaR")

# Inicializar sistema
if "qa_chain" not in st.session_state:
    with st.spinner("Cargando documentos y modelo..."):
        chunks = cargar_documentos()
        vectorstore = crear_vectorstore(chunks)
        llm = crear_llm()
        st.session_state.qa_chain = crear_qa_chain(vectorstore, llm)

# Historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        st.empty()

# Entrada del usuario (estilo chat)
if pregunta := st.chat_input("Escribí tu pregunta sobre la UNLaR..."):
    # Guardar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    # Generar respuesta del asistente
    with st.chat_message("assistant"):
        with st.spinner("Buscando la respuesta en los documentos..."):
            respuesta = st.session_state.qa_chain.invoke(pregunta)
            texto = respuesta["answer"]
            st.markdown(texto)

            # Mostrar fuentes opcionales
            with st.expander("📚 Ver fuentes"):
                for doc in respuesta["source_documents"]:
                    st.write(doc.metadata.get("source"))

    # Guardar mensaje del asistente
    st.session_state.messages.append({"role": "assistant", "content": texto})
