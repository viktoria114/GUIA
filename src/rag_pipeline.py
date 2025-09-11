# src/rag_pipeline.py
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from config import HF_TOKEN
import os

# 1. CARGAR DOCUMENTOS
def cargar_documentos():
    """Carga todos los PDFs de la carpeta docs/ y los divide en chunks"""
    try:
        loader = DirectoryLoader(
            "./docs", 
            glob="**/*.pdf", 
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        documentos = loader.load()
        
        # Dividir en chunks pequeños
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documentos)
        print(f"✅ Se cargaron {len(chunks)} chunks de documentos")
        return chunks
        
    except Exception as e:
        print(f"❌ Error cargando documentos: {e}")
        return []

# 2. CREAR BASE DE DATOS VECTORIAL
def crear_vectorstore(chunks):
    """Crea y guarda la base de datos vectorial con embeddings"""
    try:
        # Modelo de embeddings (gratis y eficiente)
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en",
            model_kwargs={"device": "cpu"},
                    )
        
        # Crear vectorstore
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="../chroma_db"  # Guarda para reusar
        )
        print("✅ Vectorstore creado y guardado")
        return vectorstore
        
    except Exception as e:
        print(f"❌ Error creando vectorstore: {e}")
        return None

# 3. CREAR LLM (Hugging Face)
def crear_llm():
    """Configura el modelo de lenguaje con Hugging Face"""
    try:
        llm = HuggingFaceEndpoint(
            repo_id="HuggingFaceH4/zephyr-7b-beta",  
            task="conversational",
            temperature=0.1,
            max_new_tokens=512,
            top_p=0.9,
            huggingfacehub_api_token=HF_TOKEN
        )
        print("✅ LLM configurado con HuggingFaceH4/zephyr-7b-beta")
        return llm
    except Exception as e:
        print(f"❌ Error configurando LLM: {e}")
        return None


# 4. CREAR MEMORIA PARA CAG
def crear_memoria():
    """Crea memoria para recordar el contexto de la conversación"""
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="result"
    )

# 5. CADENA DE CAG (¡Aquí está la magia!)
def crear_qa_chain(vectorstore, llm):
    """Crea la cadena de QA con soporte para modelos conversacionales"""
    try:
        memory = crear_memoria()

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            ),
            memory=memory,
            return_source_documents=True,
            verbose=True
        )
        print("✅ Cadena CAG creada con ConversationalRetrievalChain")
        return qa_chain

    except Exception as e:
        import traceback
        print(f"❌ Error creando cadena CAG: {e}")
        traceback.print_exc()
        return None

# 6. FUNCIÓN PARA PROBAR
def probar_sistema():
    """Función para probar que todo funciona"""
    print("🧪 Probando sistema...")
    chunks = cargar_documentos()
    if not chunks:
        return False
        
    vectorstore = crear_vectorstore(chunks)
    if not vectorstore:
        return False
        
    llm = crear_llm()
    if not llm:
        return False
        
    qa_chain = crear_qa_chain(vectorstore, llm)
    if not qa_chain:
        return False
        
    # Prueba simple
    try:
        respuesta = qa_chain.invoke("¿Cuales son las reglas mas importantes en la Unlar?")
        print("✅ Sistema funcionando:", respuesta["answer"][:100] + "...")
        return True
    except Exception as e:
        import traceback
        print("❌ Error en prueba:", str(e))
        traceback.print_exc()
        return False


# Ejecutar prueba si se corre este archivo directamente
if __name__ == "__main__":
    probar_sistema()