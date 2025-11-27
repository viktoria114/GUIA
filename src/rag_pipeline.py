# src/rag_pipeline.py
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from config import OR_TOKEN
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
import traceback

# 1. CARGAR DOCUMENTOS
def cargar_documentos():
    try:
        loader = DirectoryLoader(
            "./docs",
            glob="**/*.pdf",
            loader_cls=PyMuPDFLoader,
            show_progress=True
        )
        documentos = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=300,
            length_function=len
        )
        chunks = text_splitter.split_documents(documentos)
        
        print(f" Se generaron {len(chunks)} chunks (antes de limpiar duplicados)")

        # ----- Limpieza -----
        unique_chunks = []
        seen_content_normalized = set()

        for chunk in chunks:  # Normalizamos el texto: quitamos espacios al inicio/final
            normalized_content = chunk.page_content.strip().lower() 
            if len(normalized_content) < 50: # Ignoramos chunks muy cortos 
                 continue
            if normalized_content not in seen_content_normalized:
                unique_chunks.append(chunk)
                seen_content_normalized.add(normalized_content)
        
        print(f" Se cargarán {len(unique_chunks)} chunks ÚNICOS (después de limpiar)")

        return unique_chunks # Devolvemos la lista limpia
    
    except Exception as e:
        print(f" Error cargando documentos: {e}")
        return []

# 2. CREAR VECTORSTORE
def crear_vectorstore(chunks):
    try:
        embeddings = HuggingFaceEmbeddings(
               model_name="BAAI/bge-m3",              
    model_kwargs={'device': 'cuda'},  # o 'cpu' 
    encode_kwargs={'normalize_embeddings': True},
    cache_folder="./hf_models",            
        )
        print(" HuggingFaceEmbeddings inicializado correctamente.")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        print(" Vectorstore creado y guardado")
        return vectorstore
    except Exception as e:
        print(f" Error creando vectorstore: {e}")
        traceback.print_exc()
        return None

# 3. CREAR LLM
def crear_llm():
    try:
        llm = ChatOpenAI(
            model="deepseek/deepseek-chat-v3.1",
            temperature=0.1,
            max_tokens=1024,
            openai_api_key=OR_TOKEN,
            openai_api_base="https://openrouter.ai/api/v1"
        )
        print(" LLM configurado con OpenRouter Deepseek")
        return llm
    except Exception as e:
        print(f" Error configurando LLM: {e}")
        return None

# 4. MEMORIA
def crear_memoria():
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        chat_memory=ChatMessageHistory()
    )

# 5. CADENA DE QA CON CONTEXTO UNLaR
def crear_qa_chain(vectorstore, llm):
    try:
        memory = crear_memoria()
        historial = memory.chat_memory.messages
        print(f" Memoria: {historial}")


        # Prompt base con contexto de la UNLaR
        system_prompt = """
Eres un asistente virtual diseñado para ayudar a estudiantes de la Universidad Nacional de La Rioja (UNLaR).
Tu función principal es responder preguntas y brindar asistencia sobre temas académicos, administrativos y de la vida universitaria en la UNLaR.
=== OBJETIVOS ===
- Brindar respuestas precisas y fieles al contenido de los documentos proporcionados.
- Incluir toda la información relevante encontrada en las fuentes, priorizando la claridad y la utilidad para el estudiante.
- Mantener una comunicación cercana y amigable, usando un tono cordial y accesible para un público joven.
- Detecta automáticamente el idioma en que el usuario escribe su pregunta, y responde en ese mismo idioma, sin traducirlo al español por defecto.

=== REGLAS DE RESPUESTA ===
1. Presenta las respuestas en forma de resumen explicativo. 
   - Usa párrafos para explicaciones generales.
   - Emplea listas numeradas o con viñetas solo cuando la información se preste a enumeraciones claras.
2. No uses citas textuales a menos que el usuario solicite explícitamente el texto exacto.
3. Si no encuentras información suficiente en los documentos, responde de manera clara que no cuentas con esa información. 
   - Sugiere amablemente al usuario consultar la página oficial de la UNLaR, contactar a la universidad por sus medios oficiales o acudir presencialmente a las oficinas para confirmar.
4. Si la pregunta no está relacionada con la UNLaR o su contexto educativo/administrativo, informa al usuario que está fuera de tu propósito principal.
5. Adapta el nivel de detalle según la pregunta:
   - Para consultas breves, responde de forma concisa pero completa.
   - Para consultas más complejas, responde de forma más detallada, sin omitir información relevante.
6. Nunca pidas datos personales ni sugieras que el usuario los proporcione.
7. Si el usuario no solicita información específica, responde cortésmente sin forzar una búsqueda en los documentos.

=== CONTEXTO DISPONIBLE ===
La siguiente información ha sido extraída de documentos oficiales de la UNLaR. Utiliza este contexto para responder a las preguntas del usuario de la mejor manera posible.
{context}

=== INSTRUCCIÓN ===
Pregunta del usuario: {question}

Tu respuesta debe seguir fielmente estas reglas y objetivos.
"""

        prompt = PromptTemplate(
            template=system_prompt,
            input_variables=["context", "question"] 
        )

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_type="mmr", 
            search_kwargs={"k": 4, 'fetch_k': 20}), 
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": prompt},
            verbose=True
        )
        print(" Cadena RAG creada con contexto UNLaR")
        return qa_chain
    except Exception as e:
        print(f" Error creando cadena RAG: {e}")
        traceback.print_exc()
        return None

# 6. PROBAR SISTEMA
def probar_sistema():
    print(" Probando sistema...")
    chunks = cargar_documentos()
    if not chunks: return False

    vectorstore = crear_vectorstore(chunks)
    if not vectorstore: return False

    llm = crear_llm()
    if not llm: return False

    qa_chain = crear_qa_chain(vectorstore, llm)
    if not qa_chain: return False

    try:
        respuesta = qa_chain.invoke({"question": "¿Cuales son los alcances del titulo de Licenciatura en sistemas?"})
        for doc in respuesta["source_documents"]:
         print(doc.metadata.get("source"))
        print(" Sistema funcionando:", respuesta["answer"][:300] + "...")
        return True
    except Exception as e:
        print(" Error en prueba:", str(e))
        traceback.print_exc()
        return False
    
    

if __name__ == "__main__":
    probar_sistema()
