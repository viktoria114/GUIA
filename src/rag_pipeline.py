# src/rag_pipeline.py
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from config import HF_TOKEN
import os
import traceback

# 1. CARGAR DOCUMENTOS
def cargar_documentos():
    try:
        loader = DirectoryLoader(
            "./docs",
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        documentos = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=100,
            length_function=len
        )
        chunks = text_splitter.split_documents(documentos)
        print(f"✅ Se cargaron {len(chunks)} chunks de documentos")
        return chunks
    except Exception as e:
        print(f"❌ Error cargando documentos: {e}")
        return []

# 2. CREAR VECTORSTORE
def crear_vectorstore(chunks):
    try:
        embeddings = HuggingFaceEmbeddings(
               model_name="BAAI/bge-m3",               # 👈 nombre correcto
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True},
    cache_folder="./hf_models",             # opcional: para guardar local
        )
        print("✅ HuggingFaceEmbeddings inicializado correctamente.")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="../chroma_db"
        )
        print("✅ Vectorstore creado y guardado")
        return vectorstore
    except Exception as e:
        print(f"❌ Error creando vectorstore: {e}")
        traceback.print_exc()
        return None

# 3. CREAR LLM
def crear_llm():
    try:
        llm = ChatOpenAI(
            model="meta-llama/Llama-3.1-8B-Instruct:cerebras",
            temperature=0.1,
            max_tokens=1024,
            openai_api_key=HF_TOKEN,
            openai_api_base="https://router.huggingface.co/v1"
        )
        print("✅ LLM configurado con ChatOpenAI")
        return llm
    except Exception as e:
        print(f"❌ Error configurando LLM: {e}")
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

        # Prompt base con contexto de la UNLaR
        system_prompt = """
Eres un asistente experto en la Universidad Nacional de La Rioja (UNLaR). 
Proporciona respuestas precisas y claras basadas en documentos de la UNLaR. 
Si no sabes la respuesta exacta, indica que no tienes información suficiente.

Documentos relevantes: {context}

Pregunta del usuario: {question}
Respuesta:
"""

        prompt = PromptTemplate(
            template=system_prompt,
            input_variables=["context", "question"]  # OBLIGATORIO para StuffDocumentsChain
        )

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3, "lambda_mult": 0.7}),
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": prompt},
            verbose=True
        )
        print("✅ Cadena RAG creada con contexto UNLaR")
        return qa_chain
    except Exception as e:
        print(f"❌ Error creando cadena RAG: {e}")
        traceback.print_exc()
        return None

# 6. PROBAR SISTEMA
def probar_sistema():
    print("🧪 Probando sistema...")
    chunks = cargar_documentos()
    if not chunks: return False

    vectorstore = crear_vectorstore(chunks)
    if not vectorstore: return False

    llm = crear_llm()
    if not llm: return False

    qa_chain = crear_qa_chain(vectorstore, llm)
    if not qa_chain: return False

    try:
        respuesta = qa_chain.invoke({"question": "¿Cuales son los alcances del titulo de Ingeniero en sistemas?"})
        for doc in respuesta["source_documents"]:
         print(doc.metadata.get("source"))
        print("✅ Sistema funcionando:", respuesta["answer"][:300] + "...")
        return True
    except Exception as e:
        print("❌ Error en prueba:", str(e))
        traceback.print_exc()
        return False

if __name__ == "__main__":
    probar_sistema()
