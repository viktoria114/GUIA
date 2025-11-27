#Guia
 
GUIA es un asistente virtual que responde dudas sobre la Universidad Nacional de La Rioja (UNLaR).  
Utiliza Retrieval-Augmented Generation (RAG) para buscar en documentos PDF oficiales.  
Está orientado a estudiantes y personal administrativo de la UNLaR.

## Características principales 
- Chat interactivo en tiempo real vía Streamlit.  
- Búsqueda de respuestas dentro de archivos PDF.  
- Memoria de conversación para mantener contexto.  
- Visualización opcional de las fuentes consultadas.

## Usabilidad
- Interfaz gráfica sencilla y amigable.  
- Detecta y responde en el idioma del usuario.  
- Soporta múltiples documentos en formato PDF.  
- Feedback al usuario mediante indicadores de carga.

## Funcionalidades clave  
- **Carga de documentos** desde la carpeta `docs/`.  
- **Fragmentación y limpieza** de texto (chunks) para embeddings.  
- **Creación de vectorstore** con Chroma y embeddings de HuggingFace.  
- **Configuración de LLM** mediante OpenRouter y Deepseek.  
- **Cadena RAG conversacional** con memoria de buffer.  
- **Interfaz de chat** que enlaza todo el pipeline.

## Tecnologías utilizadas  
**FrontEnd**  
- Streamlit  

**BackEnd**  
- Python 3.10+  
- LangChain & extensiones (`langchain-openai`, `langchain-community`)  
- ChromaDB  

**Librerías adicionales**  
- python-dotenv  
- HuggingFace Embeddings  
- PyMuPDF
- sentence-transformers, transformers  
- torch  

## Cómo Instalar y Usar  

### Clonar el repositorio  
```bash
git clone https://github.com/viktoria114/GUIA.git
cd GUIA
```

### Instalar dependencias  
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### Configurar variables de entorno  
Crear un archivo `.env` en la raíz y definir:  
```dotenv
OR_TOKEN="tu_token_de_openrouter"
```

### Ejecutar el proyecto localmente  
```bash
streamlit run src/app.py
```

## Estructura del Proyecto  
```
GUIA/
├── docs/                   # PDFs de la UNLaR
├── src/
│   ├── app.py              # Interfaz Streamlit
│   ├── config.py           # Carga de variables de entorno
│   └── rag_pipeline.py     # Lógica RAG/CAG
├── .env                    # Variables de entorno
├── requirements.txt        # Dependencias
└── README.txt              # Guía de instalación y notas
```

## Próximos Pasos  
- Añadir carga de formatos Excel y Word.  
- Implementar cache incremental de embeddings.  
- Soporte de múltiples modelos LLM configurables.  
- Panel de administración para actualizar documentos.  
- Autenticación de usuarios en la interfaz.

## Créditos y Despliegue  
Por: María Victoria Arancio Oviedo
GitHub: https://github.com/viktoria114/GUIA
