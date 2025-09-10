- solicitamos una arquitectura u organizacion de carpetasa deepseek y nos dio:
GUIA/
├── docs/                   # Documentos de la UNLaR (PDFs, Excel, etc.)
├── data/                   # Datos procesados (opcional)
├── src/
│   ├── app.py              # Interfaz con Streamlit
│   ├── rag_pipeline.py     # Lógica de RAG/CAG
│   └── config.py           # Configuración de APIs
├── .env                    # Variables de entorno (NO subir a GitHub)
└── requirements.txt        # Dependencias

- las creamos e instalamos. Phyton vs 3.10 o mas.
- creamos el entorno virtual:
python -m venv guia_env
guia_env\Scripts\activate

- luego instalamos las bibliotecas que elegimos:
# Instalar dependencias básicas
pip install langchain chromadb streamlit huggingface-hub

# Para manejar documentos (PDFs, Excel, etc.)
pip install pypdf openpyxl python-docx

# Modelos de embeddings (vectores de texto)
pip install sentence-transformers

# Cliente para APIs de LLMs (DeepSeek-V3 u otros)
pip install requests transformers huggingface_hub

# Para procesar imágenes/PDFs escaneados (opcional)
pip install donut-python 

- configuramos .env con el token propio de cada integrante para que no nos gastemos tan rapido las respuestas gratis
HF_TOKEN="tu_token_de_hugging_face"

- configuramos el requirements.txt para hacer la instalacion mas sencilla con todas las dependencias:


python -m venv guia_env
guia_env\Scripts\activate
pip install -r requirements.txt

PERO NO FUNCIONO aaaa

PROCEDEMOS A la Eliminación temporal de requirements.txt
Problema identificado: El archivo requirements.txt generado automáticamente contenía dependencias conflictivas entre paquetes, específicamente:

Conflictos de versiones de Pydantic:

langchain==0.3.27 requiere pydantic>=2.7.4

chromadb==0.4.0 requiere pydantic<2.0

Incompatibilidad irreconciliable

Problemas de compilación de NumPy:

numpy==1.26.4 requiere herramientas de compilación (Visual Studio Build Tools)

Entorno de desarrollo sin compiladores C++ instalados

Error de metadata generation durante la instalación

Dependencias excesivas:

El requirements.txt original contenía +150 paquetes

Muchos de ellos eran dependencias transitivas innecesarias

Riesgo alto de conflictos y sobrecarga del entorno

Solución implementada:
✅ Instalación manual controlada con versiones específicas compatibles:

pip install "langchain==0.1.17" "chromadb==0.4.22" "streamlit==1.49.1" "huggingface-hub==0.34.4" "pypdf==6.0.0" "openpyxl==3.1.5" "python-docx==1.2.0" "sentence-transformers==5.1.0" "transformers==4.56.1" "torch==2.8.0" "python-dotenv==1.1.1" "requests==2.32.5" "numpy==1.24.3"

DESCUBRIMOS Q TODOS LOS PROBLEMAS DE Incompatibilidad SON POR EL PYTHON 13, HABIA Q USAR EL 11

py -3.11 -m venv guia_env
pip install "langchain==0.1.17" "chromadb==0.4.22" "streamlit==1.49.1" "huggingface-hub==0.34.4" "pypdf==6.0.0" "openpyxl==3.1.5" "python-docx==1.2.0" "sentence-transformers==5.1.0" "transformers==4.56.1" "torch==2.8.0" "python-dotenv==1.1.1" "requests==2.32.5" "numpy==1.24.3"