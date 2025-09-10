import os
from dotenv import load_dotenv

load_dotenv()  # Carga .env

HF_TOKEN = os.getenv("HF_TOKEN")