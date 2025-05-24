# Atom.AI-RAG

Atom.AI-RAG es una aplicación de chat interactiva que utiliza técnicas de RAG (Retrieval-Augmented Generation) para responder preguntas basadas en documentos PDF. La aplicación está construida con Streamlit, LangChain y Ollama, ofreciendo una interfaz amigable para cargar documentos y realizar consultas sobre su contenido.

## 🚀 Características

- Carga y procesamiento de documentos PDF
- Interfaz de chat interactiva
- Visualización del grafo de estados del sistema
- Sistema de RAG implementado con LangChain y Ollama
- Almacenamiento vectorial con Chroma
- Interfaz de usuario intuitiva con Streamlit

## 📋 Prerrequisitos

- Python 3.8 o superior
- [uv](https://github.com/astral-sh/uv) para la gestión de entornos virtuales
- [Ollama](https://ollama.ai/) instalado y configurado
- Modelos de Ollama:
  - llama3.2:3b
  - nomic-embed-text:latest

## 🛠️ Instalación

1. Clonar el repositorio:
```bash
git clone <url-del-repositorio>
cd Atom.AI-RAG
```

2. Crear y activar el entorno virtual con uv:
```bash
uv venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
```

3. Instalar dependencias:
```bash
uv pip install -r requirements.txt
```

4. Configurar variables de entorno:
Crear un archivo `.env` en la raíz del proyecto con las siguientes variables:
