# RAG ChatBot con Streamlit y LangChain

Este proyecto implementa un chatbot basado en RAG (Retrieval-Augmented Generation) que permite hacer preguntas sobre documentos PDF utilizando Streamlit para la interfaz y LangChain para el procesamiento del lenguaje natural.

## 🚀 Características

- Carga y procesamiento de documentos PDF
- Interfaz de usuario intuitiva con Streamlit
- Sistema de chat con historial de conversación
- Visualización del grafo de procesamiento
- Almacenamiento persistente de embeddings

## 📋 Requisitos Previos

- Python 3.8 o superior
- uv instalado (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Ollama instalado y configurado
- Modelo llama3.2:3b descargado en Ollama
- Modelo nomic-embed-text:latest descargado en Ollama

## 🛠️ Instalación

1. Clonar el repositorio:
```bash
git clone <url-del-repositorio>
cd <nombre-del-directorio>
```

2. Inicializar el proyecto con uv:
```bash
uv venv
source .venv/bin/activate  # En Linux/Mac
# o
.venv\Scripts\activate     # En Windows
```

3. Instalar las dependencias del proyecto:
```bash
uv pip sync pyproject.toml
```

4. Crear un archivo `.env` en la raíz del proyecto con las siguientes variables:
```
LANGCHAIN_API_KEY=tu_api_key
USER_AGENT=tu_user_agent
```

## 🚀 Ejecución

1. Asegúrate de que Ollama esté corriendo en tu sistema
2. Ejecuta la aplicación:
```bash
uv run streamlit run src/app.py
```

## 📝 Ejemplos de Uso

![image](https://github.com/user-attachments/assets/fed0ea0f-7823-4ecd-acb3-66530b6a7776)


### Ejemplo 1: Consulta sobre ingresos
**Pregunta:** "¿Cuál es el enfoque principal de VoidLoop en sus soluciones de software?"
**Respuesta esperada:** "El enfoque principal de VoidLoop es construir herramientas que mejoren la eficiencia y productividad empresarial mediante el uso de agentes de inteligencia artificial y automatización de software. Su objetivo es crear soluciones innovadoras para mejorar los procesos empresariales. Esto se logra utilizando tecnologías como LLMs, RPA y NLP."

### Ejemplo 2: Consulta sobre crecimiento
**Pregunta:** "¿Qué tipos de proyectos o soluciones incluye el portafolio de VoidLoop?"
**Respuesta esperada:** "VoidLoop incluye en su portafolio proyectos y soluciones como asistentes IA para atención al cliente en banca, automatización de tareas administrativas con bots RPA y soluciones personalizadas con NLP y visión computacional. También ofrece soluciones modernas con tecnologías como LLMs, RPA, LangChain, FastAPI y despliegue en la nube. Estos proyectos están desarrollados por un equipo de ingenieros expertos en IA aplicada y automatización.

### Ejemplo 3: Consulta sobre segmentos
**Pregunta:** "¿Quiénes son los creadores de VoidLoop y qué experiencia aportan al equipo?"
**Respuesta esperada:** "No tengo información específica sobre los nombres de los creadores de VoidLoop. Sin embargo, puedo decir que el equipo es formado por ingenieros expertos en IA aplicada y automatización, con experiencia en startups y en la creación de soluciones innovadoras en el campo de la inteligencia artificial y la automatización del software."

## 📁 Estructura del Proyecto

```
.
├── src/
│   └── app.py
├── data/
│   └── (archivos PDF y visualizaciones)
├── chroma_langchain_db/
│   └── (base de datos de embeddings)
├── .env
├── pyproject.toml
└── README.md
```

## 🔧 Configuración Adicional

### Modelos de Ollama
Asegúrate de tener los siguientes modelos descargados en Ollama:
```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text:latest
```

### Ajustes de Rendimiento
- El tamaño de los chunks está configurado a 400 caracteres con un solapamiento de 50
- Se recuperan los 2 documentos más relevantes para cada consulta
- La temperatura del modelo está configurada a 0.6 para un balance entre creatividad y precisión

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue para discutir los cambios propuestos.

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.
