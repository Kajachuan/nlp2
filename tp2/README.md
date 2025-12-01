# TP2: Chatbot con RAG y Base de Datos Vectorial

Trabajo práctico que implementa un chatbot conversacional con búsqueda semántica en documentos.

## Contenido

### 📁 Archivos principales

- **`chatbot.py`** - Aplicación web interactiva con Streamlit que proporciona una interfaz para consultar documentos del CV. Integra:
  - Memoria conversacional mediante LangChain
  - Modelo LLM de Groq (llama-3.1-8b-instant)
  - Base de datos vectorial Pinecone para contexto relevante
  - Embeddings de HuggingFace

- **`pinecone.ipynb`** - Notebook que prepara la base de datos vectorial:
  - Carga PDFs de documentos (CV) desde `docs/cv/`
  - Divide documentos en chunks procesables
  - Crea índice en Pinecone con embeddings
  - Configura el pipeline RAG (Retrieval-Augmented Generation)

### 📚 Tecnologías

- **LangChain**: Gestión de conversaciones y cadenas de procesamiento
- **Groq**: Proveedor de modelos LLM rápidos
- **Pinecone**: Base de datos vectorial para búsqueda semántica
- **HuggingFace**: Embeddings (all-MiniLM-L6-v2)
- **Streamlit**: Interfaz web interactiva

## Ejecución

### Configuración inicial

```bash
# Variables de entorno necesarias:
export GROQ_API_KEY='tu-clave-aqui'
export PINECONE_API_KEY='tu-clave-aqui'
```

### Preparar índice vectorial

Ejecutar `pinecone.ipynb` para cargar documentos en Pinecone.

### Iniciar el chatbot

```bash
streamlit run chatbot.py
```

El chatbot responderá preguntas basadas en el contenido de los documentos del CV, manteniendo contexto de conversaciones anteriores.
