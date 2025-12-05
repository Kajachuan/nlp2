# TP3: Chatbot con RAG, Agentes y Base de Datos Vectorial

Trabajo práctico que implementa un chatbot conversacional con búsqueda semántica (RAG), agentes y una base de datos vectorial para contexto relevante.

## Contenido

### 📁 Archivos principales

- `chatbot.py` — Aplicación web con Streamlit que permite interactuar con los documentos almacenados en `docs/`. Integra:
  - Memoria conversacional (LangChain)
  - Recuperación semántica vía Pinecone
  - Uso de embeddings (HuggingFace u otro proveedor configurado)
  - Llamadas a un LLM (por ejemplo Groq u otro proveedor configurado en el entorno)

- `agents.py` — Código para ejecutar experimentos o flujos con agentes: orquestación de herramientas, llamadas a modelos, y ejemplos de uso de agentes que combinan recuperación y ejecución de acciones.

- `pinecone.ipynb` — Notebook para preparar el índice vectorial:
  - Carga y procesamiento de PDFs (o documentos) desde `docs/`.
  - División en chunks, generación de embeddings y subida a Pinecone.

- `docs/` — Carpeta con subcarpetas por persona (por ejemplo `beatriz/`, `carlos/`, `kevin/`) que contienen los CV/documentos usados como fuente de conocimiento.

## Tecnologías

- LangChain — Gestión de conversación y memoria.
- Pinecone — Base de datos vectorial para búsqueda semántica.
- HuggingFace (o proveedor de embeddings) — Generación de embeddings.
- Groq (u otro LLM) — Modelo de lenguaje para generación y razonamiento.
- Streamlit — Interfaz web.

## Configuración y ejecución

1. Configurar variables de entorno necesarias (ejemplos):

```bash
export GROQ_API_KEY='tu-clave-groq'
export PINECONE_API_KEY='tu-clave-pinecone'
```

2. Preparar el índice vectorial (ejecute el notebook):

Abra `pinecone.ipynb` con Jupyter/Colab/VSCode y ejecute las celdas para procesar los documentos en `docs/` y subirlos a Pinecone.

3. Ejecutar el chatbot localmente:

```bash
streamlit run chatbot.py
```

4. Ejecutar ejemplos/agents:

```bash
python agents.py
```

## Notas

- Asegúrese de revisar las rutas dentro de `pinecone.ipynb` y `chatbot.py` para confirmar la ubicación de los documentos (carpeta `docs/`).
- Si desea cambiar el modelo LLM o el proveedor de embeddings, actualice las variables de entorno o la configuración dentro de los archivos.
- Este repositorio contiene subcarpetas en `docs/` con CVs de prueba; adapte el pipeline de ingesta si añade nuevos documentos o formatos.
