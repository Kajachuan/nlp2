"""
Chatbot con Memoria Persistente usando LangChain, Pinecone y Groq
=====================================================

Este archivo implementa un chatbot con interfaz web usando Streamlit que:
- Mantiene memoria de conversaciones anteriores
- Utiliza diferentes modelos de LLM a través de Groq
- Permite personalización del comportamiento del bot
- Gestiona la memoria conversacional automáticamente

Tecnologías utilizadas:
- Streamlit: Para la interfaz web
- LangChain: Para gestión de memoria y cadenas de conversación
- Groq: Como proveedor de modelos LLM
- Pinecone: Base de datos vectorial para contexto relevante
- Python: Lenguaje de programación

Instrucciones para ejecutar:
    streamlit run chatbot.py

Variables de entorno necesarias:
    GROQ_API_KEY: Tu clave API de Groq (obtener en https://console.groq.com)
    PINECONE_API_KEY: Tu clave API de Pinecone (obtener en https://www.pinecone.io/)
"""

# ========================================
# IMPORTACIÓN DE LIBRERÍAS NECESARIAS
# ========================================

import streamlit as st           # Framework para crear aplicaciones web interactivas
import os                      # Para acceso a variables de entorno

# Importaciones específicas de LangChain para gestión de conversaciones

from langchain_core.prompts import (
    ChatPromptTemplate,           # Template para estructurar mensajes de chat
    MessagesPlaceholder,          # Marcador de posición para el historial
)
from langchain_groq import ChatGroq              # Integración LangChain-Groq
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings  # Modelo de embeddings
from langchain_pinecone import PineconeVectorStore # Integración LangChain-Pinecone
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

def main():
    """
    Función principal de la aplicación de chatbot.
    
    Esta función coordina todos los componentes del chatbot:
    1. Configuración de la interfaz de usuario
    2. Gestión de la memoria conversacional
    3. Integración con el modelo de lenguaje
    4. Procesamiento de preguntas y respuestas
    
    Funcionalidades principales:
    - Interfaz web responsiva con Streamlit
    - Memoria de conversación con longitud configurable
    - Selección de diferentes modelos LLM
    - Personalización del prompt del sistema
    - Historial persistente durante la sesión
    """
    
    # ========================================
    # CONFIGURACIÓN INICIAL Y AUTENTICACIÓN
    # ========================================
    
    # Obtener la clave API de Groq desde las variables de entorno
    # Esto es una práctica de seguridad recomendada para no exponer credenciales en el código
    groq_api_key = os.getenv('GROQ_API_KEY')
    
    # Verificar si la clave API está configurada
    if not groq_api_key:
        st.error("⚠️ GROQ_API_KEY no está configurada en las variables de entorno")
        st.info("💡 Configura tu clave API: export GROQ_API_KEY='tu-clave-aqui'")
        st.stop()  # Detener la ejecución si no hay clave API

    pinecone_api_key = os.getenv('PINECONE_API_KEY')

    if not pinecone_api_key:
        st.error("⚠️ PINECONE_API_KEY no está configurada en las variables de entorno")
        st.info("💡 Configura tu clave API: export PINECONE_API_KEY='tu-clave-aqui'")
        st.stop()  # Detener la ejecución si no hay clave API

    # ========================================
    # CONFIGURACIÓN DE LA INTERFAZ PRINCIPAL
    # ========================================
    
    # Configurar el título y descripción de la aplicación
    st.title("🤖 Chatbot del CV de Kevin Cajachuán")
    st.markdown("""
    **¡Bienvenido al chatbot!** 
    
    Este chatbot utiliza:
    - 🧠 **Memoria conversacional**: Recuerda el contexto de tu conversación
    - 🔄 **Modelos intercambiables**: Puedes elegir diferentes LLMs
    - ⚙️ **Personalización**: Configura el comportamiento del asistente
    - 🚀 **Powered by Groq**: Respuestas rápidas y precisas
    - 📚 **Base de datos vectorial**: Contexto relevante para mejores respuestas
    """)

    # ========================================
    # PANEL DE CONFIGURACIÓN LATERAL
    # ========================================
    
    st.sidebar.title('⚙️ Configuración del Chatbot')
    st.sidebar.markdown("---")
    
    model = 'llama-3.1-8b-instant'
    
    # Control deslizante para la longitud de memoria
    st.sidebar.subheader("🧠 Configuración de Memoria")
    conversational_memory_length = st.sidebar.slider(
        'Longitud de la memoria conversacional:', 
        min_value=1, 
        max_value=10, 
        value=5,
        help="Número de intercambios anteriores que el bot recordará. Más memoria = mayor contexto pero mayor costo computacional"
    )
    
    # Mostrar información sobre la memoria
    st.sidebar.caption(f"💭 El bot recordará los últimos {conversational_memory_length} intercambios")

    # ========================================
    # CONFIGURACIÓN DE LA MEMORIA CONVERSACIONAL
    # ========================================
    
    # Nueva API: gestionamos historial con RunnableWithMessageHistory + InMemoryChatMessageHistory
    if "session_id" not in st.session_state:
        st.session_state.session_id = "default"
    if "history_store" not in st.session_state:
        st.session_state.history_store = {}
    
    # ========================================
    # GESTIÓN DEL HISTORIAL DE CONVERSACIÓN
    # ========================================
    
    # Inicializar el historial de chat en el estado de la sesión de Streamlit
    # st.session_state permite mantener datos entre ejecuciones de la aplicación
    if 'historial_chat' not in st.session_state:
        st.session_state.historial_chat = []
        st.sidebar.success("💬 Nueva conversación iniciada")
    else:
        # Mostrar información del historial en la barra lateral
        st.sidebar.info(f"💬 Conversación con {len(st.session_state.historial_chat)} mensajes")
    
    # Botón para limpiar el historial
    if st.sidebar.button("🗑️ Limpiar Conversación"):
        st.session_state.historial_chat = []
        # Reiniciar historial de LangChain para la sesión actual
        sid = st.session_state.session_id
        if sid in st.session_state.history_store:
            st.session_state.history_store[sid] = InMemoryChatMessageHistory()
        st.sidebar.success("✅ Conversación limpiada")
        st.rerun()  # Recargar la aplicación
    
    # ========================================
    # INTERFAZ DE ENTRADA DEL USUARIO
    # ========================================
    
    # Crear el campo de entrada para las preguntas del usuario
    user_question = st.chat_input("Escribe tu mensaje aquí...")

    # ========================================
    # CONFIGURACIÓN DEL MODELO DE LENGUAJE
    # ========================================
    
    # Inicializar el cliente de ChatGroq con las configuraciones seleccionadas
    try:
        groq_chat = ChatGroq(
            groq_api_key=groq_api_key,     # Clave API para autenticación
            model_name=model,              # Modelo seleccionado por el usuario
            temperature=0.7,               # Creatividad de las respuestas (0=determinista, 1=creativo)
            max_tokens=1000,               # Máximo número de tokens en la respuesta
        )
        st.sidebar.success("✅ Modelo conectado correctamente")
    except Exception as e:
        st.sidebar.error(f"❌ Error al conectar con Groq: {str(e)}")
        st.stop()

    # ========================================
    # CONFIGURACIÓN DEL EMBEDDING
    # ========================================

    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # ========================================
    # CONFIGURACIÓN DE LA DB VECTORIAL
    # ========================================

    index_name = 'kajachuan'
    namespace = "nlp2"

    vectorstore = PineconeVectorStore(
        pinecone_api_key = pinecone_api_key,
        index_name=index_name,
        embedding=embed_model,
        namespace=namespace,
    )
    retriever=vectorstore.as_retriever()

    # ========================================
    # PROCESAMIENTO DE LA PREGUNTA Y RESPUESTA
    # ========================================

    # Si el usuario ha hecho una pregunta,
    if user_question:

        # Mostrar indicador de carga mientras se procesa
        with st.spinner('🤔 El chatbot está pensando...'):
            
            try:
                system_prompt = (
                    "Eres un asistente para tareas de preguntas y respuestas. "
                    "Usa las siguientes partes del contexto recuperado para responder"
                    "la pregunta. Si no sabes la respuesta, di que no lo sabes."
                    "Usa un máximo de tres oraciones y mantén la respuesta concisa."
                    "\n\n"
                    "{context}"
                )
                # ========================================
                # CONSTRUCCIÓN DEL TEMPLATE DE CONVERSACIÓN
                # ========================================
                
                # Crear un template de chat que incluye:
                # 1. Mensaje del sistema (personalidad/instrucciones)
                # 2. Historial de conversación (memoria)
                # 3. Mensaje actual del usuario
                prompt = ChatPromptTemplate.from_messages([
                    
                    # Mensaje del sistema - Define el comportamiento del chatbot
                    ("system", system_prompt),
                    
                    # Marcador de posición para el historial - Se reemplaza automáticamente
                    MessagesPlaceholder(variable_name="historial_chat"),
                    
                    # Template para el mensaje actual del usuario
                    ("human", "{input}")
                ])
                
                # ========================================
                # CREACIÓN DE LA CADENA DE CONVERSACIÓN
                # ========================================
                
                question_answer_chain = create_stuff_documents_chain(groq_chat, prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                
                # Obtener/crear historial de la sesión actual
                session_id = st.session_state.session_id
                store = st.session_state.history_store
                if session_id not in store:
                    store[session_id] = InMemoryChatMessageHistory()
                
                # Envolver con memoria conversacional usando la nueva API
                chain_with_memory = RunnableWithMessageHistory(
                    rag_chain,
                    lambda sid: store.setdefault(sid, InMemoryChatMessageHistory()),
                    input_messages_key="input",
                    history_messages_key="historial_chat",
                    output_messages_key="answer",
                )
                
                # ========================================
                # GENERACIÓN DE LA RESPUESTA
                # ========================================
                
                # Enviar la pregunta al modelo y obtener la respuesta
                result = chain_with_memory.invoke(
                    {"input": user_question},
                    config={"configurable": {"session_id": session_id}},
                )
                response = result["answer"]
                
                # ========================================
                # ALMACENAMIENTO Y VISUALIZACIÓN
                # ========================================
                
                st.session_state.historial_chat.append({"role": "user", "content": user_question})
                st.session_state.historial_chat.append({"role": "bot", "content": response})
                
                # ========================================
                # MOSTRAR LA CONVERSACIÓN
                # ========================================
                
                # Mostrar la respuesta actual destacada
                st.markdown("### 💬 Conversación:")

                for msg in st.session_state.historial_chat:
                    role = msg["role"]
                    content = msg["content"]

                    if role == "user":
                        st.chat_message("user").markdown(content)
                    else:
                        st.chat_message("assistant").markdown(content)
                
                # Información adicional sobre la respuesta
                st.caption(f"📊 Modelo: {model} | 🧠 Memoria: {conversational_memory_length} mensajes")
                
            except Exception as e:
                # Manejo de errores durante el procesamiento
                st.error(f"❌ Error al procesar la pregunta: {str(e)}")
                st.info("💡 Verifica tu conexión a internet y la configuración de la API")


if __name__ == "__main__":
    # Punto de entrada de la aplicación
    # Solo ejecutar main() si este archivo se ejecuta directamente
    main()
