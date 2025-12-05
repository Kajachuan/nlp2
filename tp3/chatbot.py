import streamlit as st
from agents import CVMultiAgentSystem

# ------------------------------------------------------------
# 1. INICIALIZACIÓN DEL SISTEMA (UNA SOLA VEZ)
# ------------------------------------------------------------
@st.cache_resource
def initialize_system():
    """Inicializa el sistema una sola vez y lo cachea"""
    names = ["beatriz", "carlos", "kevin"]
    return CVMultiAgentSystem(agent_names=names)

# Inicializar el sistema (solo una vez, se cachea)
system = initialize_system()

# ------------------------------------------------------------
# 2. CONFIGURACIÓN DE LA APLICACIÓN
# ------------------------------------------------------------
st.set_page_config(
    page_title="Chatbot Multi-agente de CVs",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Chatbot Multi-agente de CVs")
st.markdown("""
Consulta información de CVs de diferentes personas. 
Puedes preguntar por una persona específica o por varias.
""")

# Sidebar con información
with st.sidebar:
    st.header("ℹ️ Información")
    st.markdown("""
    **Agentes disponibles:**
    - Beatriz
    - Carlos  
    - Kevin (default)
    
    **Ejemplos de preguntas:**
    - "Hola, ¿cómo estás?" (Kevin responde)
    - "Beatriz, ¿cuál es tu experiencia?"
    - "¿Qué saben Beatriz y Carlos de Python?"
    """)
    
    # Mostrar estado del sistema
    st.header("🔄 Estado del sistema")
    st.success("✅ Sistema inicializado correctamente")
    
    # Botón para limpiar historial
    if st.button("🗑️ Limpiar conversación"):
        st.session_state.history = []
        st.rerun()

# ------------------------------------------------------------
# 3. INICIALIZAR HISTORIAL
# ------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ------------------------------------------------------------
# 4. MOSTRAR HISTORIAL COMPLETO
# ------------------------------------------------------------
for role, message in st.session_state.history:
    with st.chat_message(role):
        st.markdown(message)

# ------------------------------------------------------------
# 5. INPUT DEL USUARIO
# ------------------------------------------------------------
user_input = st.chat_input("Escribe tu consulta sobre los CVs...")

if user_input:
    # Guardar mensaje del usuario en historial
    st.session_state.history.append(("user", user_input))
    
    # Mostrar mensaje del usuario
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Mostrar indicador de procesamiento
    with st.chat_message("assistant"):
        with st.spinner("Procesando tu consulta..."):
            # Ejecutar el sistema
            final_answer = system.ask(user_input)
            st.markdown(final_answer)
    
    # Guardar respuesta en historial
    st.session_state.history.append(("assistant", final_answer))