import os
import uuid
from typing import TypedDict, List, Dict, Optional
from langgraph.graph import StateGraph, END
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ============================================================================
# 1. DEFINICIÓN DEL ESTADO
# ============================================================================

class AgentState(TypedDict):
    question: str
    original_question: str
    target_agents: List[str]
    agent_count: int
    query_type: str
    agent_responses: Dict[str, str]
    final_response: Optional[str]
    current_step: str
    conversation_id: str

# ============================================================================
# 2. CLASE PARA MANEJAR VECTORSTORES
# ============================================================================

class VectorStoreManager:
    def __init__(self):
        self.embed_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.index_name = "cv-index"
        self.api_key = os.getenv("PINECONE_API_KEY")
        
    def get_vectorstore(self, namespace: str) -> PineconeVectorStore:
        return PineconeVectorStore(
            pinecone_api_key=self.api_key,
            index_name=self.index_name,
            embedding=self.embed_model,
            namespace=namespace
        )
    
# ============================================================================
# 3. AGENTE ESPECIALIZADO
# ============================================================================

class CVSpecialistAgent:
    def __init__(self, agent_name: str, vectorstore_manager: VectorStoreManager, all_agents: List[str]):
        self.agent_name = agent_name
        self.vs_manager = vectorstore_manager
        self.all_agents = all_agents
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant",
            temperature=0
        )
        
        self.system_template = f"""
        Eres {agent_name}, un profesional respondiendo preguntas sobre tu propio CV.
        
        INSTRUCCIONES ESTRICTAS:
        1. SOLO puedes hablar sobre ti mismo ({agent_name})
        2. Si la pregunta menciona a otras personas, IGNÓRALAS COMPLETAMENTE
        3. Si no tienes información sobre algo, di "No tengo esa información en mi CV"
        4. Sé conciso, profesional y específico
        5. Responde en tercera persona
        
        Usa la siguiente información de tu CV para responder:
        {{context}}
        """
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_template),
            ("user", "{question}")
        ])
    
    def respond(self, question: str) -> str:
        """Método principal del agente para responder preguntas"""
        try:
            # Adaptar la pregunta
            adapted_question = self._adapt_question(question)
            
            # Obtener documentos relevantes
            vectorstore = self.vs_manager.get_vectorstore(self.agent_name)
            retriever = vectorstore.as_retriever()
            
            # Crear la cadena de RAG
            chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
            
            # Ejecutar
            response = chain.invoke(adapted_question)
            return response
            
        except Exception as e:
            return f"Error en agente {self.agent_name}: {str(e)}"
    
    def _adapt_question(self, question: str) -> str:
        """Adapta la pregunta para que el agente responda solo sobre sí mismo"""
        question_lower = question.lower()
        
        # Filtrar la lista de agentes para remover todos menos el actual
        agents_to_remove = [agent for agent in self.all_agents if agent != self.agent_name]
        
        # Remover menciones a otros agentes
        for agent in agents_to_remove:
            question_lower = question_lower.replace(agent, "")
        
        return f"{question_lower.strip()}\n\nResponde ÚNICAMENTE sobre {self.agent_name}."
    
# ============================================================================
# 4. ROUTER
# ============================================================================

class CVAgentRouter:
    def __init__(self, available_agents: List[str], default_agent: str = "kevin"):
        self.available_agents = available_agents
        self.default_agent = default_agent
        
    def route(self, state: AgentState) -> AgentState:
        """Analiza la pregunta y determina a qué agentes dirigirla"""
        question = state["question"].lower()
        
        # Detectar agentes mencionados
        detected_agents = []
        for agent in self.available_agents:
            if agent != "default" and agent in question:
                detected_agents.append(agent)
        
        # Decisión del router
        if not detected_agents:
            state["target_agents"] = [self.default_agent]
            state["agent_count"] = 1
            state["query_type"] = "default"
        elif len(detected_agents) == 1:
            state["target_agents"] = detected_agents
            state["agent_count"] = 1
            state["query_type"] = "single"
        else:
            state["target_agents"] = detected_agents
            state["agent_count"] = len(detected_agents)
            state["query_type"] = "multiple"
        
        state["current_step"] = "routed"
        return state
    
# ============================================================================
# 5. CONSOLIDADOR DE RESPUESTAS
# ============================================================================

class ResponseConsolidator:
    def __init__(self):
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant",
            temperature=0
        )
        
        self.consolidation_prompt = ChatPromptTemplate.from_messages([
            ("system", """Eres un consolidador experto. Tu trabajo es combinar respuestas de múltiples personas.
            
            INSTRUCCIONES:
            1. Mantén la esencia de cada respuesta individual
            2. Organiza la información de forma clara y estructurada
            3. Resalta similitudes y diferencias cuando sea relevante
            4. Responde completamente a la pregunta original
            5. Mantén un tono profesional y coherente"""),
            ("user", """PREGUNTA ORIGINAL: {question}

            RESPUESTAS INDIVIDUALES:
            {responses_text}
            
            Por favor, proporciona una respuesta consolidada que combine esta información:""")
        ])
    
    def consolidate(self, state: AgentState) -> AgentState:
        """Consolida respuestas de múltiples agentes"""
        # Si solo hay un agente, devolver su respuesta directamente
        if len(state["agent_responses"]) == 1:
            state["final_response"] = list(state["agent_responses"].values())[0]
            return state
        
        # Preparar texto de respuestas
        responses_text = "\n\n---\n\n".join(
            [f"RESPUESTA DE {agent.upper()}:\n{resp}" 
             for agent, resp in state["agent_responses"].items()]
        )
        
        # Crear y ejecutar cadena de consolidación
        chain = self.consolidation_prompt | self.llm | StrOutputParser()
        
        try:
            consolidated = chain.invoke({
                "question": state["original_question"],
                "responses_text": responses_text
            })
            state["final_response"] = consolidated
        except Exception as e:
            state["final_response"] = f"Error al consolidar respuestas: {str(e)}"
        
        return state
    
# ============================================================================
# 6. FUNCIONES PARA NODOS DEL GRAFO
# ============================================================================

def create_agent_node_function(agent_name: str, agent):
    """Crea una función de nodo para un agente específico"""
    def agent_node(state: AgentState) -> AgentState:
        if agent_name not in state.get("target_agents", []):
            return state
        
        try:
            # Obtener respuesta del agente
            response = agent.respond(state["question"])
            
            # Actualizar estado
            if "agent_responses" not in state:
                state["agent_responses"] = {}
            state["agent_responses"][agent_name] = response
            state["current_step"] = "agent_responded"
            
            # SI ES EL ÚNICO AGENTE, ESTABLECER TAMBIÉN EN final_response
            if state.get("agent_count", 0) == 1:
                state["final_response"] = response
            
        except Exception as e:
            error_msg = f"Error en agente {agent_name}: {str(e)}"
            state["agent_responses"][agent_name] = error_msg
            if state.get("agent_count", 0) == 1:
                state["final_response"] = error_msg
        
        return state
    
    return agent_node

def multiple_agents_node(state: AgentState, agents: Dict[str, CVSpecialistAgent]) -> AgentState:
    """Procesa múltiples agentes"""
    target_agents = state.get("target_agents", [])
    
    if not target_agents:
        return state
    
    # Procesar cada agente secuencialmente
    if "agent_responses" not in state:
        state["agent_responses"] = {}
    
    for agent_name in target_agents:
        try:
            agent = agents[agent_name]
            response = agent.respond(state["question"])
            state["agent_responses"][agent_name] = response
        except Exception as e:
            state["agent_responses"][agent_name] = f"Error: {str(e)}"
    
    state["current_step"] = "multiple_processed"
    return state

def consolidator_node(state: AgentState, consolidator: ResponseConsolidator) -> AgentState:
    """Nodo del consolidador"""
    return consolidator.consolidate(state)

# ============================================================================
# 7. CONSTRUCCIÓN DEL GRAFO LANGGRAPH
# ============================================================================

def build_cv_agent_graph(agent_names: List[str]):
    """Construye y compila el grafo completo"""
    
    # Inicializar manager y agentes
    vs_manager = VectorStoreManager()
    
    agents = {}
    for agent_name in agent_names:
        agents[agent_name] = CVSpecialistAgent(
            agent_name=agent_name,
            vectorstore_manager=vs_manager,
            all_agents=agent_names
        )
    
    # Crear instancias
    router = CVAgentRouter(available_agents=list(agents.keys()), default_agent="kevin")
    consolidator = ResponseConsolidator()
    
    # ========== DEFINIR NODOS ==========
    
    # Nodo router
    def router_node(state: AgentState) -> AgentState:
        return router.route(state)
    
    # Nodo para múltiples agentes (con agents inyectado)
    def multiple_agents_node_wrapper(state: AgentState) -> AgentState:
        return multiple_agents_node(state, agents)
    
    # Nodo consolidador (con consolidator inyectado)
    def consolidator_node_wrapper(state: AgentState) -> AgentState:
        return consolidator_node(state, consolidator)
    
    # ========== CONSTRUIR GRAFO ==========
    
    workflow = StateGraph(AgentState)
    
    # Agregar nodos
    workflow.add_node("router", router_node)
    workflow.add_node("process_multiple", multiple_agents_node_wrapper)
    workflow.add_node("consolidator", consolidator_node_wrapper)
    
    # Agregar nodos para cada agente individual
    for agent_name, agent in agents.items():
        workflow.add_node(f"agent_{agent_name}", create_agent_node_function(agent_name, agent))
    
    # ========== LÓGICA DE ENRUTAMIENTO ==========
    
    def decide_next_step(state: AgentState) -> str:
        query_type = state.get("query_type", "default")
        
        if query_type == "single":
            agent_name = state["target_agents"][0]
            return f"agent_{agent_name}"
        elif query_type == "multiple":
            return "process_multiple"
        else:  # default
            return "agent_kevin"
    
    # Conectar router a los nodos
    workflow.add_conditional_edges(
        "router",
        decide_next_step,
        {
            **{f"agent_{name}": f"agent_{name}" for name in agent_names},
            "process_multiple": "process_multiple"
        }
    )
    
    # Conectar agentes individuales al final
    for agent_name in agents.keys():
        workflow.add_edge(f"agent_{agent_name}", END)
    
    # Conectar proceso múltiple al consolidador y luego al final
    workflow.add_edge("process_multiple", "consolidator")
    workflow.add_edge("consolidator", END)
    
    # Punto de entrada
    workflow.set_entry_point("router")
    
    # Compilar grafo
    return workflow.compile()

# ============================================================================
# 8. SISTEMA PRINCIPAL
# ============================================================================

class CVMultiAgentSystem:
    """Sistema principal que orquesta todos los componentes - COMPLETAMENTE"""
    
    def __init__(self, agent_names: List[str]):
        self.agent_names = agent_names
        self.graph = build_cv_agent_graph(agent_names)
        print("✅ Sistema Multi-Agente inicializado correctamente")
        print(f"   Agentes disponibles: {', '.join(agent_names)} (default: kevin)")
        print("   Vectorstore: cv-index en Pinecone")
        print("   LLM: Groq (Llama 3.1 8B Instant)")
    
    def ask(self, question: str) -> str:
        """Método principal para hacer preguntas al sistema"""
        
        # Crear estado inicial
        initial_state = {
            "question": question,
            "original_question": question,
            "target_agents": [],
            "agent_count": 0,
            "query_type": "",
            "agent_responses": {},
            "final_response": None,
            "current_step": "start",
            "conversation_id": str(uuid.uuid4())
        }
        
        try:
            # Ejecutar el grafo
            result = self.graph.invoke(initial_state)
            
            # Obtener respuesta final
            final_response = result.get("final_response")
            
            # Si no hay final_response pero sí hay respuestas de agentes
            if not final_response and result.get("agent_responses"):
                # Si solo hay un agente, usar su respuesta
                if len(result["agent_responses"]) == 1:
                    final_response = list(result["agent_responses"].values())[0]
                    result["final_response"] = final_response
                # Si hay múltiples, crear una consolidación simple
                elif len(result["agent_responses"]) > 1:
                    responses_text = "\n\n".join(
                        [f"**{agent.capitalize()}**:\n{resp}" 
                         for agent, resp in result["agent_responses"].items()]
                    )
                    final_response = f"Respuestas de {len(result['agent_responses'])} personas:\n\n{responses_text}"
                    result["final_response"] = final_response
            
            return result.get("final_response", "No se pudo generar respuesta")
            
        except Exception as e:
            error_msg = f"Error en el sistema: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg