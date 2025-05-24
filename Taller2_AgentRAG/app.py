import os
import streamlit as st
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings 
from langchain_chroma import Chroma

from langchain import hub
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader

from langgraph.graph import START, END
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ['USER_AGENT'] = os.getenv("USER_AGENT")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_processed" not in st.session_state:
    st.session_state.document_processed = False

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0.5,
    top_p=0.9,
    top_k=20,
    num_predict=512,
    max_new_tokens=9000,
)
prompt = hub.pull("rlm/rag-prompt")

graph_builder = StateGraph(MessagesState)

 # Inicializar vector_store y embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
current_dir = os.path.dirname(os.path.abspath(__file__))
chroma_dir = os.path.join(current_dir, "AtomAI_ChrLang_db")
os.makedirs(chroma_dir, exist_ok=True)
vector_store = Chroma(
                    collection_name="AtomAI_collection",
                    embedding_function=embeddings,
                    persist_directory=chroma_dir,
                )

st.set_page_config(
    page_title="Atom.AI-RAG",
    page_icon=r"C:\Users\danri\Documents\Esp_IA\2doSemestre\TecnicasIA_LengModels\Taller2_AgentRAG\img\AtomAI_Icon.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📄Atom.AI-RAG")
st.markdown(
    "Carga un PDF en la barra lateral, procesa el documento y luego realiza preguntas "
    "basadas en su contenido."
)

with st.sidebar:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(current_dir, "img", "AtomAI_Logo.png")
    st.image(logo_path, use_container_width=True)
    st.header("📄 Cargar documento")
    uploaded_file = st.file_uploader("Elige un PDF", type=["pdf"])
    if uploaded_file and not st.session_state.get("document_processed", False):
        try:
            with st.spinner("⏳ Procesando documento..."):
                data_dir = os.path.join(current_dir, "data")
                os.makedirs(data_dir, exist_ok=True)
                data_path = os.path.join(data_dir, uploaded_file.name)
                with open(data_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Cargar PDF y dividir en fragmentos
                loader = PyPDFLoader(data_path)
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", " ", ""],
                )
                all_splits = text_splitter.split_documents(docs)

                # Añadir documentos
                vector_store.add_documents(documents=all_splits)
                st.session_state.document_processed = True
                st.success("✅ Documento cargado y procesado.")
        except Exception as e:
            st.error(f"❌ Error al procesar el documento: {str(e)}")
    elif st.session_state.get("document_processed", False):
        st.info("📄 Documento ya procesado anteriormente.")

    st.divider()
    st.subheader("⚙️ Controles")
    if st.button("🗑️ Limpiar conversación", use_container_width=True):
        st.session_state.messages = []
        st.success("Historial borrado.")
    st.write("") 

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join( 
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tools = ToolNode([retrieve])

def generate(state: MessagesState):
    """Generate answer."""
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use four sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}

graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)
graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)
graph = graph_builder.compile()

# Guardar el grafo
img_bytes = graph.get_graph().draw_mermaid_png()
graph_path = os.path.join(current_dir, "data", "langgraph_visualization.png")
os.makedirs(os.path.dirname(graph_path), exist_ok=True)
with open(graph_path, "wb") as f:
    f.write(img_bytes)

left_col, right_col = st.columns([3, 1], vertical_alignment="top", border=False)
with left_col:
    # Container para los mensajes
    messages = st.container(height=300)

    # Mostrar historial del chat
    for msg in st.session_state.messages:
        with messages.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input del usuario
    user_input = st.chat_input("💬 Preguntame sobre tu documento...")

    if user_input and st.session_state.document_processed:
        # Agregar mensaje del usuario al historial
        st.session_state.messages.append({"role": "user", "content": user_input})
        with messages.chat_message("user"):
            st.markdown(user_input)

        # Procesar la respuesta
        response = graph.invoke({"messages": [{"type": "human", "content": user_input}]})
        messages_response = response["messages"]
        final_message = messages_response[-1]
        if hasattr(final_message, "content"):
            assistant_response = final_message.content
        else:
            assistant_response = final_message["content"]

        # Mostrar respuesta del asistente
        with messages.chat_message("assistant"):
            st.markdown(assistant_response)

        # Agregar respuesta del asistente al historial
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    elif not st.session_state.document_processed:
        st.info("⬆️ Sube un archivo PDF para comenzar.")

with right_col:
    st.subheader("🔗 Grafo de estados")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    graph_path = os.path.join(current_dir, "data", "langgraph_visualization.png")
    if os.path.exists(graph_path):
        st.image(graph_path, use_container_width=True)
        st.caption("Diagrama generado automáticamente por LangGraph.")
    else:
        st.info("El grafo se generará tras la primera ejecución.")