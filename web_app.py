import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory as ChatMessageHistory

st.set_page_config(page_title="Enterprise AI Assistant", page_icon="🤖", layout="centered")
st.title("🤖 Advanced Project Assistant")
st.caption("Powered by Local Ollama (Llama 3) & ChromaDB")

# =====================================================================
# INITIALIZE BACKEND COMPONENTS (Cached so they run only once)
# =====================================================================
@st.cache_resource
def initialize_rag():
    knowledge_base_text = """
    Project Genesis is our company's secret internal AI initiative.
    The budget for Project Genesis is $5,000,000 for the fiscal year 2026.
    The lead researcher running the project is Dr. Sarah Jenkins. 
    The strict deadline for the prototype delivery is October 14, 2026.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
    chunks = text_splitter.split_text(knowledge_base_text)
    
    embeddings_model = OllamaEmbeddings(model="nomic-embed-text")
    vector_database = Chroma.from_texts(chunks, embeddings_model)
    return vector_database.as_retriever(search_kwargs={"k": 2})

retriever_tool = initialize_rag()

# Manage persistent memory across browser refreshes using Streamlit Session State
if "memory_storage" not in st.session_state:
    st.session_state.memory_storage = ChatMessageHistory()
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Ask me anything about Project Genesis."}]

# =====================================================================
# CONSTRUCT PIPELINE
# =====================================================================
llm_brain = ChatOllama(model="llama3", temperature=0.3)

system_instruction_string = """You are an advanced company assistant. 
Use the following pieces of retrieved context to answer the user's question. 
Retrieved Context:
{context}
Answer accurately based on the context and conversation history."""

prompt_blueprint = ChatPromptTemplate.from_messages([
    ("system", system_instruction_string),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def execution_chain(inputs):
    retrieved_docs = retriever_tool.invoke(inputs["input"])
    formatted_context = format_docs(retrieved_docs)
    return prompt_blueprint.format_prompt(
        context=formatted_context,
        chat_history=inputs.get("chat_history", []),
        input=inputs["input"]
    ).to_messages()

final_bot_chain = RunnableWithMessageHistory(
    execution_chain | llm_brain,
    lambda session_id: st.session_state.memory_storage,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# =====================================================================
# UI RENDERING
# =====================================================================
# Display prior chat logs on screen repaint
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Listen for new user typing input
if user_query := st.chat_input("Type your message here..."):
    # Display human bubble
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate response via our LangChain RAG pipeline
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            bot_reply = final_bot_chain.invoke(
                {"input": user_query},
                config={"configurable": {"session_id": "web_session"}}
            )
            response_text = bot_reply.content
            st.markdown(response_text)
            
    st.session_state.messages.append({"role": "assistant", "content": response_text})