import streamlit as st
import random
from langchain_groq import ChatGroq
from langchain_community.embeddings import DeterministicFakeEmbedding
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# =====================================================================
# UI CONFIGURATION & HERO BRANDING
# =====================================================================
st.set_page_config(page_title="Center for AI-IoT Innovations", page_icon="🌐", layout="wide")

# =====================================================================
# UI CONFIGURATION & HERO BRANDING
# =====================================================================
# INJECT THIS BLOCK TO REMOVE THE CLOUD PLATFORM FOOTER STRIP COMPLETELY:
st.markdown(
    """
    <style>
    /* Hides the complete bottom footer toolbar wrapper container */
    footer {visibility: hidden; height: 0px;}
    .stAppDeployDropdown {display: none;}
    stViewerActionButton {display: none;}
    
    /* Removes extra blank padding space created by forcing the footer away */
    .stAppViewer {padding-bottom: 0px;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌐 Center for AI-IoT Innovations")

st.caption("Sponsored by Ministry of Electronics and Information Technology (MeitY)")
st.markdown(
    "*An AI-IoT Research Center focused on intelligent sensors, embedded systems, cloud technologies, and real-world smart applications.*"
)
st.markdown("---")

# =====================================================================
# SIDEBAR: TELEMETRY VIEWERS FOR ONGOING SYSTEMS
# =====================================================================

st.sidebar.markdown("---")

# # Air Quality Telemetry Node
# st.sidebar.subheader("🍃 Air Quality Monitoring")
# st.sidebar.metric(label="PM2.5 Index", value=f"{round(random.uniform(12.0, 35.5), 1)} µg/m³", delta="Optimal")
# st.sidebar.metric(label="CO₂ Concentration", value=f"{random.randint(400, 425)} ppm", delta="Normal")

# st.sidebar.markdown("---")

# # Water Systems Telemetry Node
# st.sidebar.subheader("🌊 Water & Environment Tracking")
# st.sidebar.metric(label="Current River Level", value=f"{round(random.uniform(2.1, 4.8), 2)} meters", delta="Stable")
# st.sidebar.metric(label="Turbidity Index", value=f"{round(random.uniform(1.2, 3.8), 2)} NTU", delta="-0.2 NTU", delta_color="inverse")

st.sidebar.markdown("---")
st.sidebar.info("💡 Hub Status: Operational\nLocation: Adi Shankara Innovation Lab")

# =====================================================================
# CORE ENGINE: RAG STORAGE KNOWLEDGE INGESTION
# =====================================================================
@st.cache_resource
def initialize_knowledge_base():
    # Context data perfectly structured around the 4 core pillars
    system_knowledge = """
    =====================================================================
    CENTER OVERVIEW
    =====================================================================
    Facility Name: Center for AI-IoT Innovations
    Official Website: https://aiiot.it.com
    Sponsorship: Funded and sponsored by the Ministry of Electronics and Information Technology (MeitY), Government of India.
    Location: Adi Shankara Institute of Engineering and Technology (ASIET), Kalady, Kerala, India.
    Core Objective: Integration of electronic hardware design with artificial intelligence (AI) and explainable AI (XAI).

    =====================================================================
    SECTION 1: AIR QUALITY MONITORING (PROJECT 1)
    =====================================================================
    Project Name: Air Quality Monitoring
    React Page Link: /product-details/indoor-monitor (for AQMS-Indoor) or /product-details/outdoor-station (for AQMS-Outdoor)
    Description: Developed a smart sensor module capable of real-time monitoring and management of environmental pollutants.
    Tracked Parameters: PM2.5, PM10, CO₂, CO, NH₃, temperature, and humidity.
    Hardware Architecture: The module integrates multiple gas and environmental sensors to continuously measure ambient air quality parameters.
    Research Milestones: Published comprehensive analyses focusing on Recurrent Neural Networks (RNN) model comparison for predictive air pollution forecasting.

    =====================================================================
    SECTION 2: SMART WATER LEVEL MONITORING (PROJECT 2)
    =====================================================================
    Project Name: Smart Water Level Monitoring
    React Page Link: /product-details/flood-alert (Predictive Flood Alert) or /river-forecast[cite: 1, 3]
    Description: A robust and intelligent system designed for real-time water level tracking with an integrated early-warning alert mechanism to mitigate flood risks.
    System Capabilities: Continuously monitors water fluctuations using sensor-based measurements and ensures timely notifications during critical water level rises.
    Cloud & AI Logic: Feeds live telemetry streams into predictive data models to calculate river-level forecasts and alert regional dashboards.

    =====================================================================
    SECTION 3: DIGITAL WATER DISTRIBUTION (PROJECT 3)
    =====================================================================
    Project Name: Digital Water Distribution
    React Page Link: /product-details/distribution-net (Digital Flow Meter) or /demand-forecasting[cite: 1, 3]
    Description: The Digital Water Distribution system creates a virtual replica (digital twin) of water distribution networks.
    System Capabilities: Enables real-time monitoring, leak detection, and predictive maintenance.
    AI Integration: Uses IoT sensors and AI analytics to optimize water flow distribution and significantly reduce water wastage.
    Intellectual Property: Supported by an official Indian Patent filed for an Artificial Intelligence Integrated Water Distribution and Monitoring System.

    =====================================================================
    SECTION 4: STARTUP & SKILL DEVELOPMENT (PROJECT 4)
    =====================================================================
    Project Name: Startup & Skill Development
    React Page Link: /product-details/iot-training (IoT Workshops) or /product-details/pcb-workshop[cite: 3]
    Description: Building a vibrant startup ecosystem while offering hands-on skill development programs centered around our IoT solutions.
    Target Audience: Empowers innovators, engineering students, and tech professionals to adopt emerging technologies and bring ideas to market.
    Educational Offerings: Conducts regular hands-on workshops covering custom Flask frameworks, PCB design workflows, and IoT cloud connectivity solutions.
    """

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
    chunks = text_splitter.split_text(system_knowledge)
    
    embeddings = DeterministicFakeEmbedding(size=768)
    vector_db = Chroma.from_texts(chunks, embeddings)
    return vector_db.as_retriever(search_kwargs={"k": 2})

retriever_tool = initialize_knowledge_base()

# =====================================================================
# BRAIN INITIALIZATION: CONFIGURE CLOUD PROVIDER
# =====================================================================
api_key = st.secrets.get("GROQ_API_KEY", "MISSING_KEY")

if api_key == "MISSING_KEY":
    st.warning("⚠️ Cloud API Key missing. Please check your `.streamlit/secrets.toml` file.")
    st.stop()

llm_brain = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=api_key, temperature=0.2)

if "chat_history_store" not in st.session_state:
    st.session_state.chat_history_store = ChatMessageHistory()
if "ui_chat_logs" not in st.session_state:
    st.session_state.ui_chat_logs = [{"role": "assistant", "content": "Welcome! Ask me anything about our MeitY-sponsored research projects, hardware sensor setups, or dynamic system integration routes."}]

# Build Pipeline Logic Chains
system_instruction = """You are an expert AI-IoT Technical Research Assistant. 
Answer questions accurately based on the established center capabilities and project contexts provided.

Retrieved Research Context:
{context}

Respond professionally, highlighting how our sensor networks, AI integration, and ecosystem can assist the user."""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_instruction),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def execution_pipeline(inputs):
    docs = retriever_tool.invoke(inputs["input"])
    context_str = format_docs(docs)
    return prompt_template.format_prompt(
        context=context_str,
        chat_history=inputs.get("chat_history", []),
        input=inputs["input"]
    ).to_messages()

bot_engine = RunnableWithMessageHistory(
    execution_pipeline | llm_brain,
    lambda session_id: st.session_state.chat_history_store,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# =====================================================================
# FRONTEND WINDOW DISPLAY LOOP
# =====================================================================
for msg in st.session_state.ui_chat_logs:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt_input := st.chat_input("Ask about our ongoing initiatives ..."):
    st.session_state.ui_chat_logs.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)
        
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge indexes..."):
            response = bot_engine.invoke(
                {"input": prompt_input},
                config={"configurable": {"session_id": "cloud_prod_run"}}
            )
            st.markdown(response.content)
            
    st.session_state.ui_chat_logs.append({"role": "assistant", "content": response.content})