import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from openai import AzureOpenAI
from datetime import datetime

# -------------------- CONFIG --------------------
AZURE_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_API_KEY = os.environ["AZURE_OPENAI_KEY"]
API_VERSION = os.environ["OPENAI_API_VERSION"]

# Azure deployment names
EMBEDDING_DEPLOYMENT = "text-embedding-3-small"
CHAT_DEPLOYMENT = "RPA-Test-Nano"

# -------------------- CLIENT --------------------
client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version=API_VERSION,
)

# -------------------- HELPERS --------------------
def get_embedding(text: str):
    resp = client.embeddings.create(
        model=EMBEDDING_DEPLOYMENT,
        input=text
    )
    return resp.data[0].embedding

def prepare_documents(df: pd.DataFrame):
    docs = []
    for _, row in df.iterrows():
        parts = []
        for col in df.columns:
            parts.append(f"{col}: {row[col]}")
        docs.append(" | ".join(parts))
    return docs

def build_nn_index(documents):
    embeddings = [get_embedding(doc) for doc in documents]
    X = np.array(embeddings)
    nn = NearestNeighbors(n_neighbors=5, metric="cosine")
    nn.fit(X)
    return nn, X, documents

def query_rag(question, nn, X, documents, top_k=5):
    q_emb = np.array(get_embedding(question)).reshape(1, -1)
    distances, indices = nn.kneighbors(q_emb, n_neighbors=top_k)
    
    retrieved_docs = [documents[i] for i in indices[0]]
    context = "\n".join(retrieved_docs)
    
    resp = client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Always answer using only the data provided. Pay close attention to all columns (ProcessName, Owner, Step, Tool, etc."},
            {"role": "user", "content": f"Data:\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0
    )
    return resp.choices[0].message.content

# -------------------- Streamlit UI --------------------
st.set_page_config(
    page_title="Vigilant Analytics Platform",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Corporate CSS - Vigilant Style
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Reset and Base Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Main app background - Clean white */
    .stApp {
        background: #ffffff;
        background-image: 
            radial-gradient(circle at 20% 50%, rgba(120, 119, 198, 0.05) 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, rgba(255, 119, 168, 0.05) 0%, transparent 50%),
            radial-gradient(circle at 40% 20%, rgba(255, 200, 124, 0.05) 0%, transparent 50%);
    }
    
    /* Sidebar - Clean professional style */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
        border-right: 1px solid #e0e0e0;
        box-shadow: 2px 0 10px rgba(0, 0, 0, 0.05);
    }
    
    section[data-testid="stSidebar"] .css-1d391kg {
        padding-top: 2rem;
    }
    
    /* Professional Header */
    .corporate-header {
        background: linear-gradient(90deg, #ff7c00, #ffa733);
        padding: 2rem 3rem;
        border-radius: 0;
        margin: -3rem -3rem 2rem -3rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        position: relative;
        overflow: hidden;
    }
    
    .corporate-header::before {
        content: '';
        position: absolute;
        top: 0;
        right: -100px;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.1) 0%, transparent 70%);
        animation: float 20s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-20px) rotate(180deg); }
    }
    
    .header-content {
        position: relative;
        z-index: 1;
    }
    
    .company-logo {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    
    .logo-text {
        font-family: 'Poppins', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: white;
        letter-spacing: -0.5px;
    }
    
    .platform-badge {
        background: rgba(255, 255, 255, 0.15);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-left: 1rem;
        backdrop-filter: blur(10px);
    }
    
    .header-subtitle {
        color: rgba(255, 255, 255, 0.9);
        font-size: 0.95rem;
        font-weight: 400;
        margin-top: 0.5rem;
    }
    
    /* Live Status Indicator */
    .status-bar {
        display: flex;
        align-items: center;
        gap: 2rem;
        margin-top: 1.5rem;
        padding-top: 1.5rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .status-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.85rem;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        animation: pulse 2s infinite;
        box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.2);
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    
    /* Chat Container - Clean Card Style 
    .chat-wrapper {
        background: white;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        border: 1px solid #e5e7eb;
        padding: 2rem;
        margin: 2rem 0;
        min-height: 400px;
        max-height: 500px;
        overflow-y: auto;
    }*/
    
    /* Messages - Modern Corporate Style */
    .user-message {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        padding: 0.875rem 1.25rem;
        border-radius: 18px 18px 4px 18px;
        margin: 1rem 0 1rem auto;
        max-width: 70%;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.25);
        font-size: 0.95rem;
        line-height: 1.6;
        animation: slideInRight 0.3s ease;
    }
    
    .bot-message {
        background: #f3f4f6;
        color: #1f2937;
        padding: 0.875rem 1.25rem;
        border-radius: 18px 18px 18px 4px;
        margin: 1rem auto 1rem 0;
        max-width: 70%;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        border: 1px solid #e5e7eb;
        font-size: 0.95rem;
        line-height: 1.6;
        animation: slideInLeft 0.3s ease;
    }
    
    .welcome-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border: 2px solid #e5e7eb;
        padding: 3rem;
        border-radius: 16px;
        text-align: center;
        margin: 2rem auto;
        max-width: 600px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    
    .welcome-title {
        color: #1f2937;
        font-size: 1.75rem;
        font-weight: 600;
        margin-bottom: 1rem;
        font-family: 'Poppins', sans-serif;
    }
    
    .welcome-text {
        color: #6b7280;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    /* Animations */
    @keyframes slideInRight {
        from { transform: translateX(20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideInLeft {
        from { transform: translateX(-20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    /* Input Area - Professional Style */
    .stTextInput > div > div > input {
        background: white;
        border: 2px solid #e5e7eb;
        border-radius: 10px;
        padding: 0.875rem 1.25rem;
        font-size: 0.95rem;
        transition: all 0.2s ease;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        outline: none;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #9ca3af;
    }
    
    /* Buttons - Corporate Style */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.625rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(59, 130, 246, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    /* Metrics Cards - Professional Dashboard Style */
    [data-testid="metric-container"] {
        background: white;
        padding: 1.25rem;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
    }
    
    [data-testid="metric-container"]:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
    }
    
    [data-testid="metric-container"] [data-testid="metric-label"] {
        color: #ff8c00;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #ff8c00;
        font-size: 1.75rem;
        font-weight: 700;
        margin-top: 0.25rem;
    }
    
    /* Sidebar Components */
    .sidebar-section {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    .sidebar-title {
        color: #ff8c00;
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #f3f4f6;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: #ff8c00 !important;
    }
    
    /* Success Alert */
    .stSuccess {
        background: #f0fdf4;
        color: #15803d;
        border: 1px solid #bbf7d0;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Error Alert */
    .stError {
        background: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Divider */
    hr {
        border: none;
        border-top: 1px solid #e5e7eb;
        margin: 1.5rem 0;
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f9fafb;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #d1d5db;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #9ca3af;
    }
    
    /* Loading State */
    .loading-skeleton {
        background: linear-gradient(90deg, #f3f4f6 0%, #e5e7eb 50%, #f3f4f6 100%);
        background-size: 200% 100%;
        animation: loading 1.5s ease-in-out infinite;
    }
    
    @keyframes loading {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- Sidebar --------------------
with st.sidebar:
    # Sidebar Header
    st.markdown("""
        <div class="sidebar-title">
            🎛️ Control Center
        </div>
    """, unsafe_allow_html=True)
    
    # Metrics Dashboard
    col1, col2 = st.columns(2)
    with col1:
        queries = len(st.session_state.get('history', [])) // 2
        st.metric(
            label="Queries",
            value=queries,
            delta=f"+{queries}" if queries > 0 else None
        )
    
    with col2:
        st.metric(
            label="Session",
            value=datetime.now().strftime("%I:%M %p")
        )
    
    st.divider()
    
    # Session Stats
    st.markdown("""
        <div style='background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 1rem 0;'>
            <h4 style='color: #1f2937; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem;'>
                📊 Session Statistics
            </h4>
            <div style='color: #6b7280; font-size: 0.85rem;'>
                <div style='display: flex; justify-content: space-between; margin: 0.5rem 0;'>
                    <span>Response Time:</span>
                    <span style='font-weight: 600;'>~1.2s</span>
                </div>
                <div style='display: flex; justify-content: space-between; margin: 0.5rem 0;'>
                    <span>Model:</span>
                    <span style='font-weight: 600;'>Azure GPT</span>
                </div>
                <div style='display: flex; justify-content: space-between; margin: 0.5rem 0;'>
                    <span>Data Points:</span>
                    <span style='font-weight: 600;'>Loading...</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Actions
    if st.button("🔄 New Session", use_container_width=True, type="secondary"):
        st.session_state.history = []
        st.rerun()
    
    # Footer
    st.markdown("""
        <div style='margin-top: 2rem; padding: 1rem; background: #f9fafb; border-radius: 8px; border: 1px solid #e5e7eb;'>
            <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;'>
                <span style='font-size: 1.5rem;'>🏢</span>
                <span style='font-weight: 600; color: #1f2937;'>Vigilant Technologies</span>
            </div>
            <p style='color: #6b7280; font-size: 0.75rem; margin: 0;'>
                Enterprise AI Analytics Platform<br>
                © 2025 All Rights Reserved
            </p>
        </div>
    """, unsafe_allow_html=True)

# -------------------- Main Content --------------------
# Corporate Header
st.markdown("""
    <div class="corporate-header">
        <div class="header-content">
            <div class="company-logo">
                <span class="logo-text">Vigilant Analytics</span>
                <span class="platform-badge">Enterprise</span>
            </div>
            <div class="header-subtitle">
                Intelligent Process Automation & Data Analytics Platform
            </div>
            <div class="status-bar">
                <div class="status-item">
                    <span class="status-dot"></span>
                    <span>System Operational</span>
                </div>
                <div class="status-item">
                    <span style="color: rgba(255,255,255,0.6);">|</span>
                    <span>Chick-fil-A Process Dataset</span>
                </div>
                <div class="status-item">
                    <span style="color: rgba(255,255,255,0.6);">|</span>
                    <span>Azure OpenAI Connected</span>
                </div>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# -------------------- Load Excel --------------------
EXCEL_FILE = "process info ai.xlsx"

@st.cache_resource(show_spinner=False)
def load_and_index_data():
    try:
        df = pd.read_excel(EXCEL_FILE)
        documents = prepare_documents(df)
        nn, X, docs = build_nn_index(documents)
        return nn, X, docs, True, len(df)
    except Exception as e:
        return None, None, None, False, 0

# Load data
with st.spinner("⚙️ Initializing AI models and indexing data..."):
    nn, X, docs, data_loaded, num_records = load_and_index_data()

if data_loaded:
    # Professional success notification
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.success(f"✅ System initialized successfully • {num_records} records indexed")

# -------------------- Initialize Session --------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------- Chat Interface --------------------
# Chat Container
if len(st.session_state.history) == 0:
    # Welcome Screen
    st.markdown("""
        <div class="welcome-card">
            <div class="welcome-title">
                Welcome to Vigilant Analytics
            </div>
            <div class="welcome-text">
                Your AI-powered assistant is ready to analyze Chick-fil-A process data. 
                Ask questions about operations, performance metrics, or any data insights you need.
            </div>
            <div style='margin-top: 2rem; display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap;'>
                <span style='background: #f3f4f6; padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.85rem; color: #6b7280;'>
                    📊 Process Analytics
                </span>
                <span style='background: #f3f4f6; padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.85rem; color: #6b7280;'>
                    🎯 Performance Metrics
                </span>
                <span style='background: #f3f4f6; padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.85rem; color: #6b7280;'>
                    📈 Data Insights
                </span>
            </div>
        </div>
    """, unsafe_allow_html=True)
else:
    # Chat Messages
    st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
    messages_html = ""
    for role, message in st.session_state.history:
        if role == "user":
            messages_html += f'<div class="user-message">{message}</div>'
        else:
            messages_html += f'<div class="bot-message">{message}</div>'
    st.markdown(messages_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- Input Section --------------------
st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

# Professional Input Form
with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([11, 1])
    
    with col1:
        user_input = st.text_input(
            "Message",
            placeholder="Ask about process data, metrics, or insights...",
            label_visibility="collapsed"
        )
    
    with col2:
        submit = st.form_submit_button("→", use_container_width=True)

# Process Query
if submit and user_input:
    if data_loaded:
        st.session_state.history.append(("user", user_input))
        
        with st.spinner("Processing your query..."):
            try:
                response = query_rag(user_input, nn, X, docs)
                st.session_state.history.append(("bot", response))
            except Exception as e:
                st.session_state.history.append(("bot", f"I apologize, but I encountered an error: {str(e)}"))
        
        st.rerun()
    else:
        st.error("⚠️ Data not loaded. Please check the Excel file path.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <div style='color: #9ca3af; font-size: 0.85rem;'>
            Powered by Saiganesh Challa
        </div>
    </div>

""", unsafe_allow_html=True)
