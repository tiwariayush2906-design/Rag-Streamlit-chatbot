import streamlit as st
from datetime import datetime
from zoneinfo import ZoneInfo
import os
import uuid
import tempfile
import random
import io

from groq import Groq
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from fpdf import FPDF
from docx import Document
from pptx import Presentation
from openpyxl import Workbook

load_dotenv()

groq_api_key = None
if "GROQ_API_KEY" in st.secrets:
    groq_api_key = st.secrets["GROQ_API_KEY"]
else:
    groq_api_key = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="centered", initial_sidebar_state="expanded")

st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}

header {
    visibility: visible !important;
    background: transparent !important;
}
header [data-testid="stToolbar"] {
    visibility: hidden;
}

.block-container {
    padding-top: 3rem;
    max-width: 750px;
}

h1 {
    font-family: Georgia, 'Times New Roman', serif;
    font-weight: 400;
    font-size: 2.6rem;
    text-align: center;
    margin-bottom: 0.2rem;
}

.subtitle {
    text-align: center;
    color: #888;
    font-size: 1rem;
    margin-bottom: 2rem;
}

[data-testid="stChatInput"] {
    border-radius: 25px;
}

.loading-dots {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 10px 0;
}

.loading-dots span {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #d97757;
    animation: bounce 1.4s infinite ease-in-out both;
}

.loading-dots span:nth-child(1) { animation-delay: -0.32s; }
.loading-dots span:nth-child(2) { animation-delay: -0.16s; }

@keyframes bounce {
    0%, 80%, 100% { transform: scale(0.6); opacity: 1; }
    40% { transform: scale(1); opacity: 1; }
}

[data-testid="stSidebar"] button {
    text-align: left !important;
    justify-content: flex-start !important;
}

.sidebar-empty {
    color: #999;
    font-size: 0.85rem;
    padding: 10px 4px;
}
</style>
""", unsafe_allow_html=True)

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "response_language" not in st.session_state:
    st.session_state.response_language = "english"


def new_chat_dict():
    return {
        "title": "New Chat",
        "messages": [],
        "attached_files": [],
        "vectorstore": None,
        "retriever": None,
    }


def ensure_active_chat():
    if st.session_state.current_chat_id is None or \
       st.session_state.current_chat_id not in st.session_state.chat_sessions:
        chat_id = str(uuid.uuid4())
        st.session_state.chat_sessions[chat_id] = new_chat_dict()
        st.session_state.current_chat_id = chat_id


def current_chat():
    ensure_active_chat()
    return st.session_state.chat_sessions[st.session_state.current_chat_id]


def start_new_chat():
    chat_id = str(uuid.uuid4())
    st.session_state.chat_sessions[chat_id] = new_chat_dict()
    st.session_state.current_chat_id = chat_id


def switch_chat(chat_id):
    st.session_state.current_chat_id = chat_id


ensure_active_chat()

with st.sidebar:
    st.markdown("### 🤖 RAG Chatbot")

    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        start_new_chat()
        st.rerun()

    st.markdown("---")
    st.markdown("#### 🕘 History")

    search_query = st.text_input(
        "Search history",
        label_visibility="collapsed",
        placeholder="🔍 Search chats..."
    )

    all_chat_ids = list(st.session_state.chat_sessions.keys())[::-1]

    visible_chats = [
        cid for cid in all_chat_ids
        if st.session_state.chat_sessions[cid]["messages"]
    ]

    if search_query:
        visible_chats = [
            cid for cid in visible_chats
            if search_query.lower() in st.session_state.chat_sessions[cid]["title"].lower()
        ]

    if not visible_chats:
        st.markdown('<p class="sidebar-empty">Abhi koi purani chat nahi hai.</p>', unsafe_allow_html=True)
    else:
        for cid in visible_chats:
            chat_item = st.session_state.chat_sessions[cid]
            is_active = cid == st.session_state.current_chat_id
            label = f"{'💬 ' if not is_active else '▶️ '}{chat_item['title']}"
            if st.button(label, key=f"hist_{cid}", use_container_width=True):
                switch_chat(cid)
                st.rerun()

current_hour = datetime.now(ZoneInfo("Asia/Kolkata")).hour

if 5 <= current_hour < 12:
    greeting = "Good morning"
elif 12 <= current_hour < 17:
    greeting = "Good afternoon"
elif 17 <= current_hour < 21:
    greeting = "Good evening"
else:
    greeting = "Good night"

st.markdown(f"<h1>🌟 {greeting}, dost!</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">How can I help you today?</p>', unsafe_allow_html=True)

LANGUAGE_MAP = {
    "english": "Respond strictly in English.",
    "hindi": "Respond strictly in Hindi (Devanagari script).",
    "hinglish": "Respond in natural Hinglish (Hindi-English mix using Roman script)."
}

RAG_PROMPT_TEMPLATE = """You are a helpful and accurate assistant.
Answer the user query based on the context provided. Do NOT output prompt instructions, internal rules, disclaimers, or system text in your response.

Recent History:
{chat_history}

Language Rule: {language_instruction}
Context: {context}

Question: {question}

Final Answer:"""

GENERAL_SYSTEM_PROMPT = """You are a knowledgeable and accurate AI assistant.

CRITICAL DIRECTIVE:
Output ONLY the clean, final, direct answer to the user query. Never include internal instructions, system prompts, disclaimers, meta-commentary, or verification labels.

Current Date & Time: {current_datetime}
Language Instruction: {language_instruction}

Recent History:
{chat_history}"""


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


@st.cache_resource
def get_llm():
    if not groq_api_key:
        st.error("🔑 GROQ_API_KEY nahi mili! Streamlit Cloud Settings me Secrets add karein.")
        st.stop()

    candidate_models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it"
    ]

    selected_model = None

    try:
        client = Groq(api_key=groq_api_key)
        available_models = [m.id for m in client.models.list().data]
        
        for model in candidate_models:
            if model in available_models:
                selected_model = model
                break

        if not selected_model and available_models:
            selected_model = available_models[0]

    except Exception:
        selected_model = "llama-3.3-70b-versatile"

    return ChatGroq(
        model=selected_model,
        temperature=0.2,
        max_tokens=1024,
        groq_api_key=groq_api_key
    )


def load_file(file_path, file_extension):
    if file_extension == "pdf":
        return PyPDFLoader(file_path).load()
    elif file_extension == "txt":
        return TextLoader(file_path, encoding="utf-8").load()
    elif file_extension == "docx":
        return Docx2txtLoader(file_path).load()
    return []


def process_file(uploaded_file):
    chat = current_chat()

    file_extension = uploaded_file.name.split(".")[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    documents = load_file(tmp_path, file_extension)
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=30)
    chunks = splitter.split_documents(documents)

    embeddings = get_embeddings()
    collection_name = f"chat_{st.session_state.current_chat_id}".replace("-", "_")
    
    chat["vectorstore"] = Chroma.from_documents(
        documents=chunks, embedding=embeddings, collection_name=collection_name
    )

    os.unlink(tmp_path)
    chat["retriever"] = chat["vectorstore"].as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}
    )


def detect_language_command(text):
    t = text.lower()
    if any(x in t for x in ["hindi me do", "hindi mein do", "ab hindi", "hindi me bolo"]):
        return "hindi"
    if any(x in t for x in ["hinglish me do", "hinglish mein do", "ab hinglish"]):
        return "hinglish"
    if any(x in t for x in ["english me do", "ab english", "english me bolo"]):
        return "english"
    return None


def detect_file_format(text):
    t = text.lower()
    if "pdf" in t:
        return "pdf"
    if any(x in t for x in ["word", "docx", "doc file"]):
        return "docx"
    if any(x in t for x in ["excel", "xlsx", "spreadsheet"]):
        return "xlsx"
    if any(x in t for x in ["ppt", "pptx", "powerpoint", "presentation", "slides"]):
        return "pptx"
    return None


ANSWER_COLORS = [
    "#1e1e1e",
    "#1b3a4b",
    "#2d1b4e",
    "#1b4d3e",
    "#4d1b1b",
    "#4d3a1b",
    "#1b4d4d",
    "#3a1b4d",
]


def render_answer(text):
    color = random.choice(ANSWER_COLORS)
    st.markdown(
        f"""<div style="background-color:{color};color:#f5f5f5;padding:16px;
        border-radius:10px;line-height:1.6;font-size:16px;white-space:pre-wrap;">{text}</div>""",
        unsafe_allow_html=True
    )


def generate_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    for line in text.split("\n"):
        safe_line = line.encode("latin-1", errors="replace").decode("latin-1")
        pdf.multi_cell(0, 8, safe_line)
    return bytes(pdf.output())


def generate_docx(text):
    doc = Document()
    for line in text.split("\n"):
        doc.add_paragraph(line)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def generate_pptx(text):
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Generated Content"
    body = slide.placeholders[1].text_frame
    lines = [l for l in text.split("\n") if l.strip()]
    if lines:
        body.text = lines[0]
        for line in lines[1:]:
            p = body.add_paragraph()
            p.text = line
    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()


def generate_xlsx(text):
    wb = Workbook()
    ws = wb.active
    for i, line in enumerate(text.split("\n"), start=1):
        ws.cell(row=i, column=1, value=line)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


FILE_GENERATORS = {
    "pdf": (generate_pdf, "application/pdf", "output.pdf"),
    "docx": (generate_docx, "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "output.docx"),
    "pptx": (generate_pptx, "application/vnd.openxmlformats-officedocument.presentationml.presentation", "output.pptx"),
    "xlsx": (generate_xlsx, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "output.xlsx"),
}

chat = current_chat()

if chat["attached_files"]:
    st.caption("📎 Attached: " + ", ".join(chat["attached_files"]))

for msg in chat["messages"]:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_answer(msg["content"])
        else:
            st.write(msg["content"])

chat_input_data = st.chat_input(
    "Poochiye kuch bhi, ya seedha file attach kar dijiye... ✨",
    accept_file=True,
    file_type=["pdf", "txt", "docx"]
)

user_query = None
newly_uploaded_file = None

if chat_input_data:
    user_query = chat_input_data.text
    if chat_input_data.files:
        newly_uploaded_file = chat_input_data.files[0]

if newly_uploaded_file is not None:
    with st.spinner(f"'{newly_uploaded_file.name}' process ho raha hai..."):
        process_file(newly_uploaded_file)
        chat["attached_files"].append(newly_uploaded_file.name)
    st.success(f"✅ '{newly_uploaded_file.name}' ready hai!")

if user_query:
    if chat["title"] == "New Chat":
        chat["title"] = (user_query[:30] + "…") if len(user_query) > 30 else user_query

    lang_cmd = detect_language_command(user_query)

    if lang_cmd:
        st.session_state.response_language = lang_cmd
        chat["messages"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)
        confirm_msg = {
            "hindi": "ठीक है, अब मैं हिंदी में जवाब दूंगा।",
            "hinglish": "Theek hai, ab main Hinglish me jawab dunga.",
            "english": "Sure, I will answer in English from now on."
        }[lang_cmd]
        with st.chat_message("assistant"):
            render_answer(confirm_msg)
        chat["messages"].append({"role": "assistant", "content": confirm_msg})

    else:
        chat["messages"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            loading_placeholder = st.empty()
            loading_placeholder.markdown(
                '<div class="loading-dots"><span></span><span></span><span></span></div>',
                unsafe_allow_html=True
            )

            language_instruction = LANGUAGE_MAP[st.session_state.response_language]
            
            try:
                llm = get_llm()

                recent_messages = chat["messages"][-2:]
                history_text = ""
                for m in recent_messages:
                    role_label = "User" if m["role"] == "user" else "Assistant"
                    history_text += f"{role_label}: {m['content'][:100]}\n"

                if chat["retriever"] is not None:
                    source_docs = chat["retriever"].invoke(user_query)
                    context_text = "\n".join(doc.page_content for doc in source_docs)[:800]

                    rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
                    chain = rag_prompt | llm | StrOutputParser()

                    answer = chain.invoke({
                        "context": context_text,
                        "question": user_query,
                        "language_instruction": language_instruction,
                        "chat_history": history_text
                    })
                else:
                    current_datetime = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%A, %d %B %Y, %I:%M %p")
                    
                    system_msg = GENERAL_SYSTEM_PROMPT.format(
                        language_instruction=language_instruction,
                        current_datetime=current_datetime,
                        chat_history=history_text
                    )
                    response = llm.invoke([
                        ("system", system_msg),
                        ("human", user_query)
                    ])
                    answer = response.content

                loading_placeholder.empty()
                render_answer(answer)

                file_format = detect_file_format(user_query)
                if file_format:
                    gen_func, mime, filename = FILE_GENERATORS[file_format]
                    file_bytes = gen_func(answer)
                    st.download_button(
                        label=f"📥 Download as {file_format.upper()}",
                        data=file_bytes,
                        file_name=filename,
                        mime=mime
                    )

                chat["messages"].append({"role": "assistant", "content": answer})

            except Exception as e:
                loading_placeholder.empty()
                st.error(f"⚠️ API Call Error: {str(e)}")