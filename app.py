# app.py - FastAPI Webhook Listener (ĐÃ THÊM LOGIC GHI NHỚ HỘI THOẠI)

import os
import uvicorn
import requests
import json
import warnings
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, Response 

# Import các thành phần RAG Core
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Tắt cảnh báo về việc chuyển đổi thư viện (Deprecation Warnings)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Tải biến môi trường ngay lập tức
load_dotenv() 

# --- I. CẤU HÌNH VÀ KHỞI TẠO RAG CORE ---

# Hằng số cấu hình (Phải khớp với các bước trước)
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL = "gemini-2.5-flash" 

# Lấy các biến môi trường cần thiết từ tệp.env
VERIFY_TOKEN = os.getenv("FACEBOOK_VERIFY_TOKEN", "DEFAULT_TOKEN")
PAGE_ACCESS_TOKEN = os.getenv("FACEBOOK_PAGE_ACCESS_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ⚠️ BỘ NHỚ HỘI THOẠI TẠM THỜI (Sẽ bị reset khi ứng dụng khởi động lại)
# Dictionary: {sender_id: [{"role": "User/AI", "content": "message"},...]}
CONVERSATION_HISTORY = {} 

# Prompt Template cho AI (Không đổi)
PROMPT_TEMPLATE = """
You are a friendly and professional customer support assistant for a Nail Salon. 
Your primary goal is to provide accurate and helpful answers based ONLY on the provided context.
If the context does not contain the answer, politely state that the information is not available in the salon's knowledge base.
Always answer in English (US).

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

# Khởi tạo các thành phần RAG toàn cầu
try:
    # 1. Khởi tạo Embeddings và ChromaDB
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    
    # 2. Khởi tạo LLM (Truyền API Key trực tiếp để đảm bảo xác thực)
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL, 
        temperature=0.1,
        api_key=GEMINI_API_KEY
    )
    
    # 3. Lắp ráp Chuỗi RAG
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}

| prompt
| llm
| StrOutputParser()
    )
    print("RAG Core Initialized successfully and ready for Facebook messages.")
except Exception as e:
    print(f"FATAL ERROR: RAG Core failed to initialize: {e}")
    rag_chain = None 


# --- II. LOGIC GIAO TIẾP VỚI FACEBOOK ---

# Địa chỉ API Messenger của Facebook
MESSAGING_URL = "https://graph.facebook.com/v18.0/me/messages"

def send_message(sender_id, text):
    """Gửi tin nhắn văn bản trở lại Facebook Messenger."""
    if not PAGE_ACCESS_TOKEN:
        print("ERROR: PAGE_ACCESS_TOKEN not found. Cannot send message.")
        return
        
    params = {"access_token": PAGE_ACCESS_TOKEN}
    headers = {"Content-Type": "application/json"}
    # Facebook giới hạn tin nhắn 640 ký tự, chúng ta cắt tin nhắn nếu cần
    if len(text) > 640:
        text = text[:637] + "..." 
        
    data = json.dumps({
        "recipient": {"id": sender_id},
        "message": {"text": text}
    })
    
    try:
        response = requests.post(MESSAGING_URL, headers=headers, params=params, data=data)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"ERROR sending message to Facebook: {e}")


def get_rewritten_question(sender_id, current_question):
    """Sử dụng lịch sử hội thoại để viết lại câu hỏi mơ hồ thành câu hỏi độc lập."""
    
    # Lấy lịch sử 4 tin nhắn gần nhất (2 cặp User/AI)
    history = CONVERSATION_HISTORY.get(sender_id,)[-4:]
    
    # Định dạng lịch sử
    chat_history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
    
    # Prompt để Gemini viết lại câu hỏi (Rất đơn giản và tập trung)
    rewrite_prompt = f"""
    You are an expert at rewriting follow-up questions to make them standalone, based on the chat history.
    Rewrite the 'CURRENT QUESTION' using the 'CHAT HISTORY' so that it can be answered without needing the history.
    Example: (User: What is the price of the Pure Love Pedicure? AI: $50. User: tell me more) -> Rewritten: 'tell me more about the Pure Love Pedicure'
    
    CHAT HISTORY:
    {chat_history_str}
    
    CURRENT QUESTION:
    {current_question}
    
    REWRITTEN QUESTION (Only output the rewritten question):
    """
    
    try:
        # Sử dụng LLM để viết lại câu hỏi
        response = llm.invoke(rewrite_prompt)
        rewritten_question = response.content.strip() 
        return rewritten_question
    except Exception as e:
        print(f"ERROR rewriting question with LLM: {e}")
        # Nếu lỗi LLM, giữ nguyên câu hỏi ban đầu
        return current_question


def process_message(sender_id, message_text):
    """Xử lý tin nhắn đến bằng cách sử dụng RAG Pipeline với cơ chế ghi nhớ."""
    global CONVERSATION_HISTORY 

    if not message_text:
        send_message(sender_id, "Sorry, I can only process text messages.")
        return
    if rag_chain is None:
        send_message(sender_id, "I apologize, our advanced AI support is temporarily unavailable. A human agent will respond shortly.")
        return
        
    # Khởi tạo lịch sử nếu cần
    if sender_id not in CONVERSATION_HISTORY:
        CONVERSATION_HISTORY[sender_id] = []
        
    # Thêm tin nhắn người dùng hiện tại vào lịch sử
    CONVERSATION_HISTORY[sender_id].append({"role": "User", "content": message_text})
    
    # 1. Giai đoạn Viết lại Câu hỏi (Nếu có lịch sử)
    if len(CONVERSATION_HISTORY[sender_id]) > 1:
        rewritten_question = get_rewritten_question(sender_id, message_text)
        print(f"Original Q: {message_text} | Rewritten Q: {rewritten_question}")
        question_to_use = rewritten_question
    else:
        question_to_use = message_text

    # 2. Giai đoạn RAG (sử dụng câu hỏi đã được viết lại/gốc)
    try:
        ai_response = rag_chain.invoke(question_to_use)
        send_message(sender_id, ai_response)
        
        # 3. Thêm phản hồi của AI vào lịch sử để sử dụng cho lần sau
        CONVERSATION_HISTORY[sender_id].append({"role": "AI", "content": ai_response})
        
    except Exception as e:
        print(f"ERROR processing RAG query: {e}")
        send_message(sender_id, "I apologize, there was an error processing your request. Please try again or wait for a human agent.")

    # Giới hạn kích thước lịch sử để tránh sử dụng quá nhiều bộ nhớ/token
    CONVERSATION_HISTORY[sender_id] = CONVERSATION_HISTORY[sender_id][-10:]


# --- III. FASTAPI APPLICATION (WEBHOOK) ---

app = FastAPI()

# 1. Endpoint Webhook - GET (Xác minh Facebook)
@app.get("/webhook")
async def verify_webhook(request: Request):
    """Xác minh Webhook cho Facebook Messenger."""
    
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")
    
    print(f"Verification attempt received: Mode={mode}, Token={token}")

    if mode and token:
        if mode == "subscribe" and token == VERIFY_TOKEN:
            print("Webhook Verified successfully! Returning challenge.")
            return Response(content=challenge, media_type="text/plain") 
        else:
            print("Verification failed: Invalid token.")
            raise HTTPException(status_code=403, detail="Verification token mismatch")
    
    raise HTTPException(status_code=400, detail="Missing required parameters")


# 2. Endpoint Webhook - POST (Nhận tin nhắn)
@app.post("/webhook")
async def handle_message(request: Request):
    """Nhận và xử lý tin nhắn từ Facebook Messenger."""
    
    data = await request.json()
    
    if data.get("object") == "page":
        for entry in data.get("entry",):
            for messaging_event in entry.get("messaging",):
                sender_id = messaging_event.get("sender", {}).get("id")
                
                if messaging_event.get("message") and messaging_event["message"].get("text"):
                    message_text = messaging_event["message"].get("text")
                    # Giao dịch được xử lý trong hàm process_message
                    process_message(sender_id, message_text)

    return "OK"


# --- IV. HƯỚNG DẪN CHẠY MÁY CHỦ CỤC BỘ ---
if __name__ == "__main__":
    print("Starting FastAPI server...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)