from fastapi import FastAPI
from api.routes import chat, upload

app = FastAPI(title="Production Multi-Tenant RAG Backend")

# Mount your modular router controllers
app.include_router(chat.router)
app.include_router(upload.router)
