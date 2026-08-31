from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import chat, upload

app = FastAPI(title="SAGES")

# Ensure this block is placed BEFORE app.include_router calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all domains (perfect for testing)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount your modular router controllers
app.include_router(chat.router, prefix="/api")
app.include_router(upload.router, prefix="/api")
