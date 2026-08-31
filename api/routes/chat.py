from typing import List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types
from google.genai import errors
from api.utils.security import get_user_id_from_auth, supabase

router = APIRouter(prefix="/chat", tags=["Chat"])
gemini_client = genai.Client()
ADMIN_STORE_ID = "fileSearchStores/sagesglobaladminsyllabus-sifzorl1hb4b"

class ChatMessage(BaseModel):
    role: str
    text: str

class ChatPayload(BaseModel):
    message: str
    history: List[ChatMessage] = []

@router.post("") # Maps directly to /api/chat
async def chat_with_rag(payload: ChatPayload, user_id: str = Depends(get_user_id_from_auth)):
    db_query = supabase.table("user_vector_stores").select("user_store_id").eq("user_id", user_id).execute()
    user_private_store = db_query.data[0].get("user_store_id") if db_query.data else None

    authorized_stores = [ADMIN_STORE_ID]
    if user_private_store:
        authorized_stores.append(user_private_store)

    formatted_history = [
        types.Content(role=turn.role, parts=[types.Part.from_text(text=turn.text)])
        for turn in payload.history
    ]

    tools_config = [types.Tool(file_search=types.FileSearch(file_search_store_names=authorized_stores))]
    
    # Ordered fallback cascade tier
    candidate_models = ["gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-1.5-flash"]
    
    for idx, model_name in enumerate(candidate_models):
        try:
            chat = gemini_client.chats.create(
                model=model_name,
                history=formatted_history,
                config=types.GenerateContentConfig(tools=tools_config)
            )
            response = chat.send_message(payload.message)
            return {"reply": response.text, "model_used": model_name}
            
        except (errors.APIError, errors.ClientError) as e:
            # Check the HTTP status code directly
            status_code = getattr(e, 'status_code', None)
            
            # CRITICAL OPTIMIZATION: Only transition models if the error is an operational rate limit/server issue
            if status_code in [429, 503] and idx < len(candidate_models) - 1:
                print(f"[Fallback Triggered]: {model_name} rate limited/busy. Shifting to {candidate_models[idx+1]}")
                continue
            else:
                # If it's a 400 bad request (Invalid Store ID), fail instantly instead of stalling for 1 minute
                raise HTTPException(
                    status_code=status_code or 500, 
                    detail=f"RAG Execution aborted. Google API returned: {str(e)}"
                )

