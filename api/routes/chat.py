from typing import List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types
from google.genai import errors
from api.utils.security import get_user_id_from_auth, supabase

router = APIRouter(prefix="/api/chat", tags=["Chat"])
gemini_client = genai.Client()
ADMIN_STORE_ID = "fileSearchStores/global-syllabus-abc"

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
            # Check if this error is specifically caused by quota exhaustion or rate limits
            is_quota_error = any(term in str(e).upper() for term in ["429", "QUOTA", "EXHAUSTED", "LIMIT"])
            
            if is_quota_error and idx < len(candidate_models) - 1:
                # Log to runtime server log and fall back to the next model in the list
                print(f"[Fallback Triggered]: {model_name} exhausted. Shifting to {candidate_models[idx+1]}")
                continue
            else:
                # Re-raise error if we have exhausted all models or encountered a non-quota error
                raise HTTPException(status_code=502, detail=f"All failover layers exhausted. API Error: {str(e)}")
