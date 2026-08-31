# api/routes/chat.py
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
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

# OPTIMIZATION: Accept user_store_id from the frontend payload
class ChatPayload(BaseModel):
    message: str
    user_store_id: Optional[str] = None  
    history: List[ChatMessage] = []

@router.post("")
async def chat_with_rag(payload: ChatPayload, user_id: str = Depends(get_user_id_from_auth)):
    user_private_store = payload.user_store_id

    # FAST ROUTE CACHE: Only query Supabase if the frontend didn't pass a cached ID
    if not user_private_store:
        print("💾 Cache Miss: Querying Supabase for user store ID...")
        db_query = supabase.table("user_vector_stores").select("user_store_id").eq("user_id", user_id).execute()
        user_private_store = db_query.data[0].get("user_store_id") if db_query.data else None
    else:
        print("⚡ Cache Hit: Reusing user store ID from frontend...")

    authorized_stores = [ADMIN_STORE_ID]
    if user_private_store:
        authorized_stores.append(user_private_store)

    formatted_history = [
        types.Content(role=turn.role, parts=[types.Part.from_text(text=turn.text)])
        for turn in payload.history
    ]

    tools_config = [types.Tool(file_search=types.FileSearch(file_search_store_names=authorized_stores))]
    candidate_models = ["gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]

    async def response_streamer():
        for idx, model_name in enumerate(candidate_models):
            try:
                response_chunks = gemini_client.chats.create_stream(
                    model=model_name,
                    history=formatted_history,
                    config=types.GenerateContentConfig(tools=tools_config),
                    message=payload.message
                )
                
                for chunk in response_chunks:
                    if chunk.text:
                        yield chunk.text
                return 
                
            except (errors.APIError, errors.ClientError) as e:
                status_code = getattr(e, 'status_code', None)
                is_transient = status_code in [429, 503] or "QUOTA" in str(e).upper() or "UNAVAILABLE" in str(e).upper()
                
                if is_transient and idx < len(candidate_models) - 1:
                    print(f"⚠️ Streaming Fallback: Shifting from {model_name}")
                    continue
                else:
                    yield f"⚠️ Stream Generation Aborted: {str(e)}"
                    return

    return StreamingResponse(response_streamer(), media_type="text/event-stream")
