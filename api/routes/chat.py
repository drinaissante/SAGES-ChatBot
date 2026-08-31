from typing import List
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

class ChatPayload(BaseModel):
    message: str
    history: List[ChatMessage] = []

@router.post("")
async def chat_with_rag(payload: ChatPayload, user_id: str = Depends(get_user_id_from_auth)):
    db_query = supabase.table("user_vector_stores").select("user_store_id").eq("user_id", user_id).execute()
    user_private_store = db_query.data.get("user_store_id") if db_query.data else None

    authorized_stores = [ADMIN_STORE_ID]
    if user_private_store:
        authorized_stores.append(user_private_store)

    formatted_history = [
        types.Content(role=turn.role, parts=[types.Part.from_text(text=turn.text)])
        for turn in payload.history
    ]

    tools_config = [types.Tool(file_search=types.FileSearch(file_search_store_names=authorized_stores))]
    candidate_models = ["gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]

    # INNER GENERATOR FUNCTION: Streams text blocks out one by one
    async def response_streamer():
        for idx, model_name in enumerate(candidate_models):
            try:
                # CORRECTION: Shifted from .create() to .create_stream()
                response_chunks = gemini_client.chats.create_stream(
                    model=model_name,
                    history=formatted_history,
                    config=types.GenerateContentConfig(tools=tools_config),
                    message=payload.message # Send message during generation initialization
                )
                
                # Loop through the stream generator chunks from Google
                for chunk in response_chunks:
                    if chunk.text:
                        yield chunk.text
                return # Break generator successfully once finished
                
            except (errors.APIError, errors.ClientError) as e:
                status_code = getattr(e, 'status_code', None)
                is_transient = status_code in [429, 503] or "QUOTA" in str(e).upper() or "UNAVAILABLE" in str(e).upper()
                
                if is_transient and idx < len(candidate_models) - 1:
                    print(f"Streaming Fallback: Shifting from {model_name}")
                    continue
                else:
                    yield f"⚠️ Stream Generation Aborted: {str(e)}"
                    return

    # Return standard event-stream protocol format back to Vercel gateway
    return StreamingResponse(response_streamer(), media_type="text/event-stream")
