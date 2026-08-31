# api/routes/chat.py
from typing import List, Optional
import anyio  # Built into FastAPI/Starlette
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
    user_store_id: Optional[str] = None  
    history: List[ChatMessage] = []

@router.post("")
async def chat_with_rag(payload: ChatPayload, user_id: str = Depends(get_user_id_from_auth)):
    user_private_store = payload.user_store_id
    user_msg_clean = payload.message.lower().strip()

    # ⚡ OPTIMIZATION A: If it's a simple greeting, bypass the RAG vector engine entirely
    # This prevents cold-starting your file search stores for a simple "hey"
    is_greeting = user_msg_clean in ["hey", "hi", "hello", "yo", "greetings"]

    if is_greeting:
        tools_config = None
    else:
        if not user_private_store:
            db_query = supabase.table("user_vector_stores").select("user_store_id").eq("user_id", user_id).execute()
            user_private_store = db_query.data.get("user_store_id") if db_query.data else None

        authorized_stores = [ADMIN_STORE_ID]
        if user_private_store:
            authorized_stores.append(user_private_store)
        tools_config = [types.Tool(file_search=types.FileSearch(file_search_store_names=authorized_stores))]

    formatted_history = [
        types.Content(role=turn.role, parts=[types.Part.from_text(text=turn.text)])
        for turn in payload.history
    ]

    candidate_models = ["gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]

    # ⚡ OPTIMIZATION B: Asynchronous streamer wrapper
    async def response_streamer():
        for idx, model_name in enumerate(candidate_models):
            try:
                chat = gemini_client.chats.create(
                    model=model_name,
                    history=formatted_history,
                    config=types.GenerateContentConfig(tools=tools_config) if tools_config else None
                )
                
                # We use a helper function to iterate through Google's sync stream on an async worker thread
                def get_chunks():
                    return chat.send_message_stream(payload.message)

                # Offload the blocking SDK iteration away from FastAPI's main thread
                response_stream = await anyio.to_thread.run_sync(get_chunks)
                
                for chunk in response_stream:
                    if chunk.text:
                        escaped_text = chunk.text.replace("\n", "\\n")
                        yield f"data: {escaped_text}\n\n"
                        # Sleep momentarily to clear the ASGI channel buffer
                        await anyio.sleep(0.01)
                return  
                
            except (errors.APIError, errors.ClientError) as e:
                status_code = getattr(e, 'status_code', None)
                is_transient = status_code in [429, 503] or "QUOTA" in str(e).upper() or "UNAVAILABLE" in str(e).upper()
                
                if is_transient and idx < len(candidate_models) - 1:
                    continue
                else:
                    yield f"data: ⚠️ Stream Aborted: {str(e)}\n\n"
                    return

    # ⚡ OPTIMIZATION C: Add direct no-buffering proxy directives to the headers
    return StreamingResponse(
        response_streamer(), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Stops Vercel/Nginx from compressing the stream
        }
    )
