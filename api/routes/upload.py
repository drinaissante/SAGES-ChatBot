import shutil
import os
from fastapi import APIRouter, Depends, UploadFile, File
from google import genai
from google.genai import types
from api.utils.security import get_user_id_from_auth, supabase

router = APIRouter(prefix="/upload", tags=["Upload"])
gemini_client = genai.Client()

@router.post("")  
async def upload_document(
    file: UploadFile = File(...), 
    user_id: str = Depends(get_user_id_from_auth)
):
    db_query = supabase.table("user_vector_stores").select("user_store_id").eq("user_id", user_id).execute()
    user_store_id = db_query.data[0].get("user_store_id") if db_query.data else None

    if not user_store_id:
        new_store = gemini_client.file_search_stores.create(
            config=types.CreateFileSearchStoreConfig(display_name=f"Private_Store_{user_id}")
        )
        user_store_id = new_store.name
        supabase.table("user_vector_stores").insert({"user_id": user_id, "user_store_id": user_store_id}).execute()

    temp_path = f"/tmp/{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        gemini_client.files.upload(
            file=temp_path,
            config=types.UploadFileConfig(file_search_store_name=user_store_id)
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return {"status": "processing", "message": "Document dispatched to backend RAG engine."}
