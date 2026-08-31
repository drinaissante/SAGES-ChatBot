import os
from typing import Optional
from fastapi import Header, HTTPException, status
from supabase import create_client, Client

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("Missing Supabase configuration variables.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

def get_user_id_from_auth(authorization: Optional[str] = Header(None)) -> str:
    """Decodes the client bearer token against Supabase Auth to safely verify identity."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header."
        )
    
    jwt_token = authorization.split(" ")[1]
    try:
        user_response = supabase.auth.get_user(jwt_token)
        return user_response.user.id
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired Supabase user session token."
        )
