"""
Streaming Chat Support
Handles dataset-aware chat with streaming responses.
"""

import json
import uuid
from datetime import datetime
from typing import AsyncGenerator, Optional, List, Dict, Any
from utils.ai_client import get_client, map_model_id
import os


async def stream_chat_response(
    messages: List[Dict[str, str]],
    model: str,
    file_id: Optional[str] = None,
    fingerprint: Optional[Dict[str, Any]] = None,
    context: Optional[str] = None,
    api_key: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """
    Stream chat responses with optional dataset context.
    
    Args:
        messages: Chat message history
        model: Model to use for completion
        file_id: Optional file context
        fingerprint: Optional dataset fingerprint for context
        context: Additional context string
        api_key: Optional API key override
        
    Yields:
        JSON-encoded chat deltas
    """
    
    openrouter_key = os.getenv("OPENROUTER_API_KEY") or api_key
    mapped_model = map_model_id(model)
    client = get_client(mapped_model, openrouter_key)
    
    # Build system context
    system_msg = "You are Optima AI, an expert data analyst."
    
    if fingerprint:
        system_msg += f"\n\nDataset context:\n{json.dumps(fingerprint, indent=2)}"
    
    if context:
        system_msg += f"\n\nAdditional context:\n{context}"
    
    # Prepare API messages
    api_messages = [
        {"role": "system", "content": system_msg},
        *messages
    ]
    
    try:
        # Create completion stream
        stream = client.chat.completions.create(
            model=mapped_model,
            messages=api_messages,
            stream=True,
            temperature=0.7,
            max_tokens=2000,
        )
        
        # Process stream
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                delta = chunk.choices[0].delta.content
                # Yield as Server-Sent Event format
                yield f"data: {json.dumps({'delta': delta})}\n\n"
        
        # Signal end
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        # Stream error as JSON
        error_msg = f"Stream error: {str(e)}"
        yield f"data: {json.dumps({'error': error_msg})}\n\n"


def create_chat_message(
    role: str,
    content: str,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a chat message with metadata.
    
    Args:
        role: "user" or "assistant"
        content: Message content
        model: Model used (if assistant)
        
    Returns:
        Chat message object
    """
    return {
        "id": str(uuid.uuid4()),
        "role": role,
        "content": content,
        "timestamp": int(datetime.utcnow().timestamp() * 1000),
        "model": model,
    }
