"""
System API server - OpenAI compatible interface

Usage:
    uvicorn system_api_server:app --host 0.0.0.0 --port 8000 --reload

Example:
    curl http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "my-system",
        "messages": [
          {"role": "system", "content": "{...vega_spec json...}"},
          {"role": "user", "content": "Which origin shows the tightest clustering?"}
        ]
      }'
"""

import json
import uuid
import time
import base64
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# import core system
import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import validate_config
from core import get_session_manager, get_vega_service



# FastAPI App

app = FastAPI(
    title="Visual Analysis System API",
    description="OpenAI-compatible API for the visual analysis system",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models (OpenAI Compatible)


class ChatMessage(BaseModel):
    role: str  # system, user, assistant
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 2000
    temperature: Optional[float] = 0.0
    stream: Optional[bool] = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: Dict[str, Any]
    finish_reason: str


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


# Helper Functions


def extract_vega_spec(messages: List[ChatMessage]) -> Optional[Dict]:
    """
    Extract Vega spec from messages.

    Supportsformats:
    1. JSON in system message
    2. vega_spec field in user message
    """
    for msg in messages:
        if msg.role == "system":
            try:
                # try to parse JSON in system message
                content = msg.content.strip()
                if content.startswith("{"):
                    return json.loads(content)
            except json.JSONDecodeError:
                pass
    
    # try to extract vega_spec from user message
    for msg in messages:
        if msg.role == "user":
            try:
                content = json.loads(msg.content)
                if isinstance(content, dict) and "vega_spec" in content:
                    return content["vega_spec"]
            except (json.JSONDecodeError, TypeError):
                pass
    
    return None


def extract_user_question(messages: List[ChatMessage]) -> tuple[str, bool]:
    """
    Extract the last user message as question, and check if in benchmark mode
    
    Returns:
        tuple: (question, benchmark_mode)
    """
    benchmark_mode = False
    question = ""
    
    for msg in reversed(messages):
        if msg.role == "user":
            content = msg.content
            # check if in evaluation mode
            if "EVALUATION_FORMAT" in content or "OUTPUT FORMAT INSTRUCTIONS" in content:
                benchmark_mode = True
                # extract question (remove format instructions)
                if "EVALUATION_FORMAT" in content:
                    question = content.split("EVALUATION_FORMAT")[0].strip()
                elif "OUTPUT FORMAT INSTRUCTIONS" in content:
                    question = content.split("OUTPUT FORMAT INSTRUCTIONS")[0].strip()
                else:
                    question = content
            else:
                # if content is JSON, try to extract question field
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict) and "question" in parsed:
                        question = parsed["question"]
                    else:
                        question = content
                except (json.JSONDecodeError, TypeError):
                    question = content
            break
    
    return question, benchmark_mode


def extract_final_answer(result: Dict) -> str:
    """Extract final answer from system result"""
    mode = result.get("mode", "")
    
    if mode == "goal_oriented":
        # goal_oriented mode: extract from last iteration
        iterations = result.get("iterations", [])
        if iterations:
            last_iter = iterations[-1]
            decision = last_iter.get("decision", {})
            
            # try to get direct answer
            answer = decision.get("answer", "")
            if answer:
                return answer
            
            # otherwise use key_insights
            insights = decision.get("key_insights", [])
            if insights:
                return "\n".join(insights)
    
    elif mode == "autonomous_exploration":
        # autonomous mode: use final_report
        report = result.get("final_report", {})
        if report:
            summary = report.get("summary", "")
            insights = report.get("key_insights", [])
            if summary:
                return summary
            if insights:
                return "\n".join(insights)
        
        # otherwise merge all explorations' insights
        explorations = result.get("explorations", [])
        all_insights = []
        for exp in explorations:
            analysis = exp.get("analysis_summary", {})
            insights = analysis.get("key_insights", [])
            all_insights.extend(insights)
        if all_insights:
            return "\n".join(all_insights)
    
    elif mode == "chitchat":
        return result.get("response", "")
    
    return str(result.get("response", result.get("message", "")))


def extract_tool_calls(result: Dict) -> List[Dict]:
    """Extract tool call history from system result"""
    mode = result.get("mode", "")
    tool_calls = []
    
    if mode == "goal_oriented":
        iterations = result.get("iterations", [])
        for it in iterations:
            tool_exec = it.get("tool_execution", {})
            if tool_exec and tool_exec.get("tool_name"):
                tool_calls.append({
                    "name": tool_exec.get("tool_name"),
                    "params": tool_exec.get("tool_params", {}),
                    "result": tool_exec.get("tool_result", {})
                })
    
    elif mode == "autonomous_exploration":
        explorations = result.get("explorations", [])
        for exp in explorations:
            tool_exec = exp.get("tool_execution", {})
            if tool_exec and tool_exec.get("tool_name"):
                tool_calls.append({
                    "name": tool_exec.get("tool_name"),
                    "params": tool_exec.get("tool_params", {}),
                    "result": tool_exec.get("tool_result", {})
                })
    
    return tool_calls


def extract_reasoning(result: Dict) -> str:
    """Extract reasoning from system result"""
    mode = result.get("mode", "")
    reasonings = []
    
    if mode == "goal_oriented":
        iterations = result.get("iterations", [])
        for it in iterations:
            decision = it.get("decision", {})
            reasoning = decision.get("reasoning", "")
            if reasoning:
                reasonings.append(reasoning)
    
    elif mode == "autonomous_exploration":
        explorations = result.get("explorations", [])
        for exp in explorations:
            analysis = exp.get("analysis_summary", {})
            reasoning = analysis.get("reasoning", "")
            if reasoning:
                reasonings.append(reasoning)
    
    return "\n\n".join(reasonings)



# API Endpoints

@app.get("/")
async def root():
    """Health check"""
    return {"status": "ok", "service": "Visual Analysis System API"}


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI compatible)"""
    return {
        "object": "list",
        "data": [
            {
                "id": "my-system",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local"
            }
        ]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """
    OpenAI compatible Chat Completions endpoint
    
    Internal call to session_manager.process_query()
    """
    
    # 1. extract vega spec and question
    vega_spec = extract_vega_spec(request.messages)
    if not vega_spec:
        raise HTTPException(
            status_code=400, 
            detail="No vega_spec found in messages. Put it in the system message as JSON."
        )
    
    question, benchmark_mode = extract_user_question(request.messages)
    if benchmark_mode:
        print(f"[System API] Benchmark mode detected: {benchmark_mode}")
    if not question:
        raise HTTPException(
            status_code=400,
            detail="No user question found in messages."
        )
    
    # 2. create session and run
    session_mgr = get_session_manager()
    session_id = session_mgr.create_session(vega_spec)
    
    if not session_id:
        raise HTTPException(
            status_code=500,
            detail="Failed to create session"
        )
    
    try:
        # 3. process query (pass benchmark_mode)
        result = session_mgr.process_query(session_id, question, benchmark_mode=benchmark_mode)
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Query processing failed")
            )
        
        # 4. extract results
        answer = extract_final_answer(result)
        tool_calls = extract_tool_calls(result)
        reasoning = extract_reasoning(result)
        
        # get final spec
        session = session_mgr.get_session(session_id)
        final_spec = session.get("vega_spec", {}) if session else {}
        
        # 5. build OpenAI compatible response
        response_message = {
            "role": "assistant",
            "content": answer,
            # extra fields for evaluation
            "final_spec": final_spec,
            "tool_calls_history": tool_calls,
            "reasoning": reasoning,
            "mode": result.get("mode", ""),
            "iterations": len(result.get("iterations", result.get("explorations", [])))
        }
        
        return ChatCompletionResponse(
            id=f"chatcmpl-{session_id[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=response_message,
                    finish_reason="stop"
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=len(str(request.messages)),
                completion_tokens=len(answer),
                total_tokens=len(str(request.messages)) + len(answer)
            )
        )
        
    finally:
        # clean up session
        if session_id in session_mgr.sessions:
            del session_mgr.sessions[session_id]



# startup check

@app.on_event("startup")
async def startup_event():
    """Validate configuration at startup"""
    errors = validate_config()
    if errors:
        print("  Configuration warnings:")
        for error in errors:
            print(f"  - {error}")
    else:
        print(" Configuration validated")
    
    print(" Visual Analysis System API started")
    print("   Endpoint: http://localhost:8000/v1/chat/completions")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

