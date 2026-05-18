"""FastAPI web interface — run with: make serve"""
from dataclasses import dataclass, field
from uuid import uuid4

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from fitness_coach.hub import build as build_hub

# ── Session store (in-memory; resets on server restart) ───────────────────────

@dataclass
class _Session:
    history: list = field(default_factory=list)
    pending_message: str | None = None  # held during topic-change confirmation

_sessions: dict[str, _Session] = {}

_AFFIRMATIVE  = {"yes", "y", "yeah", "yep", "sure", "yup", "ok", "okay", "new", "fresh", "start fresh", "start over"}
_CONTINUATION = {"no", "n", "nope", "nah", "continue", "same", "keep going", "same topic", "not new"}

# ── API ────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Fitness Coach")


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    route: str
    confidence: float | None
    needs_input: bool
    session_id: str


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    session_id = req.session_id or str(uuid4())
    session = _sessions.setdefault(session_id, _Session())

    user_input = req.message

    # Mirror demo.py topic-change confirmation logic
    if session.pending_message is not None:
        answer = user_input.lower().strip()
        if answer in _CONTINUATION:
            user_input = session.pending_message
        elif answer in _AFFIRMATIVE:
            session.history.clear()
            user_input = session.pending_message
        else:
            # Direct intent — start fresh and process the new message as-is
            session.history.clear()
        session.pending_message = None

    session.history.append(HumanMessage(content=user_input))

    result = build_hub().invoke({"messages": session.history})

    route    = result.get("route", "—")
    response = result.get("response") or ""

    if route == "NEW_TOPIC":
        session.history.pop()
        session.pending_message = user_input
    else:
        session.history.append(AIMessage(content=response))

    return ChatResponse(
        response=response,
        route=route,
        confidence=result.get("confidence"),
        needs_input=result.get("needs_input", False),
        session_id=session_id,
    )


@app.delete("/session/{session_id}")
async def clear_session(session_id: str) -> dict:
    _sessions.pop(session_id, None)
    return {"cleared": True}


# ── UI ─────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return _HTML


_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Fitness Coach</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #f0f2f5; height: 100dvh;
    display: flex; flex-direction: column;
  }
  header {
    background: #1a1a2e; color: #fff;
    padding: 14px 20px;
    display: flex; justify-content: space-between; align-items: center;
    flex-shrink: 0;
  }
  header h1 { font-size: 17px; font-weight: 600; letter-spacing: .02em; }
  #new-chat {
    background: transparent; border: 1px solid rgba(255,255,255,.35);
    color: #fff; padding: 5px 14px; border-radius: 6px;
    cursor: pointer; font-size: 13px;
  }
  #new-chat:hover { background: rgba(255,255,255,.1); }
  #messages {
    flex: 1; overflow-y: auto;
    padding: 20px 16px; display: flex; flex-direction: column; gap: 14px;
  }
  .bubble { max-width: 72%; word-wrap: break-word; line-height: 1.5; }
  .bubble.user {
    align-self: flex-end;
    background: #1a1a2e; color: #fff;
    padding: 10px 15px; border-radius: 18px 18px 4px 18px;
    font-size: 15px;
  }
  .bubble.assistant {
    align-self: flex-start;
    background: #fff; color: #1a1a2e;
    padding: 12px 15px; border-radius: 18px 18px 18px 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,.1);
    font-size: 15px; white-space: pre-wrap;
  }
  .meta { font-size: 11px; color: #aaa; margin-top: 5px; }
  .typing { align-self: flex-start; background: #fff; padding: 12px 18px;
            border-radius: 18px; box-shadow: 0 1px 3px rgba(0,0,0,.1);
            color: #aaa; font-size: 20px; letter-spacing: 3px; }
  #input-row {
    display: flex; gap: 8px; padding: 14px 16px;
    background: #fff; border-top: 1px solid #e8e8e8; flex-shrink: 0;
  }
  #msg {
    flex: 1; padding: 11px 16px; border: 1px solid #ddd; border-radius: 24px;
    font-size: 15px; outline: none;
  }
  #msg:focus { border-color: #1a1a2e; }
  #send {
    background: #1a1a2e; color: #fff; border: none;
    padding: 11px 22px; border-radius: 24px;
    cursor: pointer; font-size: 15px; font-weight: 500;
  }
  #send:hover { background: #2d2d50; }
  #send:disabled { opacity: .45; cursor: not-allowed; }
</style>
</head>
<body>
<header>
  <h1>Fitness Coach</h1>
  <button id="new-chat">New Chat</button>
</header>
<div id="messages">
  <div class="bubble assistant">
    Hi! I can help you plan workouts, answer exercise questions, or log your training. What can I do for you?
  </div>
</div>
<div id="input-row">
  <input id="msg" type="text" placeholder="Ask me anything…" autocomplete="off" autofocus>
  <button id="send">Send</button>
</div>
<script>
  let sessionId = null;
  const feed   = document.getElementById("messages");
  const input  = document.getElementById("msg");
  const sendBtn = document.getElementById("send");

  function scrollDown() { feed.scrollTop = feed.scrollHeight; }

  function addBubble(role, text, meta) {
    const wrap = document.createElement("div");
    wrap.className = "bubble " + role;
    const body = document.createElement("div");
    body.textContent = text;
    wrap.appendChild(body);
    if (meta) {
      const m = document.createElement("div");
      m.className = "meta";
      m.textContent = meta;
      wrap.appendChild(m);
    }
    feed.appendChild(wrap);
    scrollDown();
  }

  function showTyping() {
    const d = document.createElement("div");
    d.className = "typing"; d.id = "typing"; d.textContent = "...";
    feed.appendChild(d); scrollDown();
  }
  function hideTyping() { document.getElementById("typing")?.remove(); }

  async function send() {
    const text = input.value.trim();
    if (!text || sendBtn.disabled) return;
    input.value = "";
    sendBtn.disabled = true;
    addBubble("user", text);
    showTyping();
    try {
      const res = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, session_id: sessionId }),
      });
      const data = await res.json();
      sessionId = data.session_id;
      hideTyping();
      let meta = "Route: " + data.route;
      if (data.confidence != null)
        meta += "  ·  Confidence: " + Math.round(data.confidence * 100) + "%";
      addBubble("assistant", data.response, meta);
    } catch {
      hideTyping();
      addBubble("assistant", "Something went wrong — please try again.");
    } finally {
      sendBtn.disabled = false;
      input.focus();
    }
  }

  document.getElementById("new-chat").addEventListener("click", async () => {
    if (sessionId) { await fetch("/session/" + sessionId, { method: "DELETE" }); sessionId = null; }
    feed.innerHTML = "";
    addBubble("assistant", "Starting fresh! What can I help you with?");
  });

  sendBtn.addEventListener("click", send);
  input.addEventListener("keydown", e => { if (e.key === "Enter") send(); });
</script>
</body>
</html>"""
