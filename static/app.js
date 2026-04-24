// YTchat frontend. Holds chat history in memory; server is stateless, so
// every /api/chat request sends the full messages array.

let videoId = null;
let messages = [];
let inFlight = false;

const $ = (id) => document.getElementById(id);
const el = {
  url: $("url"),
  loadBtn: $("load-btn"),
  status: $("status"),
  summaryBtn: $("summary-btn"),
  summaryOut: $("summary-out"),
  messages: $("messages"),
  chatInput: $("chat-input"),
  chatForm: $("chat-form"),
  sendBtn: $("send-btn"),
};

function setControls() {
  const hasVideo = videoId !== null;
  el.loadBtn.disabled = inFlight;
  el.summaryBtn.disabled = !hasVideo || inFlight;
  el.chatInput.disabled = !hasVideo || inFlight;
  el.sendBtn.disabled = !hasVideo || inFlight;
}

async function postJSON(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || res.statusText);
  return data;
}

async function load() {
  const url = el.url.value.trim();
  if (!url) return;

  inFlight = true;
  el.status.textContent = "Loading…";
  setControls();

  try {
    const data = await postJSON("/api/load", { url });
    videoId = data.video_id;
    messages = [];
    el.messages.innerHTML = "";
    el.summaryOut.textContent = "";
    el.status.textContent = `Loaded ${data.chunks.length} chunks from ${videoId}`;
  } catch (e) {
    videoId = null;
    el.status.textContent = `Error: ${e.message}`;
  } finally {
    inFlight = false;
    setControls();
  }
}

async function summarize() {
  if (!videoId) return;
  inFlight = true;
  el.summaryOut.textContent = "Summarizing…";
  setControls();

  try {
    const data = await postJSON("/api/summary", { video_id: videoId });
    el.summaryOut.textContent = data.content;
  } catch (e) {
    el.summaryOut.textContent = `Error: ${e.message}`;
  } finally {
    inFlight = false;
    setControls();
  }
}

function renderMessage(role, content) {
  const div = document.createElement("div");
  div.className = `msg msg-${role}`;
  const label = document.createElement("strong");
  label.textContent = role === "user" ? "You: " : "YTchat: ";
  const body = document.createElement("span");
  body.textContent = content;
  div.appendChild(label);
  div.appendChild(body);
  el.messages.appendChild(div);
  div.scrollIntoView({ block: "end" });
}

async function sendChat(text) {
  const trimmed = text.trim();
  if (!videoId || !trimmed) return;

  messages.push({ role: "user", content: trimmed });
  renderMessage("user", trimmed);
  el.chatInput.value = "";

  inFlight = true;
  setControls();

  try {
    const data = await postJSON("/api/chat", { video_id: videoId, messages });
    messages.push({ role: "assistant", content: data.content });
    renderMessage("assistant", data.content);
  } catch (e) {
    renderMessage("assistant", `Error: ${e.message}`);
  } finally {
    inFlight = false;
    setControls();
    el.chatInput.focus();
  }
}

el.loadBtn.addEventListener("click", load);
el.url.addEventListener("keydown", (e) => { if (e.key === "Enter") load(); });
el.summaryBtn.addEventListener("click", summarize);
el.chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  sendChat(el.chatInput.value);
});

setControls();
