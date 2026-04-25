// YTchat frontend. Holds chat history in memory; server is stateless, so
// every /api/chat request sends the full messages array.

let videoId = null;
let messages = [];
let inFlight = false;
let player = null;

// The IFrame API calls window.onYouTubeIframeAPIReady when it finishes
// loading. Wrap that in a promise so we can await it wherever we first
// need the player, regardless of whether the API loads before or after
// the user hits Load.
const ytReady = new Promise((resolve) => {
  window.onYouTubeIframeAPIReady = resolve;
});

const $ = (id) => document.getElementById(id);
const el = {
  url: $("url"),
  loadForm: $("load-form"),
  loadBtn: $("load-btn"),
  status: $("status"),
  playerSection: $("player-section"),
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
  // Read as text first so we can surface non-JSON server errors (FastAPI
  // returns "Internal Server Error" as plain text on unhandled 500s).
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { /* leave null */ }
  if (!res.ok) {
    const detail = (data && data.detail) || text || res.statusText;
    const err = new Error(`${res.status} ${detail}`);
    err.status = res.status;
    throw err;
  }
  return data;
}

async function load() {
  const url = el.url.value.trim();
  if (!url) return;

  inFlight = true;
  el.status.textContent = "Loading transcript…";
  setControls();

  try {
    // Two-stage flow so the user sees a distinct status when we fall back
    // to Whisper (which can take tens of seconds). The server also caches
    // results, so a second viewer of the same video hits the first call
    // instantly regardless of which path produced the chunks.
    let data;
    try {
      data = await postJSON("/api/load", { url, allow_whisper: false });
    } catch (e) {
      if (e.status !== 422) throw e;
      el.status.textContent = "Generating transcript with Whisper… (this may take a minute)";
      data = await postJSON("/api/load", { url, allow_whisper: true });
    }
    videoId = data.video_id;
    messages = [];
    el.messages.innerHTML = "";
    el.summaryOut.textContent = "";
    el.status.textContent = `Loaded ${data.chunks.length} chunks from ${videoId}`;
    await mountPlayer(videoId);
  } catch (e) {
    videoId = null;
    el.status.textContent = `Error: ${e.message}`;
  } finally {
    inFlight = false;
    setControls();
  }
}

async function mountPlayer(id) {
  await ytReady;
  el.playerSection.hidden = false;
  if (player) {
    // Swap the current video without re-creating the iframe.
    player.loadVideoById(id);
  } else {
    player = new YT.Player("player", {
      width: "100%",
      height: "100%",
      videoId: id,
    });
  }
}

async function summarize() {
  if (!videoId) return;
  inFlight = true;
  el.summaryOut.textContent = "Summarizing…";
  setControls();

  try {
    const data = await postJSON("/api/summary", { video_id: videoId });
    el.summaryOut.replaceChildren(renderWithTimestamps(data.content));
  } catch (e) {
    el.summaryOut.textContent = `Error: ${e.message}`;
  } finally {
    inFlight = false;
    setControls();
  }
}

// Matches [MM:SS] and [HH:MM:SS]. Group 3 is present only for hours.
const TIMESTAMP_RE = /\[(\d{1,2}):(\d{2})(?::(\d{2}))?\]/g;

function timestampSeconds(g1, g2, g3) {
  return g3 !== undefined
    ? Number(g1) * 3600 + Number(g2) * 60 + Number(g3)
    : Number(g1) * 60 + Number(g2);
}

// Build a DOM fragment with [MM:SS] spans replaced by anchor tags. Text
// content always flows through textContent so LLM output can never inject
// HTML. The href/target pair is kept as a graceful fallback: if JS fails
// or the player isn't ready yet, the link still opens YouTube at the
// right timestamp in a new tab.
function renderWithTimestamps(text) {
  const frag = document.createDocumentFragment();
  let last = 0;
  for (const match of text.matchAll(TIMESTAMP_RE)) {
    const [full, g1, g2, g3] = match;
    if (match.index > last) {
      frag.appendChild(document.createTextNode(text.slice(last, match.index)));
    }
    const seconds = timestampSeconds(g1, g2, g3);
    const a = document.createElement("a");
    a.className = "ts";
    a.href = `https://www.youtube.com/watch?v=${videoId}&t=${seconds}s`;
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    a.dataset.seconds = String(seconds);
    a.textContent = full;
    frag.appendChild(a);
    last = match.index + full.length;
  }
  if (last < text.length) {
    frag.appendChild(document.createTextNode(text.slice(last)));
  }
  return frag;
}

function renderMessage(role, content) {
  const div = document.createElement("div");
  div.className = `msg msg-${role}`;
  const chip = document.createElement("span");
  chip.className = "role";
  chip.textContent = role === "user" ? "You" : "YTchat";
  div.appendChild(chip);
  div.appendChild(renderWithTimestamps(content));
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

el.loadForm.addEventListener("submit", (e) => { e.preventDefault(); load(); });
el.summaryBtn.addEventListener("click", summarize);
el.chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  sendChat(el.chatInput.value);
});

// Delegated click handler: if a [MM:SS] link is clicked and the player is
// ready, seek in-page instead of navigating. If the player isn't ready
// yet we let the default href take over and open YouTube in a new tab.
document.addEventListener("click", (e) => {
  const a = e.target.closest("a.ts");
  if (!a || !player) return;
  e.preventDefault();
  player.seekTo(Number(a.dataset.seconds), true);
  player.playVideo();
});

setControls();
