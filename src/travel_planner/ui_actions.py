"""Action bar under each answer: copy, read aloud, thumbs up/down, retry.

Copy and read-aloud run in the browser (clipboard and speech synthesis), so they
are drawn in a tiny iframe; feedback and retry are native Streamlit widgets.
"""

import json
import re

import streamlit as st

_ICON = 'width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"'
COPY_ICON = f'<svg {_ICON}><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V6a2 2 0 0 1 2-2h8"/></svg>'
CHECK_ICON = f'<svg {_ICON}><path d="M5 12l5 5L20 7"/></svg>'
SPEAK_ICON = f'<svg {_ICON}><path d="M11 5L6 9H3v6h3l5 4V5z"/><path d="M15.5 8.5a5 5 0 0 1 0 7"/><path d="M18.5 6a8.5 8.5 0 0 1 0 12"/></svg>'
STOP_ICON = f'<svg {_ICON}><rect x="7" y="7" width="10" height="10" rx="1.5"/></svg>'

_TEMPLATE = """
<style>
  html, body { margin: 0; background: transparent; }
  .bar { display: flex; gap: 2px; align-items: center; height: 34px; }
  button { background: none; border: none; padding: 6px; border-radius: 6px; cursor: pointer;
           color: #8a8a8a; display: inline-flex; }
  button:hover { color: #c8c8c8; background: rgba(128,128,128,0.15); }
</style>
<div class="bar">
  <button id="copy" title="Copy" aria-label="Copy">__COPY__</button>
  <button id="speak" title="Read aloud" aria-label="Read aloud">__SPEAK__</button>
</div>
<script>
const TEXT = __TEXT__;
const SPEECH = __SPEECH__;
const ICONS = __ICONS__;
const copyBtn = document.getElementById("copy");
const speakBtn = document.getElementById("speak");

function fallbackCopy(text) {
  const ta = document.createElement("textarea");
  ta.value = text; ta.setAttribute("readonly", "");
  ta.style.position = "fixed"; ta.style.opacity = "0";
  document.body.appendChild(ta);
  ta.select(); ta.setSelectionRange(0, text.length);  // iOS needs the range
  let ok = false;
  try { ok = document.execCommand("copy"); } catch (e) {}
  document.body.removeChild(ta);
  return ok;
}

copyBtn.addEventListener("click", async () => {
  let ok = false;
  try { await navigator.clipboard.writeText(TEXT); ok = true; } catch (e) { ok = fallbackCopy(TEXT); }
  if (ok) {
    copyBtn.innerHTML = ICONS.check; copyBtn.title = "Copied";
    setTimeout(() => { copyBtn.innerHTML = ICONS.copy; copyBtn.title = "Copy"; }, 1500);
  }
});

function chunks(text) {
  // Long single utterances get cut off in some browsers, so speak sentence-sized pieces
  const parts = text.match(/[^.!?\\n]+[.!?]*\\s*/g) || [text];
  const out = []; let cur = "";
  for (const p of parts) {
    if ((cur + p).length > 220 && cur) { out.push(cur); cur = ""; }
    cur += p;
  }
  if (cur.trim()) out.push(cur);
  return out;
}

function resetSpeak() { speakBtn.innerHTML = ICONS.speak; speakBtn.title = "Read aloud"; }

speakBtn.addEventListener("click", () => {
  const synth = window.speechSynthesis;
  if (!synth) { speakBtn.title = "Read aloud isn't supported in this browser"; return; }
  if (synth.speaking || synth.pending) { synth.cancel(); resetSpeak(); return; }
  const pieces = chunks(SPEECH);
  pieces.forEach((piece, i) => {
    const u = new SpeechSynthesisUtterance(piece);
    if (i === pieces.length - 1) { u.onend = resetSpeak; u.onerror = resetSpeak; }
    synth.speak(u);
  });
  speakBtn.innerHTML = ICONS.stop; speakBtn.title = "Stop reading";
});
</script>
"""


def to_speech_text(markdown_text: str) -> str:
    """Strip markdown so read-aloud doesn't pronounce symbols or URLs."""
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", markdown_text)  # [title](url) -> title
    t = re.sub(r"https?://\S+", "", t)                             # bare URLs
    t = re.sub(r"`{1,3}", "", t)
    t = re.sub(r"^\s{0,3}#{1,6}\s*", "", t, flags=re.M)           # headings
    t = re.sub(r"^\s*[-*•]\s+", "", t, flags=re.M)                 # bullets
    t = re.sub(r"[*_|>]+", "", t)
    t = t.replace("→", " to ").replace("↔", " to and from ")
    return re.sub(r"[ \t]+", " ", t).strip()


def _js(value) -> str:
    """JSON for embedding inside <script>.

    The answer is model output, so it is passed only as data: JSON-encoded, with
    <, > and & written as unicode escapes so it can never close the script tag or
    be parsed as HTML.
    """
    return (
        json.dumps(value)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def copy_speak_buttons(answer: str) -> None:
    html = (
        _TEMPLATE.replace("__COPY__", COPY_ICON)
        .replace("__SPEAK__", SPEAK_ICON)
        .replace("__TEXT__", _js(answer))
        .replace("__SPEECH__", _js(to_speech_text(answer)))
        .replace("__ICONS__", _js({"copy": COPY_ICON, "check": CHECK_ICON, "speak": SPEAK_ICON, "stop": STOP_ICON}))
    )
    st.iframe(html, width=80, height=36)


def action_bar(msg: dict, *, is_last: bool, on_feedback, on_retry) -> None:
    """Copy · read aloud · 👍/👎 · retry (retry only on the latest answer).

    A horizontal container keeps the bar on one row, also on phones
    (st.columns would stack vertically on narrow screens).
    """
    with st.container(horizontal=True, gap="small", vertical_alignment="center"):
        if not msg.get("error"):
            copy_speak_buttons(msg["content"])
            st.feedback("thumbs", key=f"fb-{msg['id']}", on_change=on_feedback, args=(msg,))
        if is_last:
            st.button("", icon=":material/refresh:", key=f"retry-{msg['id']}", help="Retry",
                      type="tertiary", on_click=on_retry)
