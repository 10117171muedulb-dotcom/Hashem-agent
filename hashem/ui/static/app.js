// Hashem Agent studio frontend — vanilla JS, no CDN (works offline).
const $ = (id) => document.getElementById(id);
const api = (path, opts) => fetch(path, Object.assign({ headers: { "Content-Type": "application/json" } }, opts));
const post = (path, body) => api(path, { method: "POST", body: JSON.stringify(body) });

let state = null;
let sessionId = null;

const IMAGE_EXT = /\.(png|jpe?g|webp|gif|bmp)$/i;
const VIDEO_EXT = /\.(mp4|webm|mov)$/i;

/* ---------------------------------- boot */
async function boot() {
  bindTabs();
  try {
    const res = await api("/api/state");
    state = await res.json();
    populate();
    $("conn-status").textContent = "متصل • " + state.version;
  } catch (e) {
    $("conn-status").textContent = "غير متصل!";
  }
  const sessions = await api("/api/sessions").then(r => r.json());
  sessionId = null; // fresh session
  addMsg("ai", "أهلًا! أنا هاشم 🎬\nاطلب مني أي تصميم بالعربية وسأنفذه فورًا: فيديو موشن جرافيك، بوستر، شعار، أو حتى كتاب كامل.");
}

function populate() {
  const cap = state.capabilities || {};
  fill("v-template", cap.video_templates || []);
  fill("v-palette", cap.palettes || []);
  fill("v-music", cap.music_styles || []);
  fill("i-palette", cap.palettes || []);
  fill("i-kind", cap.design_kinds || []);
  fill("i-platform", cap.platforms || []);
  fill("i-style", ["gradient", "circles", "bokeh", "grid", "rays", "diagonal", "radial", "solid"]);
  fill("b-theme", cap.book_themes || []);
  const presets = (state.presets || []).map(p => p.id);
  fill("s-preset", presets);
  if (state.learning) $("s-learning").textContent = JSON.stringify(state.learning, null, 2);
}

function fill(id, values) {
  const el = $(id);
  el.innerHTML = "";
  (values || []).forEach(v => {
    const o = document.createElement("option");
    o.value = v; o.textContent = v;
    el.appendChild(o);
  });
}

/* ---------------------------------- tabs */
function bindTabs() {
  document.querySelectorAll("nav button").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("nav button").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
      btn.classList.add("active");
      $("tab-" + btn.dataset.tab).classList.add("active");
      if (btn.dataset.tab === "gallery") loadGallery();
      if (btn.dataset.tab === "settings") refreshLearning();
    });
  });
}

/* ---------------------------------- chat */
$("chat-send").addEventListener("click", sendChat);
$("chat-text").addEventListener("keydown", e => { if (e.key === "Enter") sendChat(); });

function addMsg(role, text) {
  const log = $("chat-log");
  const div = document.createElement("div");
  div.className = "msg " + role;
  div.textContent = text;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
  return div;
}

function addMedia(container, path) {
  const wrap = document.createElement("div");
  wrap.className = "media";
  if (VIDEO_EXT.test(path)) {
    const v = document.createElement("video");
    v.controls = true; v.src = "/api/file?path=" + encodeURIComponent(path);
    wrap.appendChild(v);
  } else if (IMAGE_EXT.test(path)) {
    const i = document.createElement("img");
    i.src = "/api/file?path=" + encodeURIComponent(path);
    wrap.appendChild(i);
  } else {
    const a = document.createElement("a");
    a.href = "/api/file?path=" + encodeURIComponent(path);
    a.target = "_blank"; a.textContent = "📄 " + path.split(/[\\/]/).pop();
    wrap.appendChild(a);
  }
  container.appendChild(wrap);
}

async function sendChat() {
  const input = $("chat-text");
  const message = input.value.trim();
  if (!message) return;
  input.value = "";
  addMsg("user", message);
  const aiDiv = addMsg("ai", "…");

  const res = await post("/api/chat", { session: sessionId, message });
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let acc = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buffer.indexOf("\n\n")) >= 0) {
      const line = buffer.slice(0, idx); buffer = buffer.slice(idx + 2);
      if (!line.startsWith("data:")) continue;
      let ev; try { ev = JSON.parse(line.slice(5)); } catch { continue; }
      handleEvent(ev, aiDiv, (t) => { acc += t; aiDiv.textContent = acc; });
    }
  }
  $("chat-log").scrollTop = $("chat-log").scrollHeight;
}

function handleEvent(ev, aiDiv, append) {
  switch (ev.type) {
    case "token": append(ev.text || ""); break;
    case "tool_call": {
      const t = addMsg("tool", "⚙️ ينفّذ: " + ev.name);
      break;
    }
    case "tool_result": {
      const r = ev.result || {};
      const path = r.path || (r.outputs && Object.values(r.outputs)[0]);
      if (path) addMedia(aiDiv, path);
      if (r.outputs) Object.values(r.outputs).forEach(p => addMedia(aiDiv, p));
      break;
    }
    case "final": append("\n" + (ev.text || "")); break;
    case "error": append("\n⚠️ " + ev.message); break;
  }
}

/* ---------------------------------- video */
$("v-make").addEventListener("click", async () => {
  const btn = $("v-make"); btn.disabled = true; btn.textContent = "جارٍ الإنشاء…";
  $("v-preview").innerHTML = '<div class="empty">🎞️ يصيّر الإطارات…</div>';
  const points = $("v-points").value.split("\n").map(s => s.trim()).filter(Boolean);
  const res = await post("/api/render", {
    topic: $("v-topic").value || "فيديو",
    points,
    template: $("v-template").value,
    palette: $("v-palette").value,
    audio: $("v-audio").checked,
  });
  const data = await res.json();
  btn.disabled = false; btn.textContent = "🎬 إنشاء الفيديو";
  if (data.path) {
    $("v-preview").innerHTML = "";
    addMedia($("v-preview"), data.path);
  } else {
    $("v-preview").innerHTML = '<div class="empty">فشل: ' + (data.error || "") + '</div>';
  }
});

/* ---------------------------------- image */
$("i-make").addEventListener("click", async () => {
  const btn = $("i-make"); btn.disabled = true; btn.textContent = "جارٍ التصميم…";
  const res = await post("/api/tool", { name: "create_image", arguments: {
    kind: $("i-kind").value, title: $("i-title").value, subtitle: $("i-sub").value,
    palette: $("i-palette").value, style: $("i-style").value, platform: $("i-platform").value,
  }});
  const data = await res.json();
  btn.disabled = false; btn.textContent = "🖼️ تصميم الصورة";
  if (data.path) { $("i-preview").innerHTML = ""; addMedia($("i-preview"), data.path); }
  else $("i-preview").innerHTML = '<div class="empty">فشل: ' + (data.error || "") + '</div>';
});

/* ---------------------------------- book */
$("b-make").addEventListener("click", async () => {
  const btn = $("b-make"); btn.disabled = true; btn.textContent = "جارٍ الإنشاء…";
  const res = await post("/api/tool", { name: "create_book", arguments: {
    text: $("b-text").value, title: $("b-title").value, author: $("b-author").value,
    theme: $("b-theme").value, formats: $("b-formats").value.split(",").map(s => s.trim()).filter(Boolean),
  }});
  const data = await res.json();
  btn.disabled = false; btn.textContent = "📚 إنشاء الكتاب";
  const box = $("b-preview");
  if (data.outputs) {
    box.innerHTML = '<div style="display:flex;flex-direction:column;gap:8px">' +
      Object.entries(data.outputs).map(([fmt, p]) =>
        `<a href="/api/file?path=${encodeURIComponent(p)}" target="_blank" class="ghost" style="display:block;text-align:center;text-decoration:none">⬇ ${fmt.toUpperCase()}</a>`).join("") + "</div>";
  } else box.innerHTML = '<div class="empty">فشل: ' + (data.error || "") + '</div>';
});

/* ---------------------------------- gallery */
async function loadGallery() {
  const res = await post("/api/tool", { name: "list_dir", arguments: { path: state ? state.exports : "." } });
  const data = await res.json();
  const box = $("gallery"); box.innerHTML = "";
  (data.entries || []).filter(e => e.type === "file").forEach(e => {
    const full = (state.exports.replace(/\/$/, "") + "/" + e.name);
    const item = document.createElement("div");
    item.className = "g-item";
    const inner = IMAGE_EXT.test(e.name)
      ? `<img src="/api/file?path=${encodeURIComponent(full)}">`
      : `<div class="meta" style="font-size:28px;text-align:center;padding:20px">${VIDEO_EXT.test(e.name) ? "🎬" : "📄"}</div>`;
    item.innerHTML = inner + `<div class="meta">${e.name}</div>`;
    item.addEventListener("click", () => addMedia($("v-preview"), full));
    box.appendChild(item);
  });
  if (!box.children.length) box.innerHTML = '<div class="empty">لا ملفات بعد</div>';
}
$("g-refresh").addEventListener("click", loadGallery);

/* ---------------------------------- settings */
$("s-save").addEventListener("click", async () => {
  const res = await post("/api/preset", { preset_id: $("s-preset").value, api_key: $("s-key").value });
  const data = await res.json();
  $("s-note").textContent = data.ok ? "تم الحفظ ✔" : "فشل: " + data.error;
});

async function refreshLearning() {
  const res = await post("/api/tool", { name: "learning_stats", arguments: {} });
  const data = await res.json();
  $("s-learning").textContent = JSON.stringify(data, null, 2);
}

boot();
