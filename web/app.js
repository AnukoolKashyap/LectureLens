"use strict";

const $ = (s, root = document) => root.querySelector(s);
const $$ = (s, root = document) => [...root.querySelectorAll(s)];
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

const STAGES = ["Transcript", "Chunking", "Embedding", "Retrieval", "Summary", "Output"];
const STAGE_LABELS = [
  "Fetching the transcript…",
  "Cleaning and chunking…",
  "Embedding into the vector store…",
  "Retrieving key passages…",
  "Writing the summary with Mistral…",
  "Assembling the output…",
];
const SUGGESTIONS = [
  "What is the main idea?",
  "Explain the hardest concept simply",
  "What examples are used?",
  "Give me 3 revision questions",
];
const EXAMPLES = [
  "https://www.youtube.com/watch?v=…",
  "Paste a lecture link…",
  "Paste a tutorial link…",
  "Paste a seminar recording…",
];
const BRANCH_COLORS = ["#7c6bff", "#22d3ee", "#f472b6", "#34d399", "#fbbf24", "#fb7185"];

const state = {
  data: null, chat: [], player: null, ytReady: false, pendingVideo: null, videoKey: null,
  rating: 4, busy: false, tick: null, marks: [],
};

// ------------------------------------------------------------------ utils

const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function toSeconds(label) {
  const parts = String(label).match(/\d+/g) || [];
  return parts.slice(-3).reduce((acc, p) => acc * 60 + Number(p), 0);
}

function fmtDuration(sec) {
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = Math.round(sec % 60);
  return h ? `${h}h ${m}m` : `${m}:${String(s).padStart(2, "0")}`;
}

const TIME = "\\d{1,2}:\\d{2}(?::\\d{2})?";
const tsChip = (t) => `<button class="ts" data-t="${toSeconds(t)}" type="button">${esc(t)}</button>`;

function linkTimes(text) {
  return esc(text).replace(new RegExp(`[\\[(](${TIME})[\\])]`, "g"), (_, t) => tsChip(t));
}

function renderMarkdown(src) {
  const inline = (s) =>
    linkTimes(s).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/`([^`]+)`/g, "<code>$1</code>");
  const out = [];
  let list = null;
  for (const raw of String(src).split("\n")) {
    const line = raw.trim();
    const item = line.match(/^(?:[-*•]|\d+[.)])\s+(.*)/);
    if (item) { (list = list || []).push(`<li>${inline(item[1])}</li>`); continue; }
    if (list) { out.push(`<ul>${list.join("")}</ul>`); list = null; }
    if (line) out.push(`<p>${inline(line)}</p>`);
  }
  if (list) out.push(`<ul>${list.join("")}</ul>`);
  return out.join("");
}

/** Wrap each word of an element's text in a span so it can fade in. */
function wordReveal(el) {
  if (reduceMotion) return;
  let n = 0;
  const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, {
    acceptNode: (node) => (node.parentElement.closest(".ts") ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT),
  });
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  for (const node of nodes) {
    const frag = document.createDocumentFragment();
    for (const part of node.textContent.split(/(\s+)/)) {
      if (!part) continue;
      if (/^\s+$/.test(part)) { frag.append(part); continue; }
      const span = document.createElement("span");
      span.className = "w";
      span.style.setProperty("--w", Math.min(n++, 120));
      span.textContent = part;
      frag.append(span);
    }
    node.replaceWith(frag);
  }
}

function countUp(el, target, decimals, suffix) {
  if (reduceMotion) { el.textContent = target.toFixed(decimals) + suffix; return; }
  const start = performance.now(), dur = 1200;
  const step = (now) => {
    const k = Math.min(1, (now - start) / dur);
    const eased = 1 - Math.pow(1 - k, 4);
    el.textContent = (target * eased).toFixed(decimals) + suffix;
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function toast(text) {
  const t = document.createElement("div");
  t.className = "toast";
  t.textContent = text;
  $("#toasts").append(t);
  setTimeout(() => t.remove(), 3300);
}

async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) { /* keep statusText */ }
    throw new Error(detail);
  }
  return res;
}

// ------------------------------------------------------------ global chrome

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem("theme", theme); } catch (_) { /* storage blocked */ }
}
(() => {
  let saved = null;
  try { saved = localStorage.getItem("theme"); } catch (_) { /* storage blocked */ }
  applyTheme(saved || (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark"));
})();
$("#theme-btn").addEventListener("click", () => {
  applyTheme(document.documentElement.dataset.theme === "light" ? "dark" : "light");
  $$("#theme-btn svg").forEach((s) => { s.style.animation = "none"; void s.offsetWidth; s.style.animation = ""; });
});

// Ripple on buttons
document.addEventListener("pointerdown", (e) => {
  const btn = e.target.closest(".btn, .send");
  if (!btn || reduceMotion) return;
  const r = btn.getBoundingClientRect();
  const size = Math.max(r.width, r.height);
  const dot = document.createElement("span");
  dot.className = "ripple";
  dot.style.cssText = `width:${size}px;height:${size}px;left:${e.clientX - r.left - size / 2}px;top:${e.clientY - r.top - size / 2}px`;
  btn.append(dot);
  setTimeout(() => dot.remove(), 650);
});

// Spotlight that follows the cursor
document.addEventListener("pointermove", (e) => {
  const card = e.target.closest(".spot");
  if (!card) return;
  const r = card.getBoundingClientRect();
  card.style.setProperty("--mx", `${e.clientX - r.left}px`);
  card.style.setProperty("--my", `${e.clientY - r.top}px`);
});

// Settings popover
$("#settings-btn").addEventListener("click", (e) => {
  e.stopPropagation();
  const open = $("#settings").classList.toggle("open");
  $("#settings-btn").setAttribute("aria-expanded", String(open));
});
document.addEventListener("click", (e) => {
  if (!e.target.closest("#settings")) $("#settings").classList.remove("open");
});
$("#topk").addEventListener("input", (e) => ($("#topk-out").textContent = e.target.value));

// Cycling placeholder
(() => {
  const input = $("#url");
  let i = 0, ch = 0, deleting = false;
  const loop = () => {
    if (document.activeElement === input || input.value) { input.placeholder = EXAMPLES[0]; return setTimeout(loop, 800); }
    const word = EXAMPLES[i];
    ch += deleting ? -1 : 1;
    input.placeholder = word.slice(0, ch);
    let wait = deleting ? 25 : 55;
    if (!deleting && ch >= word.length) { deleting = true; wait = 1800; }
    else if (deleting && ch <= 0) { deleting = false; i = (i + 1) % EXAMPLES.length; wait = 300; }
    setTimeout(loop, wait);
  };
  if (reduceMotion) input.placeholder = EXAMPLES[0]; else loop();
})();

async function loadStatus() {
  const badge = $("#model-badge");
  try {
    const s = await (await api("/api/status")).json();
    $("#profile").innerHTML = s.profiles.map((p) => `<option value="${esc(p.key)}">${esc(p.label)}</option>`).join("");
    $("#topk").value = s.top_k;
    $("#topk-out").textContent = s.top_k;
    $("#offline").checked = s.offline;
    $(".badge-text", badge).textContent = s.llm ? s.model : "no API key — offline only";
    badge.className = "badge " + (s.llm ? "ok" : "warn");
  } catch (_) {
    $(".badge-text", badge).textContent = "server offline";
    badge.className = "badge warn";
  }
}

// ------------------------------------------------------------- summarising

function renderSteps(active, failed) {
  $("#steps").innerHTML = STAGES.map((label, i) => {
    const n = i + 1;
    const cls = failed && n === active ? "failed" : n < active ? "done" : n === active ? "active" : "";
    const mark = cls === "done" ? "✓" : cls === "failed" ? "!" : n;
    return `<li class="${cls}"><span class="node">${mark}</span>${label}</li>`;
  }).join("");
  $("#pipe-fill").style.width = `${Math.max(4, ((active - (failed ? 1 : 0.5)) / STAGES.length) * 100)}%`;
  $("#pipe-label").textContent = failed ? "Stopped" : STAGE_LABELS[active - 1] || "Starting…";
}

async function summarise(useSample) {
  if (state.busy) return;
  const url = $("#url").value.trim();
  if (!url && !useSample) {
    const form = $("#url-form");
    form.classList.remove("shake"); void form.offsetWidth; form.classList.add("shake");
    $("#url").focus();
    return;
  }

  state.busy = true;
  $("#error").hidden = true;
  $("#progress").hidden = false;
  $("#url-form").classList.add("busy");
  $("#go").disabled = true;
  $("#go span").textContent = "Working…";
  let step = 1;
  renderSteps(step);
  const started = Date.now();
  const timer = setInterval(() => {
    $("#progress-note").textContent = `${((Date.now() - started) / 1000).toFixed(1)}s`;
  }, 100);

  try {
    // A playlist link, or several links at once, runs as one course.
    const isPlaylist = /[?&]list=/.test(url) && !/\/watch\?|youtu\.be\//.test(url);
    const many = url.split(/[\s,]+/).filter(Boolean);
    const res = await api("/api/summarize", {
      url: isPlaylist || many.length > 1 ? "" : url,
      playlist: isPlaylist ? url : "",
      urls: !isPlaylist && many.length > 1 ? many : [],
      use_sample: useSample,
      profile: $("#profile").value || "lecture",
      top_k: Number($("#topk").value),
      offline: $("#offline").checked,
      reference: $("#reference").value,
    });
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let nl;
      while ((nl = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, nl).trim();
        buffer = buffer.slice(nl + 1);
        if (!line) continue;
        const ev = JSON.parse(line);
        if (ev.type === "progress") { step = ev.step; renderSteps(step); }
        else if (ev.type === "error") throw Object.assign(new Error(ev.message), { detail: ev.detail });
        else if (ev.type === "result") {
          renderSteps(STAGES.length + 1);
          $("#pipe-label").textContent = "Done";
          $("#pipe-fill").style.width = "100%";
          await new Promise((r) => setTimeout(r, reduceMotion ? 0 : 450));
          showWorkspace(ev.data);
        }
      }
    }
  } catch (e) {
    renderSteps(step, true);
    const extra = e.detail && e.detail !== e.message ? `<code>${esc(e.detail)}</code>` : "";
    $("#error").innerHTML = `<strong>${esc(e.message)}</strong>${extra}`;
    $("#error").hidden = false;
  } finally {
    clearInterval(timer);
    state.busy = false;
    $("#url-form").classList.remove("busy");
    $("#go").disabled = false;
    $("#go span").textContent = "Summarise";
  }
}

$("#url-form").addEventListener("submit", (e) => { e.preventDefault(); summarise(false); });
$("#sample-btn").addEventListener("click", () => summarise(true));
$("#new-btn").addEventListener("click", () => {
  if (state.player && state.player.pauseVideo) state.player.pauseVideo();
  $("#workspace").hidden = true;
  $("#landing").hidden = false;
  $("#progress").hidden = true;
  restartAnimations($("#landing"));
  $("#url").select();
});

function restartAnimations(root) {
  $$(".reveal, .enter", root).forEach((el) => { el.style.animation = "none"; void el.offsetWidth; el.style.animation = ""; });
}

// ------------------------------------------------------------------ player

function videoId(url) {
  const m = String(url).match(/(?:v=|youtu\.be\/|embed\/|shorts\/)([\w-]{11})/);
  return m ? m[1] : null;
}

const EMBED = "https://www.youtube.com/embed/";
const EMBED_ALLOW = "accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share";

function embedUrl(id, start) {
  let u = EMBED + id + "?enablejsapi=1&rel=0&modestbranding=1";
  if (start != null) u += "&autoplay=1&start=" + Math.floor(start);
  return u;
}

/** Paint the player.
 *
 * The iframe goes in immediately rather than waiting for the IFrame API.
 * That script is third-party and is routinely blocked by extensions or a
 * slow network; waiting on it used to leave a black box with no player and
 * no error. The API, when it loads, attaches to the iframe already on the
 * page and upgrades seeking from a reload to a real seekTo().
 */
function mountPlayer(id) {
  if (!id) { $("#no-video").hidden = false; $("#player").hidden = true; return; }
  $("#no-video").hidden = true;
  $("#player").hidden = false;
  state.videoKey = id;

  if (state.player && state.player.cueVideoById) { state.player.cueVideoById(id); return; }

  $("#player").innerHTML =
    '<iframe id="ytframe" src="' + embedUrl(id) + '" title="YouTube video player" ' +
    'frameborder="0" allow="' + EMBED_ALLOW + '" allowfullscreen></iframe>';
  state.player = null;
  attachPlayerApi();
}

/** Upgrade the plain iframe to an API-controlled player, if the API is here. */
function attachPlayerApi() {
  if (!state.ytReady || state.player || !window.YT || !YT.Player) return;
  if (!document.getElementById("ytframe")) return;
  try {
    state.player = new YT.Player("ytframe", {});
  } catch (err) {
    console.warn("YouTube API attach failed — the embed still plays:", err);
    state.player = null;
  }
}

window.onYouTubeIframeAPIReady = () => {
  state.ytReady = true;
  state.pendingVideo = null;
  attachPlayerApi();
};

// If the API has not arrived by now it is blocked or unreachable. The embed
// above still plays; say so once in the console so the cause is visible.
setTimeout(() => {
  if (!state.ytReady) {
    console.warn("youtube.com/iframe_api did not load — the video still plays, " +
                 "but clicking a timestamp reloads the embed instead of seeking.");
  }
}, 6000);

function seek(seconds) {
  attachPlayerApi();
  if (state.player && state.player.seekTo) {
    state.player.seekTo(seconds, true);
    if (state.player.playVideo) state.player.playVideo();
  } else {
    // No API: reload the embed at the requested moment. Less smooth, but it
    // still takes you to the right place.
    const frame = document.getElementById("ytframe");
    if (!frame || !state.videoKey) { toast("No video to seek in"); return; }
    frame.src = embedUrl(state.videoKey, seconds);
  }
  const r = $(".video-wrap").getBoundingClientRect();
  if (r.bottom < 80 || r.top > innerHeight) $(".video-wrap").scrollIntoView({ behavior: "smooth", block: "start" });
}

document.addEventListener("click", (e) => {
  const chip = e.target.closest("[data-t]");
  if (chip) { e.stopPropagation(); seek(Number(chip.dataset.t)); }
});

/** Follow the playhead: highlight the current timeline entry. */
function startTracking() {
  clearInterval(state.tick);
  state.tick = setInterval(() => {
    const p = state.player;
    if (!p || !p.getCurrentTime || !state.marks.length) return;
    const t = p.getCurrentTime();
    const playing = p.getPlayerState && p.getPlayerState() === 1;
    let idx = -1;
    state.marks.forEach((m, i) => { if (t >= m.start) idx = i; });
    $$("#timeline li").forEach((li, i) => {
      li.classList.toggle("now", i === idx);
      li.classList.toggle("past", i < idx);
      const bar = $(".tl-bar i", li);
      if (!bar) return;
      if (i < idx) bar.style.width = "100%";
      else if (i > idx) bar.style.width = "0";
      else {
        const end = state.marks[i + 1] ? state.marks[i + 1].start : (state.data.stats.duration || t + 1);
        bar.style.width = `${Math.min(100, ((t - state.marks[i].start) / Math.max(1, end - state.marks[i].start)) * 100)}%`;
      }
    });
    const np = $("#now-playing");
    if (playing && idx >= 0) {
      const text = state.marks[idx].topic;
      if ($("#now-text").textContent !== text) $("#now-text").textContent = text;
      np.hidden = false;
    } else np.hidden = true;
  }, 400);
}

// --------------------------------------------------------------- workspace

function showWorkspace(d) {
  state.data = d;
  state.chat = [];
  $("#landing").hidden = true;
  $("#workspace").hidden = false;
  restartAnimations($("#workspace"));
  scrollTo({ top: 0 });

  mountPlayer(videoId(d.video.url));
  document.title = `${d.video.title} · LectureLens`;
  $("#title").textContent = d.video.title;
  const st = d.stats;
  const chips = [
    d.video.author && esc(d.video.author),
    st.duration && fmtDuration(st.duration),
    `${st.words.toLocaleString()} words`,
    `${st.chunks} chunks · ${st.retrieved} retrieved`,
  ].filter(Boolean).map((t, i) => `<span style="--i:${i}">${t}</span>`);
  chips.push(`<span class="${d.generated_by_llm ? "gen" : ""}" style="--i:${chips.length}">${d.generated_by_llm ? "✦ " + esc(d.model) : "offline fallback"}</span>`);
  $("#meta").innerHTML = chips.join("");
  $("#fallback-warning").hidden = d.generated_by_llm;

  // Summary
  const sumEl = $("#summary-text");
  sumEl.innerHTML = linkTimes(d.summary || "No summary returned.");
  wordReveal(sumEl);
  $("#key-points").innerHTML = (d.key_points || []).map((p, i) =>
    `<li class="pop" style="--i:${i + 4}">${linkTimes(p)}</li>`).join("") || `<li class="muted">None returned.</li>`;

  // Timeline
  state.marks = (d.timestamps || []).map((t) => ({ start: toSeconds(t.time), topic: t.topic || "" }));
  $("#timeline").innerHTML = (d.timestamps || []).map((t, i) =>
    `<li class="pop" style="--i:${i}" data-t="${toSeconds(t.time)}"><div class="tl-card">${tsChip(t.time)}<span class="tl-topic">${esc(t.topic || "")}</span><span class="tl-bar"><i></i></span></div></li>`).join("")
    || `<li class="muted">None returned.</li>`;
  startTracking();

  // Mind map
  buildMindMap(d.mind_map || {}, d.video.title);

  // Evaluation
  $("#metrics").innerHTML = d.evaluation.map((m, i) => {
    const num = parseFloat(m.value);
    const isRatio = !isNaN(num) && num <= 1 && /\./.test(m.value) && !/x$/.test(m.value);
    const color = BRANCH_COLORS[i % BRANCH_COLORS.length];
    return `<div class="metric pop" style="--i:${i}">${isRatio ? `<div class="ring" style="--rc:${color}" data-p="${num * 100}"></div>` : ""}
      <div><div class="v" data-num="${isNaN(num) ? "" : num}" data-dec="${(m.value.split(".")[1] || "").replace(/\D/g, "").length}" data-suf="${esc(m.value.replace(/^[\d.]+/, ""))}">${esc(m.value)}</div><div class="l">${esc(m.label)}</div></div></div>`;
  }).join("");
  $("#rouge-note").textContent = d.rouge_computed ? "" : "ROUGE not computed — add a reference summary in settings to score it.";
  const s = d.safety || { counts: {} };
  const c = s.counts || {};
  const shieldOk = `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/><path d="M9 12l2 2 4-4"/></svg>`;
  const shieldBad = `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/><path d="M12 8v5M12 16h.01"/></svg>`;
  $("#safety").innerHTML = [
    ["Prompt injection", c.prompt_injection], ["Content safety", c.content_safety],
    ["Bias / fairness", c.bias_fairness], ["Weakly grounded", c.ungrounded_claims],
  ].map(([label, n], i) => `<div class="shield pop ${n ? "flag" : ""}" style="--i:${i + 3}"><span class="s-ico">${n ? shieldBad : shieldOk}</span><div><div class="v">${n || 0}</div><div class="l">${label}</div></div></div>`).join("");
  const findings = [...(s.prompt_injection || []), ...(s.content_safety || []), ...(s.bias_fairness || [])]
    .map((f) => `<li><code>${esc(f.label)}</code> ${esc(f.excerpt)}</li>`);
  const weak = (s.ungrounded_claims || []).map((x) => `<li>Weakly grounded: ${esc(x)}</li>`);
  $("#safety-detail").innerHTML = findings.length || weak.length ? `<ul>${findings.join("")}${weak.join("")}</ul>` : "";

  // Sources
  const maxScore = Math.max(...d.sources.map((x) => x.score), 0.001);
  $("#sources-note").textContent = d.sources.length
    ? `The summary was written from these ${d.sources.length} passages only.`
    : "No passages were retrieved for this summary.";
  // Course passages come from many videos; link out instead of seeking the embedded one.
  const srcChip = (x) => x.url
    ? `<a class="ts" href="${esc(x.url)}" target="_blank" rel="noopener" onclick="event.stopPropagation()">${esc(x.time)}</a>`
    : tsChip(x.time);
  $("#sources").innerHTML = d.sources.map((x, i) =>
    `<details class="source pop" style="--i:${i}"><summary>${srcChip(x)}<span>${x.video ? `V${x.video_index} · ${esc(x.video)}` : `Passage ${i + 1}`}</span>
      <span class="score"><i><b style="width:${(x.score / maxScore) * 100}%"></b></i>${x.score}</span>
      <svg class="chev" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M6 9l6 6 6-6"/></svg></summary><p>${esc(x.text)}</p></details>`).join("");

  selectTab("summary");
  renderChat();
}

function buildMindMap(mm, fallbackTitle) {
  const branches = (mm.branches || []).map((b, i) => ({
    name: typeof b === "string" ? b : b.name || `Branch ${i + 1}`,
    kids: typeof b === "string" ? [] : b.children || [],
  }));
  const rows = Math.max(1, branches.length);
  let html = `<div class="mm-root" style="grid-row:1 / span ${rows}">${esc(mm.root || fallbackTitle)}</div>`;
  branches.forEach((b, i) => {
    const color = BRANCH_COLORS[i % BRANCH_COLORS.length];
    const kids = b.kids.map((k, j) => {
      const m = String(k).match(new RegExp(`^(.*?)\\s*\\(?(${TIME})\\)?$`));
      const body = m ? `${esc(m[1])} ${tsChip(m[2])}` : esc(k);
      return `<div class="mm-kid pop" style="--i:${i * 2 + j + 6}" data-b="${i}">${body}</div>`;
    }).join("");
    html += `<div class="mm-branch pop" style="grid-row:${i + 1};--c:${color};--i:${i + 2}" data-b="${i}">${esc(b.name)}</div>`;
    html += `<div class="mm-kids" style="grid-row:${i + 1}">${kids}</div>`;
  });
  $("#mindmap").innerHTML = html;
  $("#mm-lines").innerHTML = "";
}

function drawMindMapLines() {
  const wrap = $(".mm-wrap");
  const svg = $("#mm-lines");
  if (!wrap || wrap.offsetParent === null || getComputedStyle(svg).display === "none") return;
  const base = wrap.getBoundingClientRect();
  const pt = (el, side) => {
    const r = el.getBoundingClientRect();
    return { x: (side === "r" ? r.right : r.left) - base.left + wrap.scrollLeft, y: r.top + r.height / 2 - base.top };
  };
  const curve = (a, b) => {
    const dx = (b.x - a.x) * 0.5;
    return `M${a.x},${a.y} C${a.x + dx},${a.y} ${b.x - dx},${b.y} ${b.x},${b.y}`;
  };
  const root = $(".mm-root");
  if (!root) return;
  svg.setAttribute("width", wrap.scrollWidth);
  svg.setAttribute("height", wrap.scrollHeight);
  const paths = [];
  $$(".mm-branch").forEach((br, i) => {
    const color = BRANCH_COLORS[i % BRANCH_COLORS.length];
    paths.push({ d: curve(pt(root, "r"), pt(br, "l")), color, delay: 150 + i * 90 });
    $$(`.mm-kid[data-b="${i}"]`).forEach((kid, j) =>
      paths.push({ d: curve(pt(br, "r"), pt(kid, "l")), color, delay: 450 + i * 90 + j * 50 }));
  });
  svg.innerHTML = paths.map((p) => `<path d="${p.d}" stroke="${p.color}" style="--delay:${p.delay}ms"/>`).join("");
  $$("path", svg).forEach((p) => p.style.setProperty("--len", p.getTotalLength()));
}
let resizeTimer;
addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => { moveInk(); drawMindMapLines(); }, 120); });

function animateMetrics() {
  $$("#metrics .v[data-num]").forEach((v) => {
    if (v.dataset.num === "") return;
    countUp(v, Number(v.dataset.num), Number(v.dataset.dec), v.dataset.suf);
  });
  $$("#metrics .ring").forEach((r) => { r.style.setProperty("--p", 0); void r.offsetWidth; });
  requestAnimationFrame(() => requestAnimationFrame(() =>
    $$("#metrics .ring").forEach((r) => r.style.setProperty("--p", r.dataset.p))));
}

function moveInk() {
  const active = $(".tab.active");
  const ink = $("#tab-ink");
  if (!active || !ink) return;
  ink.style.width = `${active.offsetWidth}px`;
  ink.style.transform = `translateX(${active.offsetLeft}px)`;
}

function selectTab(name) {
  $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  $$(".panel").forEach((p) => {
    const show = p.id === `tab-${name}`;
    if (show && p.hidden) { p.hidden = false; p.style.animation = "none"; void p.offsetWidth; p.style.animation = ""; restartPops(p); }
    else p.hidden = !show;
  });
  moveInk();
  if (name === "mindmap") requestAnimationFrame(() => setTimeout(drawMindMapLines, 60));
  if (name === "eval") animateMetrics();
}
function restartPops(root) {
  $$(".pop", root).forEach((el) => { el.style.animation = "none"; void el.offsetWidth; el.style.animation = ""; });
}
$$(".tab").forEach((t) => t.addEventListener("click", () => selectTab(t.dataset.tab)));

// Stars
$("#stars").innerHTML = [1, 2, 3, 4, 5].map((n) => `<button type="button" data-star="${n}" aria-label="${n} stars">★</button>`).join("");
const paintStars = () => $$("[data-star]").forEach((b) => b.classList.toggle("on", Number(b.dataset.star) <= state.rating));
$("#stars").addEventListener("click", (e) => {
  const b = e.target.closest("[data-star]");
  if (b) { state.rating = Number(b.dataset.star); paintStars(); }
});
paintStars();
$("#feedback").addEventListener("submit", async (e) => {
  e.preventDefault();
  try {
    await api("/api/feedback", { rating: state.rating, comment: $("#fb-comment").value });
    $("#fb-comment").value = "";
    toast(`Thanks — ${state.rating}★ feedback saved`);
  } catch (err) { toast(err.message); }
});

// --------------------------------------------------------------------- chat

const messages = $("#messages");

function renderChat() {
  messages.innerHTML = "";
  if (!state.chat.length) {
    messages.innerHTML = `<div class="empty-chat"><div class="orb"></div><strong>Ask me anything about this video.</strong><br>Answers come only from the transcript, with timestamps you can click.</div>`;
  }
  state.chat.forEach((m) => appendMessage(m, false));
  $("#suggestions").innerHTML = state.chat.length ? "" :
    SUGGESTIONS.map((s, i) => `<button class="chip pop" style="--i:${i + 2}" type="button">${esc(s)}</button>`).join("");
}

function appendMessage(m, animate = true) {
  $(".empty-chat", messages)?.remove();
  const el = document.createElement("div");
  if (m.role === "user") {
    el.className = "msg user";
    el.textContent = m.content;
  } else {
    el.className = `msg bot${m.error ? " err" : ""}`;
    el.innerHTML = renderMarkdown(m.content);
    if (animate) wordReveal(el);
    if ((m.sources || []).length) {
      const cites = document.createElement("div");
      cites.className = "cites";
      cites.innerHTML = `From ${m.sources.map((s) => tsChip(s.time)).join("")}`;
      el.append(cites);
    }
  }
  if (!animate) el.style.animation = "none";
  messages.append(el);
  messages.scrollTop = messages.scrollHeight;
  return el;
}

async function send(text) {
  const question = text.trim();
  if (!question || $(".send").disabled || !state.data) return;
  const history = state.chat.map(({ role, content }) => ({ role, content }));
  const userMsg = { role: "user", content: question };
  state.chat.push(userMsg);
  appendMessage(userMsg);
  $("#suggestions").innerHTML = "";
  input.value = "";
  autosize();
  const sendBtn = $(".send");
  sendBtn.disabled = true;
  sendBtn.classList.remove("fly"); void sendBtn.offsetWidth; sendBtn.classList.add("fly");
  $(".chat-avatar").classList.add("thinking");

  const typing = document.createElement("div");
  typing.className = "msg bot";
  typing.innerHTML = `<span class="typing"><i></i><i></i><i></i></span>`;
  messages.append(typing);
  messages.scrollTop = messages.scrollHeight;

  let reply;
  try {
    const r = await (await api("/api/ask", { question, history })).json();
    reply = { role: "assistant", content: r.answer, sources: r.sources };
  } catch (e) {
    reply = { role: "assistant", content: `Couldn't answer: ${e.message}`, error: true };
  }
  typing.remove();
  state.chat.push(reply);
  const el = appendMessage(reply);
  const words = $$(".w", el).length;
  setTimeout(() => (messages.scrollTop = messages.scrollHeight), Math.min(words, 120) * 22);
  sendBtn.disabled = false;
  $(".chat-avatar").classList.remove("thinking");
  input.focus();
}

const input = $("#chat-input");
function autosize() { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 140) + "px"; }
input.addEventListener("input", autosize);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(input.value); }
});
$("#chat-form").addEventListener("submit", (e) => { e.preventDefault(); send(input.value); });
$("#suggestions").addEventListener("click", (e) => { const c = e.target.closest(".chip"); if (c) send(c.textContent); });
$("#clear-chat").addEventListener("click", () => { state.chat = []; renderChat(); });

loadStatus();
