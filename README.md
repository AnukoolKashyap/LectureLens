# AI-Powered Video Summarization for Smart Learning

**21CSE306P — Applied Generative AI** · Industry Supported Professional Elective · B.Tech 5th Sem (2026–27)
Department of Networking and Communications, School of Computing, SRMIST

A retrieval-augmented generation pipeline that turns a YouTube video into a
grounded summary with key points, real timestamps and a mind map.

---

## Quick start

```bash
# 1. install
pip install -r requirements.txt

# 2. add your Mistral key
copy .env.example .env          # Windows   (cp on Linux/macOS)
#    then open .env and paste the key from https://console.mistral.ai

# 3. check everything before you need it to work
python doctor.py --fix

# 4. run the demo UI
streamlit run app.py
```

**Run `python doctor.py` the day before, not an hour before.** It checks the
Python version, every package, the API key, all three network endpoints, the
embedding model cache and a live Mistral round trip, and names the exact command
that fixes anything failing. `--fix` also pre-downloads the ~90 MB of model
weights so the first run is not waiting on a download.

It exits 0 only when the generative path is genuinely ready. Anything less and
the pipeline falls back to extractive output, which is clearly labelled and must
not be presented as a result.

### A whole course, not just one video

A YouTube course is usually a playlist. Paste the playlist link into the same
box, or several video links at once, and the whole set is treated as one course.

```bash
python cli.py --playlist "https://www.youtube.com/playlist?list=PL..."
python cli.py --urls "https://youtu.be/AAA" "https://youtu.be/BBB" "https://youtu.be/CCC"
python cli.py --playlist "...PL..." --where-is "gradient descent"
python cli.py --playlist "...PL..." --ask "which lesson should I watch first?"
```

What a course adds over running the videos one by one:

| | |
|---|---|
| **One shared index** | Every video is chunked into the same vector store, so retrieval and the chat cross video boundaries. |
| **Cross-video answers** | A question is answered from wherever the course covers it, capped at two chunks per video so one long lecture cannot swamp the answer. |
| **`--where-is`** | Lists every place a topic appears, with the video, the timestamp and a deep link. The same idea is usually taught more than once. |
| **A course guide** | Per-video summaries, plus themes across the course, how the videos build on each other, and a suggested watch order. |
| **Resilience** | A video with no captions is skipped and recorded, rather than failing the whole course. |

The per-video summaries are produced first and only those are synthesised into
the course guide — a map-reduce, because a full course transcript never fits in
one prompt. Cost therefore grows by one call per video plus a single call at the
end, rather than exploding with course length.

Playlist reading uses yt-dlp in flat mode: it reads the playlist index and
downloads no media. If yt-dlp is missing, pasting the individual video links
still works.

Try it offline on the bundled three-lesson sample:

```bash
python sample_data/make_course_sample.py      # writes the three transcripts
python cli.py --local-course sample_data/course_01_gradient_descent.json \
                             sample_data/course_02_backpropagation.json \
                             sample_data/course_03_learning_rate.json \
              --offline --where-is "learning rate"
```

Command line instead of the UI:

```bash
python cli.py --url "https://www.youtube.com/watch?v=VIDEO_ID"
python cli.py --url URL --profile tutorial
python cli.py --url URL --reference my_reference.txt     # computes ROUGE
python cli.py --url URL --ask "what is the learning rate?"
```

No network, no API key? The bundled sample still exercises stages 1–6:

```bash
python cli.py --local sample_data/sample_lecture.json --offline
python test_pipeline.py
```

---

## Architecture

The module names map one-to-one onto the numbered blocks of the pipeline
diagram, so the slide and the code cannot drift apart.

| # | Module | What it does |
|---|---|---|
| 1 | `stage1_ingest.py` | Validates the URL, fetches metadata via oEmbed, pulls the timed transcript through the YouTube Transcript API. Each segment keeps its start time. Also expands a playlist into its videos and ingests a whole course, skipping any video without captions. |
| 2 | `stage2_preprocess.py` | Strips caption artefacts and fillers, then splits into **1000-character chunks with 200 characters of overlap**, aligned to segment boundaries so each chunk carries a true start and end time. Every chunk also records which video it came from. |
| 3 | `stage3_embed.py` | Encodes chunks with **all-MiniLM-L6-v2** (encoder-only Transformer) and indexes them in **ChromaDB** with cosine distance. |
| 4 | `stage4_retrieve.py` | Pulls `fetch_k=12` candidates, re-ranks by vector similarity + lexical overlap + position, returns **top-k = 5**. Retrieval can be scoped to one video, or run across the course with a per-video cap. |
| 5 | `stage5_summarize.py` | Sends the grounded prompt to **Mistral** and parses the JSON response. Retries on 429/5xx. |
| 6 | `stage6_output.py` | Assembles key points, timestamps, mind map; exports Markdown, JSON and Mermaid. |

Cross-cutting layers, applied across stages 1–6:

| # | Module | What it does |
|---|---|---|
| 7 | `layer7_errors.py` | Typed exceptions with user-safe messages; URL and length validation. No stack trace ever reaches the UI. |
| 8 | `layer8_redteam.py` | Prompt-injection detection and neutralisation, content-safety screen, bias screen, grounding check. |
| 9 | `layer9_eval.py` | ROUGE-1/2/L, faithfulness, relevance, compression; reviewer feedback log. |

`pipeline.py` runs these as a fixed linear sequence — `run_pipeline()` for one
video, `run_course()` for a playlist. It is a pipeline, not an
autonomous system: nothing chooses its own steps, calls tools on its own
initiative, or loops back on itself.

---

## How hallucination is controlled

Three mechanisms, in order of how much work they do:

1. **Retrieval grounding.** The model only ever sees the retrieved chunks, and
   the prompt states that anything absent from them must be answered
   `not covered in this video`.
2. **Timestamp validation.** Stage 5 discards any timestamp the model returns
   that was not present in the retrieved context, so a timestamp on screen is
   always one a human can click and verify. The test suite asserts this.
3. **Grounding check.** Layer 8 measures how much of each generated sentence is
   lexically supported by the context and surfaces the weakly supported ones.
   This is a **proxy, not an entailment check**, and is labelled as such wherever
   it is displayed.

---

## Prompt engineering

All prompt text lives in `prompts.py`. Four techniques are applied:

| Technique | Purpose |
|---|---|
| Role prompting | Fixes voice and audience — note-taker, minute-taker, or technical writer |
| Grounding constraint | Forbids outside knowledge; mandates `not covered in this video` |
| Structured output | A JSON schema the parser can rely on |
| Few-shot exemplar | One worked key point fixing phrasing, length and tense |

Three use-case profiles select different variations of the same four techniques:

| Profile | Scenario | What changes |
|---|---|---|
| `lecture` | Recorded lecture → revision notes | Key points are concept statements with timestamps |
| `meeting` | Seminar/meeting → minutes | Decisions and action items separated, owners named |
| `tutorial` | Tutorial → ordered procedure | Numbered steps only, commentary excluded |

`NAIVE_PROMPT` holds the deliberately weak "Summarise this video" baseline used
for the before/after comparison.

---

## Evaluation

| Metric | Meaning | Needs |
|---|---|---|
| ROUGE-1 / 2 / L | n-gram and longest-common-subsequence overlap with a reference | A human-written reference summary |
| Faithfulness | Share of generated content words supported by retrieved context | Nothing extra |
| Relevance | Cosine similarity between summary and chunk centroid | The real embedding model |
| Compression | Transcript words ÷ summary words | Nothing extra |

**ROUGE needs a reference summary.** Without one the pipeline reports
"not computed" rather than printing a number, because a ROUGE score against no
reference is meaningless. Write a reference by hand, pass it with `--reference`,
and you get a real figure.

### Fallbacks and why they are labelled

Two fallbacks keep a demo alive when a download or a key is missing:

* **Hashing embedder** replaces all-MiniLM-L6-v2 — lexical, not semantic.
* **Extractive fallback** replaces Mistral — copies sentences, generates nothing.

When either is active, `evaluation.trustworthy` is `False`, the CLI prints a
warning, and the UI shows a red banner. Metrics measured in these modes describe
the fallback, not this system, and must not be reported as project results.

---

## Configuration

Every tunable is in `yt_summarizer/config.py`:

```python
chunk_size      = 1000    # characters
chunk_overlap   = 200     # characters
top_k           = 5       # chunks passed to the summariser
fetch_k         = 12      # candidates before re-ranking
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
llm_model       = "mistral-small-latest"
```

The Streamlit sidebar edits chunk size, overlap, top-k and re-ranking live,
which is useful for showing their effect during a demo.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| "No transcript could be retrieved" | Captions disabled, or the video is age-restricted or private | Try another video; most lecture uploads have auto-captions |
| Falls back to the hashing embedder | The model weights could not be downloaded | Check the network, then `python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"` to pre-download |
| "MISTRAL_API_KEY is not set" | No `.env`, or the file was not copied from `.env.example` | Create `.env` with the key, then restart Streamlit |
| `rouge-score` fails to build | Its build needs an older setuptools on some Python versions | Safe to ignore — it lives in `requirements-optional.txt` precisely so it cannot abort the main install, and `layer9_eval.py` has a built-in ROUGE it switches to automatically |
| ChromaDB will not open | Corrupt store from an interrupted run | Delete `data/chroma/` and rerun |
| Everything installs but nothing reaches the network | Campus Wi-Fi or a corporate proxy blocking youtube.com, huggingface.co or api.mistral.ai | `python doctor.py` tells you which of the three is blocked. A phone hotspot is the usual fix. Download the model weights once with `--fix` while you have a working connection and stage 3 stops needing the network. |

---

## Layout

```
.
├── server.py               FastAPI app — the web UI (python server.py)
├── run_server.bat          Windows launcher for the above
├── cli.py                  command-line runner, incl. --playlist / --local-course
├── app.py                  older Streamlit UI, kept as a fallback
├── doctor.py               preflight check — run this first
├── test_pipeline.py        24 smoke tests, no pytest needed
├── requirements.txt        everything required
├── requirements-optional.txt   extras that may fail to build; safe to skip
├── .env.example            copy to .env and add your Mistral key
│
├── yt_summarizer/          the pipeline itself
│   ├── config.py           every tunable in one place
│   ├── prompts.py          all prompt text in one place
│   ├── pipeline.py         stage sequence; run_video() and run_course()
│   ├── stage1_ingest.py    transcripts, playlist expansion
│   ├── stage2_preprocess.py   cleaning and chunking
│   ├── stage3_embed.py     embeddings + ChromaDB
│   ├── stage4_retrieve.py  retrieval, re-ranking, cross-video search
│   ├── stage5_summarize.py generation and course synthesis
│   ├── stage6_output.py    markdown / JSON / mind-map export
│   └── layer7_errors.py  layer8_redteam.py  layer9_eval.py
│
├── web/                    front end served by server.py
│   ├── index.html
│   ├── app.js
│   └── style.css
│
├── sample_data/            offline samples: one lecture, plus a 3-video course
├── docs/                   poster, review deck, screenshots
│
├── data/                   vector store        (generated, git-ignored)
└── exports/                saved summaries     (generated, git-ignored)
```

## Tests

```bash
python test_pipeline.py
```

14 tests covering URL parsing, chunk overlap and timestamp integrity, injection
detection and neutralisation, the grounding check, each evaluation metric, and a
full offline run of all six stages. They run without network or an API key.
