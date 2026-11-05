#!/usr/bin/env python3
"""Preflight check. Run this before the review, not during it.

    python doctor.py              check everything
    python doctor.py --fix        also pre-download the embedding model weights
    python doctor.py --video URL  additionally test a real transcript fetch

Every check prints OK, WARN or FAIL. A FAIL names the exact command that fixes
it. Exit code is 0 when the generative path is ready, 1 when something would
force a fallback.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import socket
import sys
import time
import urllib.error
import urllib.request

OK, WARN, FAIL = "OK  ", "WARN", "FAIL"
RESULTS = []
BAR = "-" * 74


def record(status: str, name: str, detail: str = "", fix: str = "") -> None:
    RESULTS.append((status, name, detail, fix))
    line = "  [%s] %s" % (status, name)
    if detail:
        line += " — " + detail
    print(line)
    if fix and status != OK:
        print("         fix: %s" % fix)


def section(title: str) -> None:
    print("\n" + title)
    print(BAR)


# --------------------------------------------------------------------------

def check_python() -> None:
    section("Interpreter")
    version = sys.version_info
    detail = "%d.%d.%d on %s" % (version.major, version.minor, version.micro, platform.system())
    if version < (3, 9):
        record(FAIL, "Python version", detail, "Install Python 3.10 or newer.")
    elif version >= (3, 13):
        record(
            WARN, "Python version", detail,
            "3.13 is new enough that some wheels (torch, rouge-score) may not exist "
            "yet. 3.10 or 3.11 is the safer choice for a demo.",
        )
    else:
        record(OK, "Python version", detail)


def check_packages() -> None:
    section("Packages")
    required = [
        ("youtube_transcript_api", "youtube-transcript-api", "Stage 1 cannot fetch transcripts"),
        ("sentence_transformers", "sentence-transformers", "Stage 3 falls back to lexical hashing"),
        ("chromadb", "chromadb", "Stage 3 falls back to an in-memory store"),
        ("streamlit", "streamlit", "The demo UI will not start"),
    ]
    for module, package, consequence in required:
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, "__version__", "")
            record(OK, package, str(version))
        except ImportError:
            record(FAIL, package, consequence, "pip install %s" % package)

    try:
        importlib.import_module("dotenv")
        record(OK, "python-dotenv")
    except ImportError:
        record(
            WARN, "python-dotenv", ".env will not be loaded automatically",
            "pip install python-dotenv  (or set MISTRAL_API_KEY in your shell)",
        )

    try:
        importlib.import_module("rouge_score")
        record(OK, "rouge-score", "reference ROUGE implementation available")
    except ImportError:
        record(
            WARN, "rouge-score", "built-in ROUGE will be used instead (no stemming)",
            "pip install -r requirements-optional.txt  — safe to skip entirely",
        )


def check_api_key() -> bool:
    section("Mistral API key")
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    key = os.getenv("MISTRAL_API_KEY", "").strip()
    if not key:
        record(
            FAIL, "MISTRAL_API_KEY", "not set — stage 5 will run the extractive fallback",
            "copy .env.example .env   then paste your key from https://console.mistral.ai",
        )
        return False
    if len(key) < 20:
        record(FAIL, "MISTRAL_API_KEY", "set but implausibly short (%d chars)" % len(key),
               "Check you pasted the whole key.")
        return False
    record(OK, "MISTRAL_API_KEY", "set (%d chars, ends ...%s)" % (len(key), key[-4:]))
    return True


def _probe(url: str, timeout: int = 10):
    """Return (reachable, detail).

    A real HTTPS request, not a TCP connect. An intercepting proxy will happily
    complete a TCP handshake on port 443 for a host it then refuses to serve, so
    a socket check reports success for hosts that do not actually work — exactly
    the false reassurance a preflight tool must never give.

    Any HTTP status counts as reachable, including 401 and 403: the server
    answered, which is what is being tested. Only transport failures count as
    unreachable.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "doctor.py/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return True, "HTTP %d" % response.status
    except urllib.error.HTTPError as exc:
        return True, "HTTP %d (server answered)" % exc.code
    except urllib.error.URLError as exc:
        return False, str(getattr(exc, "reason", exc))[:80]
    except socket.timeout:
        return False, "timed out after %ds" % timeout
    except Exception as exc:
        return False, str(exc)[:80]


def check_network() -> None:
    section("Network")
    targets = [
        ("https://www.youtube.com/robots.txt",
         "www.youtube.com", "Stage 1 transcript fetch"),
        ("https://huggingface.co/api/models/sentence-transformers/all-MiniLM-L6-v2",
         "huggingface.co", "Stage 3 embedding model download"),
        ("https://api.mistral.ai/v1/models",
         "api.mistral.ai", "Stage 5 summarisation"),
    ]
    for url, host, purpose in targets:
        reachable, detail = _probe(url)
        if reachable:
            record(OK, host, "%s — %s" % (detail, purpose))
        else:
            record(
                FAIL, host, "unreachable (%s) — %s will fail" % (detail, purpose),
                "Campus Wi-Fi and corporate proxies commonly block these. Try a phone "
                "hotspot, and confirm with: curl -I %s" % url,
            )


def check_embedding_model(download: bool = False) -> None:
    section("Embedding model")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        record(FAIL, "all-MiniLM-L6-v2", "sentence-transformers not installed",
               "pip install sentence-transformers")
        return

    if not download:
        try:
            from huggingface_hub import try_to_load_from_cache

            cached = try_to_load_from_cache(
                "sentence-transformers/all-MiniLM-L6-v2", "config.json"
            )
            if cached:
                record(OK, "all-MiniLM-L6-v2", "already cached locally")
                return
        except Exception:
            pass
        record(
            WARN, "all-MiniLM-L6-v2", "not confirmed in the local cache",
            "python doctor.py --fix   downloads it now (~90 MB) so the demo does not "
            "wait on it",
        )
        return

    print("         downloading (~90 MB, one time)...")
    started = time.time()
    try:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        vector = model.encode(["preflight check"], normalize_embeddings=True)[0]
        record(
            OK, "all-MiniLM-L6-v2",
            "loaded in %.1fs, %d dimensions" % (time.time() - started, len(vector)),
        )
    except Exception as exc:
        record(FAIL, "all-MiniLM-L6-v2", str(exc)[:120],
               "Check the huggingface.co result above; retry on a different network.")


def check_mistral(has_key: bool) -> None:
    section("Mistral round trip")
    if not has_key:
        record(WARN, "live call", "skipped, no API key set")
        return

    from yt_summarizer.config import settings

    payload = json.dumps({
        "model": settings.llm_model,
        "messages": [{"role": "user", "content": "Reply with the single word: ready"}],
        "max_tokens": 5,
        "temperature": 0,
    }).encode("utf-8")

    request = urllib.request.Request(
        settings.llm_endpoint,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer %s" % settings.mistral_api_key,
        },
        method="POST",
    )
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))
        reply = body["choices"][0]["message"]["content"].strip()
        record(OK, settings.llm_model,
               "replied %r in %.1fs" % (reply[:20], time.time() - started))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:150]
        hint = {
            401: "The key is wrong or revoked. Generate a new one.",
            402: "Billing or free-tier quota issue on your Mistral account.",
            429: "Rate limited. Wait a minute and retry.",
        }.get(exc.code, "Check the model name in config.py.")
        record(FAIL, "live call", "HTTP %s: %s" % (exc.code, detail), hint)
    except Exception as exc:
        record(FAIL, "live call", str(exc)[:120], "Check the api.mistral.ai result above.")


def check_transcript(url: str) -> None:
    section("Transcript fetch")
    from yt_summarizer.layer7_errors import PipelineError
    from yt_summarizer.stage1_ingest import ingest

    try:
        doc = ingest(url)
        record(OK, "transcript", "%d segments, %d words, title %r" % (
            len(doc.segments), len(doc.full_text.split()), doc.title[:40]))
    except PipelineError as exc:
        record(FAIL, "transcript", exc.user_message,
               "Try another video — most lecture uploads have auto-captions.")


def check_pipeline_offline() -> None:
    section("Offline pipeline self-test")
    from yt_summarizer.config import settings
    from yt_summarizer.pipeline import run_pipeline
    from yt_summarizer.stage1_ingest import default_sample

    previous = settings.offline_mode
    settings.offline_mode = True
    try:
        run = run_pipeline(local_path=default_sample(), do_export=False)
        record(OK, "stages 1-6", "%d chunks indexed, %d retrieved, %d key points" % (
            len(run.chunks), len(run.retrieved), len(run.summary.key_points)))
        injections = run.summary.safety["counts"]["prompt_injection"]
        if injections:
            record(OK, "layer 8 injection screen", "caught %d planted attempt(s)" % injections)
        else:
            record(FAIL, "layer 8 injection screen", "missed the planted attempt",
                   "The sample should trigger it. Check layer8_redteam.py.")
    except Exception as exc:
        record(FAIL, "stages 1-6", str(exc)[:150], "Run: python test_pipeline.py")
    finally:
        settings.offline_mode = previous


# --------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Preflight check for the summarisation pipeline")
    parser.add_argument("--fix", action="store_true",
                        help="Pre-download the embedding model weights")
    parser.add_argument("--video", help="Also test fetching a real video transcript")
    args = parser.parse_args(argv)

    print("=" * 74)
    print("AI-Powered Video Summarization — preflight check")
    print("=" * 74)

    check_python()
    check_packages()
    has_key = check_api_key()
    check_network()
    check_embedding_model(download=args.fix)
    check_mistral(has_key)
    if args.video:
        check_transcript(args.video)
    check_pipeline_offline()

    fails = [r for r in RESULTS if r[0] == FAIL]
    warns = [r for r in RESULTS if r[0] == WARN]

    print("\n" + "=" * 74)
    print("%d OK, %d warnings, %d failures" % (
        len(RESULTS) - len(fails) - len(warns), len(warns), len(fails)))
    print("=" * 74)

    if fails:
        print("\nThe generative path is NOT ready. Fix these first:")
        for _, name, detail, fix in fails:
            print("  - %s: %s" % (name, detail or "failed"))
            if fix:
                print("      %s" % fix)
        print(
            "\nUntil these pass, the pipeline runs its fallbacks. That output is "
            "extractive, not generated,\nand every metric it produces is marked "
            "untrustworthy. Do not present it as a result."
        )
        return 1

    print("\nReady. The generative path will run end to end.")
    if warns:
        print("Warnings above are non-blocking.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
