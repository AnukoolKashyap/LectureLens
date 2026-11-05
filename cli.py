#!/usr/bin/env python3
"""Command-line runner.

    python cli.py --url "https://www.youtube.com/watch?v=VIDEO_ID"
    python cli.py --local sample_data/sample_lecture.json --offline
    python cli.py --url URL --profile tutorial --reference reference.txt
    python cli.py --url URL --ask "what is the learning rate?"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from yt_summarizer.config import settings
from yt_summarizer.layer7_errors import PipelineError
from yt_summarizer.pipeline import run_pipeline
from yt_summarizer.stage6_output import to_markdown

BAR = "=" * 78


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="AI-Powered Video Summarization for Smarter Learning",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="YouTube video URL or 11-character video id")
    source.add_argument("--local", help="Path to a local transcript (.json or .txt)")
    source.add_argument("--playlist", help="YouTube playlist URL or id — summarises every video")
    source.add_argument("--urls", nargs="+", metavar="URL",
                        help="Several video URLs, treated as one course")
    source.add_argument("--local-course", nargs="+", metavar="PATH",
                        help="Several local transcripts, treated as one course")

    parser.add_argument(
        "--profile", default="lecture", choices=["lecture", "meeting", "tutorial"],
        help="Use-case profile that selects the prompt variation (default: lecture)",
    )
    parser.add_argument("--reference", help="Path to a human-written reference summary, for ROUGE")
    parser.add_argument("--ask", help="Ask one question after summarising")
    parser.add_argument("--where-is", metavar="TOPIC",
                        help="Course mode: list every place a topic is covered")
    parser.add_argument("--offline", action="store_true", help="Force the offline fallbacks")
    parser.add_argument("--no-export", action="store_true", help="Skip writing export files")
    parser.add_argument("--json", action="store_true", help="Print the result as JSON")
    parser.add_argument("--quiet", action="store_true", help="Suppress stage progress output")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.offline:
        settings.offline_mode = True

    reference = None
    if args.reference:
        reference_path = Path(args.reference)
        if not reference_path.exists():
            print("Reference file not found: %s" % reference_path, file=sys.stderr)
            return 2
        reference = reference_path.read_text(encoding="utf-8")

    def progress(step: int, label: str) -> None:
        if not args.quiet:
            print("  [%d/6] %s..." % (step, label))

    if not args.quiet:
        print(BAR)
        print("AI-Powered Video Summarization for Smarter Learning")
        print(BAR)

    # ---- course mode -------------------------------------------------
    if args.playlist or args.urls or args.local_course:
        return run_course_cli(args)

    try:
        run = run_pipeline(
            url=args.url or "",
            local_path=Path(args.local) if args.local else None,
            profile_key=args.profile,
            reference_summary=reference,
            do_export=not args.no_export,
            progress=progress,
        )
    except PipelineError as exc:
        print("\nError: %s" % exc.user_message, file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(run.summary.to_dict(), indent=2, ensure_ascii=False))
        return 0

    print()
    print(to_markdown(run.summary))

    diag = run.summary.diagnostics
    print(BAR)
    print("Pipeline diagnostics")
    print(BAR)
    print("  chunks indexed     : %d (size %d, overlap %d)" % (
        diag["chunk_count"], diag["chunk_size"], diag["chunk_overlap"]))
    print("  chunks retrieved   : %d (top_k=%d, rerank=%s)" % (
        diag["retrieved_count"], diag["top_k"], diag["rerank"]))
    print("  embedding model    : %s%s" % (
        diag["embedding_model"], "  [DEGRADED FALLBACK]" if diag["embedder_degraded"] else ""))
    print("  vector store       : %s" % diag["vector_store"])
    print("  summarisation model: %s" % run.summary.model)
    print("  stage times (s)    : %s" % diag["stage_times"])

    safety = run.summary.safety
    print("  safety screens     : injection=%d content=%d bias=%d ungrounded=%d" % (
        safety["counts"]["prompt_injection"],
        safety["counts"]["content_safety"],
        safety["counts"]["bias_fairness"],
        safety["counts"]["ungrounded_claims"],
    ))

    if not run.evaluation.trustworthy:
        print()
        print("  !! These metrics are NOT representative of the real system:")
        for note in run.evaluation.notes:
            print("     - %s" % note)

    if run.exports:
        print()
        print("  exported:")
        for kind, path in run.exports.items():
            print("     %-8s %s" % (kind, path))

    if args.ask:
        print()
        print(BAR)
        print("Q: %s" % args.ask)
        print("A: %s" % run.ask(args.ask, args.profile))

    return 0


def run_course_cli(args) -> int:
    """Summarise a playlist or a set of videos as one course."""
    from yt_summarizer.pipeline import run_course
    from yt_summarizer.stage6_output import course_to_markdown

    def progress(step: int, label: str) -> None:
        if not args.quiet:
            print("  [%d/6] %s..." % (step, label))

    sources = args.playlist or args.urls or ""
    try:
        run = run_course(
            sources=sources,
            profile_key=args.profile,
            do_export=not args.no_export,
            progress=progress,
            local_paths=args.local_course,
        )
    except PipelineError as exc:
        print("\nError: %s" % exc.user_message, file=sys.stderr)
        return 1

    report = run.report
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        return 0

    print()
    print(course_to_markdown(report))

    diag = report.diagnostics
    print(BAR)
    print("Course diagnostics")
    print(BAR)
    print("  videos             : %d usable, %d skipped" % (
        diag["video_count"], diag["skipped_count"]))
    print("  chunks indexed     : %d (size %d, overlap %d)" % (
        diag["chunk_count"], diag["chunk_size"], diag["chunk_overlap"]))
    print("  embedding model    : %s%s" % (
        diag["embedding_model"],
        "  [DEGRADED FALLBACK]" if diag["embedder_degraded"] else ""))
    print("  vector store       : %s" % diag["vector_store"])
    print("  summarisation model: %s" % report.model)
    print("  stage times (s)    : %s" % diag["stage_times"])
    print("  total seconds      : %s" % diag["total_seconds"])
    print("  safety screens     : %s" % diag["safety"])

    if not report.generated:
        print()
        print("  !! The course guide was stitched offline, not generated.")

    if run.exports:
        print()
        print("  exported:")
        for kind, path in run.exports.items():
            print("     %-8s %s" % (kind, path))

    if args.where_is:
        print()
        print(BAR)
        print("Where is %r covered?" % args.where_is)
        print(BAR)
        for hit in run.where_is(args.where_is):
            print("  V%d %-44s %s  (score %.2f)" % (
                hit["video_index"], hit["video_title"][:44],
                hit["timestamp"], hit["score"]))
            print("      %s" % hit["preview"][:110].replace("\n", " "))

    if args.ask:
        print()
        print(BAR)
        print("Q: %s" % args.ask)
        print("A: %s" % run.ask(args.ask, args.profile))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
