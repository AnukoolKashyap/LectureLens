"""Prompt templates — the four techniques presented on Review 2, slide 4.

1. Role prompting        — fixes the voice and the audience
2. Grounding constraint  — the hallucination control
3. Structured output     — a JSON schema the parser can rely on
4. Few-shot exemplar     — one worked example fixing phrasing, length and tense

`USE_CASE_PROFILES` holds the three scenario variations from the same slide.
Edit the text here and the slide, the UI and the pipeline all stay in step.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

# --------------------------------------------------------------------------
# 1. Role prompting
# --------------------------------------------------------------------------

ROLE_LECTURE = (
    "You are an experienced lecture note-taker preparing revision notes for an "
    "undergraduate student."
)
ROLE_MEETING = (
    "You are an experienced minute-taker preparing a record of a recorded seminar "
    "or meeting."
)
ROLE_TUTORIAL = (
    "You are a technical writer converting a recorded tutorial into a procedure "
    "another person can follow without watching the video."
)

# --------------------------------------------------------------------------
# 2. Grounding constraint
# --------------------------------------------------------------------------

GROUNDING = (
    "Use only the transcript context provided below. Do not add outside knowledge, "
    "do not infer facts that are not stated, and do not guess. If the answer is not "
    "present in the context, reply exactly: not covered in this video. "
    "Every timestamp you output must be copied from a context block, never invented."
)

# --------------------------------------------------------------------------
# 3. Structured output
# --------------------------------------------------------------------------

SCHEMA = """Return a single JSON object and nothing else. Use exactly this schema:
{
  "summary": "<4-6 sentence abstractive summary of the whole video>",
  "key_points": ["<point>", "..."],
  "timestamps": [{"time": "<mm:ss copied from a context block>", "topic": "<short label>"}],
  "mind_map": {"root": "<video topic>", "branches": [{"name": "<branch>", "children": ["<leaf>"]}]}
}
Produce between 5 and 8 key points, between 4 and 8 timestamps, and between 3 and
5 mind-map branches. Do not wrap the JSON in code fences or commentary."""

# --------------------------------------------------------------------------
# 4. Few-shot exemplar
# --------------------------------------------------------------------------

FEW_SHOT = """Example of the required style for a single key point, so that
phrasing, length and tense stay consistent:

  Context block: "[2] (4:35) so gradient descent just walks downhill on the loss
  surface, and the step size is what we call the learning rate"

  Correct key point: "Gradient descent minimises the loss by stepping downhill
  on the loss surface, with step size set by the learning rate (4:35)."

Note the form: one sentence, present tense, the mechanism stated rather than
restated verbatim, and the timestamp in brackets at the end."""


SUMMARY_PROMPT = """{role}

{grounding}

{schema}

{few_shot}

TRANSCRIPT CONTEXT
------------------
{context}

Produce the JSON object now."""


QA_PROMPT = """{role}

{grounding}

Answer the student's question in no more than four sentences. End your answer
with the timestamp of the context block you relied on, in brackets.

TRANSCRIPT CONTEXT
------------------
{context}

QUESTION: {question}

ANSWER:"""


@dataclass
class UseCaseProfile:
    key: str
    label: str
    role: str
    extra_instruction: str


USE_CASE_PROFILES: Dict[str, UseCaseProfile] = {
    "lecture": UseCaseProfile(
        key="lecture",
        label="Recorded lecture to revision notes",
        role=ROLE_LECTURE,
        extra_instruction=(
            "Key points must be revision-ready statements of concepts, each carrying "
            "the timestamp where the concept is introduced."
        ),
    ),
    "meeting": UseCaseProfile(
        key="meeting",
        label="Seminar or meeting to minutes",
        role=ROLE_MEETING,
        extra_instruction=(
            "Separate decisions from action items. A key point that records a decision "
            "must begin with 'Decision:'; an action item must begin with 'Action:' and "
            "name the owner if the transcript states one."
        ),
    ),
    "tutorial": UseCaseProfile(
        key="tutorial",
        label="Tutorial to ordered procedure",
        role=ROLE_TUTORIAL,
        extra_instruction=(
            "Key points must form a numbered procedure in the order performed. Exclude "
            "narrative commentary, asides and anything not an instruction."
        ),
    ),
}


def build_summary_prompt(context: str, profile_key: str = "lecture") -> str:
    profile = USE_CASE_PROFILES.get(profile_key, USE_CASE_PROFILES["lecture"])
    return SUMMARY_PROMPT.format(
        role=profile.role + " " + profile.extra_instruction,
        grounding=GROUNDING,
        schema=SCHEMA,
        few_shot=FEW_SHOT,
        context=context,
    )


def build_qa_prompt(context: str, question: str, profile_key: str = "lecture") -> str:
    profile = USE_CASE_PROFILES.get(profile_key, USE_CASE_PROFILES["lecture"])
    return QA_PROMPT.format(
        role=profile.role,
        grounding=GROUNDING,
        context=context,
        question=question,
    )


# The deliberately weak prompt used for the before/after comparison on slide 4.
NAIVE_PROMPT = "Summarise this video.\n\n{context}"


# --------------------------------------------------------------------------
# Courses: synthesising across several videos
# --------------------------------------------------------------------------

ROLE_COURSE = (
    "You are an experienced teaching assistant writing a study guide for a "
    "course made up of several video lectures."
)

COURSE_GROUNDING = (
    "Work only from the per-video summaries given below. Do not add outside "
    "knowledge and do not invent topics that none of the summaries mention. "
    "When you refer to a video, use the number it is given here (V1, V2, ...). "
    "If the videos do not cover something, leave it out rather than guessing."
)

COURSE_SCHEMA = """Return a single JSON object and nothing else, in this schema:
{
  "overview": "<4-6 sentences describing what the course as a whole teaches>",
  "themes": [
    {"name": "<theme>",
     "videos": [1, 3],
     "detail": "<1-2 sentences on how the course develops this theme>"}
  ],
  "progression": "<2-4 sentences on how the videos build on each other>",
  "study_order": [
    {"video": 1, "why": "<short reason to watch this one at this point>"}
  ],
  "key_takeaways": ["<takeaway>", "..."]
}
Produce 3 to 6 themes, 4 to 8 key takeaways, and a study_order covering every
video exactly once. Do not wrap the JSON in code fences or commentary."""

COURSE_PROMPT = """{role}

{grounding}

{schema}

PER-VIDEO SUMMARIES
-------------------
{summaries}

Produce the JSON object now."""


def build_course_prompt(summaries_block: str) -> str:
    return COURSE_PROMPT.format(
        role=ROLE_COURSE,
        grounding=COURSE_GROUNDING,
        schema=COURSE_SCHEMA,
        summaries=summaries_block,
    )
