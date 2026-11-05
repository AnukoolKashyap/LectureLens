"""Layer 8 — quality and safety, red teaming.

Three adversarial screens run across stages 1-6:

  * prompt injection — a transcript is untrusted input. A video whose narrator
    says "ignore your previous instructions" must not be able to steer the
    summariser, so injection candidates are detected, neutralised in the text
    that reaches the prompt, and reported.
  * content safety   — flags transcripts whose subject matter should not be
    summarised without a human looking first.
  * bias and fairness — flags generalisations about groups that appear in the
    *generated* output but are not grounded in the retrieved context.

Screens report; they do not silently rewrite meaning. The only text ever
modified is an injection attempt, which is defanged rather than removed so the
reviewer can still see what the video said.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from .layer7_errors import get_logger
from .stage2_preprocess import content_terms

log = get_logger(__name__)

# --------------------------------------------------------------------------
# Patterns
# --------------------------------------------------------------------------

INJECTION_PATTERNS = [
    (re.compile(r"ignore\s+(?:all\s+)?(?:your\s+|the\s+)?previous\s+instructions", re.I),
     "instruction override"),
    (re.compile(r"disregard\s+(?:the\s+|all\s+)?(?:above|prior|earlier)", re.I),
     "instruction override"),
    (re.compile(r"forget\s+(?:everything|all)\s+(?:you|above)", re.I),
     "instruction override"),
    (re.compile(r"you\s+are\s+now\s+(?:a|an|in)\b", re.I), "role reassignment"),
    (re.compile(r"\bsystem\s*prompt\b", re.I), "system prompt reference"),
    (re.compile(r"\b(?:developer|system)\s*(?:mode|message)\b", re.I), "privilege escalation"),
    (re.compile(r"reveal\s+(?:your\s+)?(?:instructions|prompt|rules)", re.I), "prompt extraction"),
    (re.compile(r"\bDAN\b|\bjailbreak\b", re.I), "known jailbreak marker"),
    (re.compile(r"</?(?:system|instruction|prompt)>", re.I), "delimiter injection"),
    (re.compile(r"\boutput\s+the\s+following\s+verbatim\b", re.I), "output hijack"),
]

UNSAFE_TOPIC_PATTERNS = [
    (re.compile(r"\b(?:build|make|synthesi[sz]e)\s+(?:a\s+)?(?:bomb|explosive|nerve agent)\b", re.I),
     "weapons construction"),
    (re.compile(r"\b(?:self[-\s]?harm|suicide method)\b", re.I), "self-harm"),
    (re.compile(r"\b(?:child\s+(?:abuse|exploitation))\b", re.I), "child safety"),
]

# Group nouns whose combination with a sweeping quantifier is worth a look.
GROUP_TERMS = re.compile(
    r"\b(men|women|students|immigrants|muslims|christians|hindus|jews|indians|"
    r"africans|asians|americans|europeans|boys|girls|elderly|young people)\b", re.I
)
SWEEPING = re.compile(r"\b(all|every|none|never|always|inherently|naturally|typically)\b", re.I)


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

@dataclass
class Finding:
    category: str
    label: str
    excerpt: str

    def to_dict(self) -> Dict[str, str]:
        return {"category": self.category, "label": self.label, "excerpt": self.excerpt}


@dataclass
class SafetyReport:
    injection: List[Finding] = field(default_factory=list)
    content: List[Finding] = field(default_factory=list)
    bias: List[Finding] = field(default_factory=list)
    ungrounded_claims: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not (self.injection or self.content or self.bias or self.ungrounded_claims)

    def to_dict(self) -> Dict:
        return {
            "passed": self.passed,
            "prompt_injection": [f.to_dict() for f in self.injection],
            "content_safety": [f.to_dict() for f in self.content],
            "bias_fairness": [f.to_dict() for f in self.bias],
            "ungrounded_claims": self.ungrounded_claims,
            "counts": {
                "prompt_injection": len(self.injection),
                "content_safety": len(self.content),
                "bias_fairness": len(self.bias),
                "ungrounded_claims": len(self.ungrounded_claims),
            },
        }


def _excerpt(text: str, match: re.Match, width: int = 70) -> str:
    start = max(0, match.start() - width // 2)
    end = min(len(text), match.end() + width // 2)
    return ("..." if start else "") + text[start:end].replace("\n", " ") + ("..." if end < len(text) else "")


# --------------------------------------------------------------------------
# Screens
# --------------------------------------------------------------------------

def scan_prompt_injection(text: str) -> List[Finding]:
    findings = []
    for pattern, label in INJECTION_PATTERNS:
        for match in pattern.finditer(text):
            findings.append(Finding("prompt_injection", label, _excerpt(text, match)))
    return findings


def neutralise_injection(text: str) -> str:
    """Defang injection attempts without deleting what the speaker said.

    The imperative is wrapped in a marker so the model reads it as reported
    speech inside the transcript rather than as an instruction addressed to it.
    """
    result = text
    for pattern, _ in INJECTION_PATTERNS:
        result = pattern.sub(lambda m: "[quoted transcript text: %s]" % m.group(0), result)
    return result


def scan_content_safety(text: str) -> List[Finding]:
    findings = []
    for pattern, label in UNSAFE_TOPIC_PATTERNS:
        match = pattern.search(text)
        if match:
            findings.append(Finding("content_safety", label, _excerpt(text, match)))
    return findings


def scan_bias(generated_text: str, context: str) -> List[Finding]:
    """Flag sweeping statements about groups that the context does not support."""
    findings = []
    context_lower = context.lower()
    for sentence in re.split(r"(?<=[.!?])\s+", generated_text):
        group = GROUP_TERMS.search(sentence)
        if not group or not SWEEPING.search(sentence):
            continue
        # If the same group word plus a sweeping quantifier appears in the source,
        # the generation is reporting rather than inventing.
        if group.group(0).lower() in context_lower and SWEEPING.search(context_lower):
            continue
        findings.append(Finding("bias_fairness", "unsupported generalisation", sentence.strip()))
    return findings


def find_ungrounded_claims(
    generated_sentences: Sequence[str], context: str, threshold: float = 0.45
) -> List[str]:
    """Return generated sentences whose content words are largely absent from the context.

    This is a lexical grounding proxy, not an entailment check. It catches the
    common failure — a fluent sentence about something the video never mentioned
    — and is reported as a proxy wherever it is displayed.
    """
    context_terms = set(content_terms(context))
    if not context_terms:
        return []

    ungrounded = []
    for sentence in generated_sentences:
        terms = content_terms(sentence)
        if len(terms) < 4:
            continue
        supported = sum(1 for t in terms if t in context_terms) / len(terms)
        if supported < threshold:
            ungrounded.append(sentence.strip())
    return ungrounded


def screen_input(transcript: str) -> SafetyReport:
    report = SafetyReport(
        injection=scan_prompt_injection(transcript),
        content=scan_content_safety(transcript),
    )
    if report.injection:
        log.warning("Prompt injection screen raised %d finding(s)", len(report.injection))
    if report.content:
        log.warning("Content safety screen raised %d finding(s)", len(report.content))
    return report


def screen_output(summary_text: str, key_points: Sequence[str], context: str) -> SafetyReport:
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", summary_text) if s.strip()]
    sentences += list(key_points)
    report = SafetyReport(
        bias=scan_bias(" ".join(sentences), context),
        ungrounded_claims=find_ungrounded_claims(sentences, context),
    )
    if report.bias:
        log.warning("Bias screen raised %d finding(s)", len(report.bias))
    if report.ungrounded_claims:
        log.warning("%d generated sentence(s) are weakly grounded", len(report.ungrounded_claims))
    return report


def merge(*reports: SafetyReport) -> SafetyReport:
    merged = SafetyReport()
    for report in reports:
        merged.injection.extend(report.injection)
        merged.content.extend(report.content)
        merged.bias.extend(report.bias)
        merged.ungrounded_claims.extend(report.ungrounded_claims)
    return merged
