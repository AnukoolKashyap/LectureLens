"""AI-Powered Video Summarization for Smarter Learning.

21CSE306P Applied Generative AI — course project.

Stages 1-6 form the main flow; layers 7-9 are applied across them:

    1  stage1_ingest      ingestion and transcript extraction
    2  stage2_preprocess  cleaning and overlapping chunking
    3  stage3_embed       all-MiniLM-L6-v2 embeddings indexed in ChromaDB
    4  stage4_retrieve    top-k retrieval with re-ranking
    5  stage5_summarize   grounded summarisation with Mistral
    6  stage6_output      key points, timestamps, mind map, export

    7  layer7_errors      error handling and validation
    8  layer8_redteam     red teaming: injection, safety, bias
    9  layer9_eval        ROUGE, faithfulness, relevance, reviewer feedback
"""

__version__ = "1.0.0"

from .config import settings  # noqa: F401
from .pipeline import CourseRun, PipelineRun, run_course, run_pipeline  # noqa: F401

__all__ = [
    "settings",
    "run_pipeline", "PipelineRun",
    "run_course", "CourseRun",
    "__version__",
]
