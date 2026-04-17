"""Auto-tag and auto-categorize documents during ingestion.

After text extraction, the LLM reads the document title + first few pages
and suggests:
- Collection: which domain database this belongs to
- Categories: hierarchical subject areas
- Tags: specific keywords/topics

This replaces the manual "type tags in the upload form" workflow. Users
can still override via the Manage page.

Uses the same LLM endpoint configured for entity extraction.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from backend.services.llm_service import LLMService

logger = logging.getLogger(__name__)


class AutoTagResult(BaseModel):
    """Structured output from the LLM auto-tagger."""
    collection: str = Field(
        default="default",
        description="Suggested collection name (lowercase_underscores)",
    )
    categories: list[str] = Field(
        default_factory=list,
        description="Suggested categories (2-4 recommended)",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Suggested tags (5-10 keywords)",
    )


SYSTEM_PROMPT = """\
You are a librarian organizing engineering reference documents into a
knowledge base. Given the title and first few pages of a document,
suggest how to organize it.

You must return a JSON object with these fields:

{
  "collection": "lowercase_name_for_the_domain",
  "categories": ["Category 1", "Category 2"],
  "tags": ["tag1", "tag2", "tag3"]
}

Collection naming rules:
- Use lowercase with underscores: "asm_references", "mechanical_design"
- Group by broad domain, not individual book
- Common collections: asm_references, mechanical_design, electrical_engineering,
  materials_science, welding_codes, firearms_design, physics, electronics,
  manufacturing, standards_codes

Category rules:
- 2-4 categories per document
- Hierarchical is OK: "Welding", "Brazing and Soldering"
- Use title case

Tag rules:
- 5-10 specific keywords
- Lowercase
- Include: subject matter, specific topics covered, material types mentioned,
  standards referenced, processes described
- Examples: "welding", "brazing", "nickel-alloys", "corrosion", "heat-treatment",
  "ASME", "pressure-vessels"

Output JSON only. No prose, no code fences.

/no_think
"""

USER_PROMPT_TEMPLATE = """\
Suggest collection, categories, and tags for this document.

Title: {title}
Filename: {filename}

First pages text (truncated):
{sample_text}
"""


class AutoTagger:
    """Suggests collection, categories, and tags for a document."""

    def __init__(self, llm: LLMService):
        self.llm = llm

    async def suggest(
        self,
        *,
        title: str,
        filename: str,
        sample_pages_text: list[str],
    ) -> AutoTagResult:
        """Call the LLM to suggest organization for a document.

        sample_pages_text: text from the first 3-5 pages (after title/TOC).
        """
        # Combine sample text, cap at ~3000 chars
        combined = "\n\n---\n\n".join(sample_pages_text)[:3000]

        user_msg = USER_PROMPT_TEMPLATE.format(
            title=title,
            filename=filename,
            sample_text=combined,
        )

        try:
            result = await self.llm.chat_json_structured(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                AutoTagResult,
                max_tokens=1024,
                temperature=0.1,
            )
            # Normalize collection name
            result.collection = (
                result.collection
                .strip()
                .lower()
                .replace(" ", "_")
                .replace("-", "_")
            )
            # Normalize tags
            result.tags = [
                t.strip().lower().replace(" ", "-")
                for t in result.tags
                if t.strip()
            ]
            logger.info(
                "Auto-tag suggestion: collection=%s, categories=%s, tags=%s",
                result.collection,
                result.categories,
                result.tags,
            )
            return result
        except Exception as exc:
            logger.warning("Auto-tagger failed (using defaults): %s", exc)
            return AutoTagResult()
