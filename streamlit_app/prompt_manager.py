"""Prompt preset definitions and utilities for title generation prompts."""

from __future__ import annotations

from dataclasses import dataclass
from string import Formatter
from typing import Dict, Iterable, List, Mapping

PLACEHOLDER_DESCRIPTIONS: Dict[str, str] = {
    "domain": "Domain being searched on DataForSEO.",
    "target_url": "Final destination URL where the link will point.",
    "anchor_keyword": "Anchor text requested for the placement (optional usage).",
    "anchor_country": "Primary market or region for the anchor destination.",
    "serp_keyword": "Keyword used for the site: query.",
    "num_requested": "Number of titles requested from the model.",
    "serp_titles": "Reference SERP titles joined as bullet points or lines.",
    "additional_context": "Free-form notes captured in the UI.",
}


@dataclass
class PromptTemplate:
    """Represents a named prompt template with substitution fields."""

    key: str
    name: str
    description: str
    template: str

    def render(self, context: Mapping[str, str]) -> str:
        formatter = _SafeDict(context)
        return self.template.format_map(formatter)

    @property
    def placeholders(self) -> Iterable[str]:
        return _extract_placeholders(self.template)


class _SafeDict(dict):  # pragma: no cover - trivial container
    """Default missing formatter that preserves braces for undefined keys."""

    def __missing__(self, key: str) -> str:  # noqa: D401 - simple default
        return "{" + key + "}"


def _extract_placeholders(template: str) -> List[str]:
    formatter = Formatter()
    placeholders: List[str] = []
    for _, field_name, _, _ in formatter.parse(template):
        if field_name and field_name not in placeholders:
            placeholders.append(field_name)
    return placeholders


def get_default_templates() -> List[PromptTemplate]:
    """Return the default set of prompt templates."""

    structured = PromptTemplate(
        key="structured",
        name="Editorial Alignment",
        description="Structured brief that mirrors SERP tone but keeps anchor optional.",
        template=(
            "Content Title Creation Brief\n"
            "Objective: craft {num_requested} link-ready titles for {target_url} that feel at home on {domain}.\n\n"
            "Reference SERP keyword: {serp_keyword}\n"
            "Anchor keyword (use only if natural): {anchor_keyword}\n\n"
            "Anchor operates in: {anchor_country}\n\n"
            "Reference titles:\n{serp_titles}\n\n"
            "Guidance:\n"
            "1. Combine your own knowledge of {domain}'s catalog with the SERP references to surface authentic angles.\n"
            "2. Make every title feel native to {domain} while naturally justifying a link to {target_url}.\n"
            "3. Use the anchor text only when it reads naturally inside supporting copy, never inside the title.\n"
            "4. Titles must exclude {anchor_keyword} and any casino/operator brand names entirely; reserve brand mentions for rationales.\n"
            "5. If extra notes or your knowledge indicate a placement locale, tailor the titles accordingly; otherwise keep them globally applicable while acknowledging {anchor_country} where useful.\n"
            "6. Incorporate any extra campaign notes: {additional_context}.\n"
            "7. Keep every title free of review or promo phrasing (e.g., avoid 'review', 'bonus', 'promo code').\n\n"
            "Examples:\n"
            "- Bad: \"Casiny online casino review\" (violates brand + review rule).\n"
            "- Good: \"Bankroll-friendly casino formats poker players can trust\" (brand-free, contextual).\n\n"
            "Return JSON exactly as {{\"titles\": [{{\"title\": \"...\", \"rationale\": \"...\"}}]}} with {num_requested} items."
        ),
    )

    lightweight = PromptTemplate(
        key="lightweight",
        name="Light Guidance",
        description="Minimal prompt that nudges toward relevance and tone matching.",
        template=(
            "You are drafting {num_requested} potential article titles for {domain}.\n"
            "They should make sense for linking to {target_url}.\n"
            "Anchor keyword to use only if it fits in surrounding copy, never inside the title itself: {anchor_keyword}.\n"
            "Blend what you already know about {domain}'s content with the SERP list to stay on-brand.\n"
            "Anchor operates in: {anchor_country}.\n"
            "Use any locale cues from the prompt or your knowledge of {domain}; when {anchor_country} is relevant, acknowledge it without adding brand names.\n"
            "Never mention {domain}, {anchor_keyword}, or any casino/operator brand name inside a title, and avoid review-style phrasing.\n"
            "Search context keyword: {serp_keyword}.\n"
            "Reference titles:\n{serp_titles}\n\n"
            "Add quick rationales and keep the voice consistent with the domain.\n"
            "Extra notes: {additional_context}.\n"
            "Examples:\n"
            "- Bad title: \"Casiny social casino bonuses for poker players\" (brand + promo).\n"
            "- Good title: \"How social casino drills can sharpen live poker instincts\".\n"
            "Return strictly {{\"titles\": [{{\"title\": \"...\", \"rationale\": \"...\"}}]}} with {num_requested} items and no other keys."
        ),
    )

    target_focus = PromptTemplate(
        key="target_focus",
        name="Target URL Deep Dive",
        description="Emphasises insights about the destination page when crafting angles.",
        template=(
            "You are crafting {num_requested} article titles for {domain}.\n"
            "Use your knowledge of {target_url} to hint at why linking there adds value, without naming the brand in the headline.\n"
            "Anchor keyword (only in surrounding copy, never the title): {anchor_keyword}.\n"
            "Anchor operates in: {anchor_country}.\n"
            "Reference SERP keyword: {serp_keyword}.\n"
            "SERP sample:\n{serp_titles}\n\n"
            "Rules:\n"
            "- Titles must be brand-free and omit direct mentions of casinos/operators.\n"
            "- Reflect {domain}'s tone and audience.\n"
            "- When relevant, contrast or connect the placement context with {anchor_country}.\n"
            "- Keep titles free of review/promo phrasing (no 'review', 'bonus', 'promo code').\n"
            "- Incorporate any extra notes: {additional_context}.\n\n"
            "Return strictly {{\"titles\": [{{\"title\": \"...\", \"rationale\": \"...\"}}]}} with {num_requested} items and no other keys."
        ),
    )



    topical_pulse = PromptTemplate(
        key="topical_pulse",
        name="Topical Pulse",
        description="Single preset for newsy, trend, or thought-leadership angles grounded in current context.",
        template=(
            "Craft {num_requested} timely, on-brand article titles for {domain}.\n"
            "Anchor keyword (keep out of the headline itself): {anchor_keyword}.\n"
            "Operating region or audience to acknowledge: {anchor_country}.\n"
            "Target URL to justify via supporting copy: {target_url}.\n"
            "Search context keyword: {serp_keyword}.\n"
            "Reference titles (tone guide):\n{serp_titles}\n\n"
            "Guidance:\n"
            "- First, consult your recent memory for news, launches, regulatory moves, or data releases tied to {serp_keyword}; cite these in rationales.\n"
            "- When there is no fresh headline, lean on enduring macro trends or forward-looking questions practitioners are asking.\n"
            "- Make the rationale cite the specific event, trend, or why it matters now—especially for audiences in {anchor_country}.\n"
            "- Keep titles brand-free and avoid review/promo phrasing (no 'review', 'bonus', 'promo code').\n"
            "- Use {additional_context} when provided, but never invent unsupported facts.\n\n"
            "Return strictly {{\"titles\": [{{\"title\": \"...\", \"rationale\": \"...\"}}]}} with {num_requested} items and no other keys."
        ),
    )

    return [structured, lightweight, target_focus, topical_pulse]


__all__ = [
    "PromptTemplate",
    "PLACEHOLDER_DESCRIPTIONS",
    "get_default_templates",
]
