import json
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import openai
from openai import OpenAI
import anthropic
from anthropic import Anthropic
import time
import re
from datetime import datetime, timedelta
from collections import Counter
from dataforseo_api import DataForSEOAnalyzer
from dataforseo_simple import get_site_titles, get_competitor_titles

class AIHandler:
    def __init__(self, openai_key: str = None, claude_key: str = None, dataforseo_login: str = None, dataforseo_password: str = None):
        # the newest OpenAI model is "gpt-5" which was released August 7, 2025.
        # do not change this unless explicitly requested by the user
        self.openai_client = None
        self.claude_client = None

        if openai_key:
            # Clean and validate the API key
            clean_key = str(openai_key).strip()
            # Remove any URL prefixes that might have been added
            if clean_key.startswith('https://'):
                clean_key = clean_key.replace('https://', '')
            if clean_key.startswith('http://'):
                clean_key = clean_key.replace('http://', '')

            print(f"DEBUG: Using OpenAI key starting with: {clean_key[:10]}...")
            self.openai_client = OpenAI(api_key=clean_key)

        if claude_key:
            clean_claude_key = str(claude_key).strip()
            self.claude_client = Anthropic(api_key=clean_claude_key)

        # DataForSEO integration for research context
        self.dataforseo_analyzer = None
        if dataforseo_login and dataforseo_password:
            self.dataforseo_analyzer = DataForSEOAnalyzer(dataforseo_login, dataforseo_password)
            logging.info("DataForSEO analyzer initialized for research context")

        # Available models
        self.openai_models = ["gpt-5", "gpt-5-mini"]
        self.claude_models = ["claude-sonnet-4-20250514"]

        # Research context cache
        self.research_cache = {}

        # Editorial style templates
        self.editorial_styles = {
            'professional': {
                'tone': 'authoritative and informative',
                'keywords': ['comprehensive', 'analysis', 'guide', 'complete', 'ultimate'],
                'structure': 'clear hierarchy with descriptive titles',
                'directness': 0.7
            },
            'casual': {
                'tone': 'conversational and engaging',
                'keywords': ['best', 'amazing', 'awesome', 'cool', 'top'],
                'structure': 'friendly and approachable',
                'directness': 0.6
            },
            'technical': {
                'tone': 'precise and detailed',
                'keywords': ['advanced', 'technical', 'detailed', 'in-depth', 'expert'],
                'structure': 'methodical and structured',
                'directness': 0.8
            },
            'listicle': {
                'tone': 'organized and scannable',
                'keywords': ['ways', 'tips', 'steps', 'methods', 'strategies'],
                'structure': 'numbered or bulleted format',
                'directness': 0.5
            },
            'news': {
                'tone': 'urgent and factual',
                'keywords': ['breaking', 'latest', 'new', 'update', 'report'],
                'structure': 'headline style with key facts',
                'directness': 0.9
            }
        }

    def generate_titles(self, target_site: str, anchor_text: str, source_site: str,
                       style_patterns: Dict[str, Any], model: str = "gpt-5",
                       count: int = 5, use_research_context: bool = True) -> List[Dict[str, Any]]:
        """
        Generate SEO-optimized titles using specified AI model with research context
        """
        try:
            # Get research context if enabled
            research_context = None
            if use_research_context and self.dataforseo_analyzer:
                research_context = self._get_research_context(target_site, anchor_text)
                logging.info(f"Research context obtained: {len(research_context.get('titles', []))} examples found")

            # Prepare context and prompt with research integration
            prompt = self._build_enhanced_generation_prompt(
                target_site, anchor_text, source_site, style_patterns, count, research_context
            )

            # Generate titles based on model
            if model in self.openai_models:
                titles = self._generate_with_openai(prompt, model, count)
            elif model in self.claude_models:
                titles = self._generate_with_claude(prompt, model, count)
            else:
                raise ValueError(f"Unsupported model: {model}")

            # Score and enhance titles with context awareness
            enhanced_titles = []
            for title in titles:
                scored_title = self._score_title_with_context(title, anchor_text, style_patterns, research_context)
                enhanced_titles.append(scored_title)

            # Sort by SEO score
            enhanced_titles.sort(key=lambda x: x.get('seo_score', 0), reverse=True)

            logging.info(f"Generated {len(enhanced_titles)} titles using {model}")
            return enhanced_titles

        except Exception as e:
            logging.error(f"Title generation failed: {e}")
            return []

    def _get_research_context(self, target_site: str, anchor_text: str) -> Optional[Dict[str, Any]]:
        """
        Get research context from DataForSEO for the target site
        """
        try:
            # Create cache key
            cache_key = f"{target_site}_{anchor_text}"

            # Check cache first
            if cache_key in self.research_cache:
                cache_entry = self.research_cache[cache_key]
                if datetime.now() - cache_entry['timestamp'] < timedelta(hours=24):
                    logging.info(f"Using cached research context for {target_site}")
                    return cache_entry['data']

            # Extract domain from target site
            domain = target_site.replace('https://', '').replace('http://', '').replace('www.', '').split('/')[0]

            # Get site-specific titles using DataForSEO
            site_titles = get_site_titles(domain, anchor_text)

            if not site_titles:
                # Fallback to general competitor titles
                site_titles = get_competitor_titles(anchor_text)

            if site_titles:
                research_context = self._analyze_site_context(site_titles, anchor_text, domain)

                # Cache the results
                self.research_cache[cache_key] = {
                    'data': research_context,
                    'timestamp': datetime.now()
                }

                return research_context
            else:
                logging.warning(f"No research context found for {target_site} with keyword '{anchor_text}'")
                return None

        except Exception as e:
            logging.error(f"Failed to get research context: {e}")
            return None

    def _analyze_site_context(self, titles: List[str], keyword: str, domain: str) -> Dict[str, Any]:
        """
        Analyze existing site titles to understand editorial patterns
        """
        if not titles:
            return {}

        context = {
            'domain': domain,
            'keyword': keyword,
            'sample_size': len(titles),
            'titles': titles,
            'editorial_analysis': {},
            'style_indicators': {},
            'content_patterns': {},
            'recommendations': []
        }

        # Length analysis
        lengths = [len(title) for title in titles]
        context['editorial_analysis']['avg_length'] = sum(lengths) / len(lengths)
        context['editorial_analysis']['length_range'] = [min(lengths), max(lengths)]
        context['editorial_analysis']['optimal_length'] = self._get_optimal_length_range(lengths)

        # Keyword usage analysis
        keyword_lower = keyword.lower()
        keyword_positions = []
        keyword_usage_count = 0

        for title in titles:
            title_lower = title.lower()
            if keyword_lower in title_lower:
                keyword_usage_count += 1
                pos = title_lower.find(keyword_lower)
                keyword_positions.append(pos / len(title) if len(title) > 0 else 0)

        context['editorial_analysis']['keyword_usage_rate'] = keyword_usage_count / len(titles)
        context['editorial_analysis']['avg_keyword_position'] = (
            sum(keyword_positions) / len(keyword_positions) if keyword_positions else None
        )

        # Style detection
        context['style_indicators'] = self._detect_editorial_style(titles)

        # Content patterns
        context['content_patterns'] = self._analyze_content_patterns(titles, keyword)

        # Generate recommendations
        context['recommendations'] = self._generate_context_recommendations(context)

        logging.info(f"Site context analysis complete for {domain}: {len(titles)} titles analyzed")
        return context

    def _detect_editorial_style(self, titles: List[str]) -> Dict[str, Any]:
        """
        Detect the editorial style based on title patterns
        """
        style_scores = {
            'professional': 0,
            'casual': 0,
            'technical': 0,
            'listicle': 0,
            'news': 0
        }

        format_patterns = {
            'question': 0,
            'how_to': 0,
            'numbered': 0,
            'colon_separated': 0,
            'pipe_separated': 0
        }

        tone_indicators = {
            'authoritative': 0,
            'conversational': 0,
            'urgent': 0,
            'promotional': 0
        }

        for title in titles:
            title_lower = title.lower()

            # Style scoring
            for style, template in self.editorial_styles.items():
                for keyword in template['keywords']:
                    if keyword in title_lower:
                        style_scores[style] += 1

            # Format detection
            if title.endswith('?'):
                format_patterns['question'] += 1
            if 'how to' in title_lower:
                format_patterns['how_to'] += 1
            if re.search(r'\d+', title):
                format_patterns['numbered'] += 1
            if ':' in title:
                format_patterns['colon_separated'] += 1
            if '|' in title:
                format_patterns['pipe_separated'] += 1

            # Tone analysis
            if any(word in title_lower for word in ['complete', 'comprehensive', 'ultimate', 'definitive']):
                tone_indicators['authoritative'] += 1
            if any(word in title_lower for word in ['best', 'amazing', 'awesome', 'cool']):
                tone_indicators['conversational'] += 1
            if any(word in title_lower for word in ['breaking', 'urgent', 'now', 'alert']):
                tone_indicators['urgent'] += 1
            if any(word in title_lower for word in ['free', 'deal', 'discount', 'sale']):
                tone_indicators['promotional'] += 1

        # Determine dominant style
        dominant_style = max(style_scores, key=style_scores.get) if any(style_scores.values()) else 'professional'
        dominant_format = max(format_patterns, key=format_patterns.get) if any(format_patterns.values()) else 'standard'
        dominant_tone = max(tone_indicators, key=tone_indicators.get) if any(tone_indicators.values()) else 'neutral'

        return {
            'dominant_style': dominant_style,
            'style_confidence': style_scores[dominant_style] / len(titles),
            'dominant_format': dominant_format,
            'dominant_tone': dominant_tone,
            'style_distribution': style_scores,
            'format_distribution': format_patterns,
            'tone_distribution': tone_indicators
        }

    def _analyze_content_patterns(self, titles: List[str], keyword: str) -> Dict[str, Any]:
        """
        Analyze content patterns and themes
        """
        patterns = {
            'common_words': Counter(),
            'title_starters': Counter(),
            'title_enders': Counter(),
            'content_themes': [],
            'directness_level': 0.5
        }

        # Word frequency analysis
        all_words = []
        for title in titles:
            words = re.findall(r'\b\w+\b', title.lower())
            all_words.extend(words)
            patterns['common_words'].update(words)

        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        patterns['common_words'] = Counter({word: count for word, count in patterns['common_words'].items()
                                           if word not in stop_words and len(word) > 2})

        # Title pattern analysis
        for title in titles:
            words = title.split()
            if words:
                patterns['title_starters'][words[0].lower()] += 1
                patterns['title_enders'][words[-1].lower().rstrip('.,!?')] += 1

        # Directness analysis
        keyword_lower = keyword.lower()
        direct_mentions = sum(1 for title in titles if keyword_lower in title.lower())
        patterns['directness_level'] = direct_mentions / len(titles) if titles else 0.5

        return patterns

    def _get_optimal_length_range(self, lengths: List[int]) -> Tuple[int, int]:
        """
        Determine optimal length range based on existing titles
        """
        if not lengths:
            return (50, 60)

        avg_length = sum(lengths) / len(lengths)
        # Suggest range around average, clamped to SEO best practices
        optimal_min = max(45, int(avg_length - 10))
        optimal_max = min(70, int(avg_length + 10))

        return (optimal_min, optimal_max)

    def _generate_context_recommendations(self, context: Dict[str, Any]) -> List[str]:
        """
        Generate actionable recommendations based on site context
        """
        recommendations = []

        # Length recommendations
        avg_length = context['editorial_analysis'].get('avg_length', 55)
        recommendations.append(f"Target {int(avg_length)}±5 characters based on site's existing titles")

        # Style recommendations
        style_info = context['style_indicators']
        dominant_style = style_info.get('dominant_style', 'professional')
        confidence = style_info.get('style_confidence', 0)

        if confidence > 0.3:
            recommendations.append(f"Adopt {dominant_style} editorial style (matches {confidence:.1%} of existing content)")

        # Format recommendations
        dominant_format = style_info.get('dominant_format', 'standard')
        if dominant_format != 'standard':
            recommendations.append(f"Consider {dominant_format.replace('_', ' ')} format based on site preferences")

        # Keyword usage recommendations
        keyword_usage_rate = context['editorial_analysis'].get('keyword_usage_rate', 0)
        if keyword_usage_rate > 0.7:
            recommendations.append("Include target keyword directly (high keyword usage on site)")
        elif keyword_usage_rate < 0.3:
            recommendations.append("Consider subtle keyword integration (site uses indirect approach)")

        return recommendations

    def _build_enhanced_generation_prompt(self, target_site: str, anchor_text: str,
                                         source_site: str, style_patterns: Dict[str, Any],
                                         count: int, research_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Build enhanced prompt with research context integration
        """
        # Fallback to original method if no research context
        if not research_context:
            return self._build_generation_prompt(target_site, anchor_text, source_site, style_patterns, count)

        # Extract context information
        domain = research_context.get('domain', target_site)
        sample_size = research_context.get('sample_size', 0)
        editorial_analysis = research_context.get('editorial_analysis', {})
        style_indicators = research_context.get('style_indicators', {})
        recommendations = research_context.get('recommendations', [])

        # Build context-aware prompt
        prompt = f"""
You are an expert SEO title generator with access to detailed research about the target site's editorial style.

TARGET SITE RESEARCH:
- Domain: {domain}
- Analysis based on {sample_size} existing titles
- Average title length: {editorial_analysis.get('avg_length', 55):.0f} characters
- Keyword usage rate: {editorial_analysis.get('keyword_usage_rate', 0):.1%}
- Dominant editorial style: {style_indicators.get('dominant_style', 'professional')}
- Preferred format: {style_indicators.get('dominant_format', 'standard')}
- Content tone: {style_indicators.get('dominant_tone', 'neutral')}

TASK DETAILS:
- Target Website: {target_site}
- Target Keyword: "{anchor_text}"
- Source Website: {source_site}
- Required Count: {count} titles

EDITORIAL STYLE TEMPLATE:
{self._get_style_template(style_indicators.get('dominant_style', 'professional'))}

SITE-SPECIFIC RECOMMENDATIONS:
"""

        for i, rec in enumerate(recommendations, 1):
            prompt += f"{i}. {rec}\n"

        prompt += f"""

CONTEXT-AWARE GENERATION REQUIREMENTS:
1. Match the site's established editorial style and tone
2. Use similar title length patterns ({editorial_analysis.get('avg_length', 55):.0f}±10 characters)
3. Apply appropriate keyword integration based on site's usage patterns
4. Follow the site's preferred format conventions
5. Maintain consistency with the domain's content strategy
6. Ensure titles would fit naturally among existing content

GENERATE {count} titles that seamlessly blend with {domain}'s existing editorial approach.

Return as JSON array:
[
    {{
        "title": "Your contextually appropriate title",
        "reasoning": "How this matches the site's editorial style",
        "context_match_score": "high/medium/low",
        "style_adherence": "explanation of style matching"
    }}
]

Focus on creating titles that would be indistinguishable from content naturally created by {domain}'s editorial team.
"""

        return prompt

    def _get_style_template(self, style: str) -> str:
        """
        Get detailed style template for the identified editorial style
        """
        if style not in self.editorial_styles:
            style = 'professional'

        template = self.editorial_styles[style]

        return f"""
Tone: {template['tone']}
Key phrases: {', '.join(template['keywords'])}
Structure: {template['structure']}
Directness level: {template['directness']} (0=subtle, 1=direct)
"""

    def _build_generation_prompt(self, target_site: str, anchor_text: str,
                                source_site: str, style_patterns: Dict[str, Any],
                                count: int) -> str:
        """Build comprehensive prompt for title generation"""

        # Extract key information from style patterns
        editorial_style = style_patterns.get('editorial_style', 'neutral')
        avg_length = style_patterns.get('avg_length', 55)
        common_formats = style_patterns.get('common_formats', [])
        keyword_directness = style_patterns.get('keyword_directness', 0.5)
        site_tolerance = style_patterns.get('site_tolerance', {})
        content_types = style_patterns.get('content_types', [])

        prompt = f"""
You are an expert SEO title generator. Create {count} compelling, SEO-optimized titles for a link placement opportunity.

CONTEXT:
- Target Site: {target_site}
- Anchor Text/Keyword: "{anchor_text}"
- Source Site: {source_site}

SITE ANALYSIS:
- Editorial Style: {editorial_style}
- Preferred Title Length: {avg_length} characters
- Common Formats: {', '.join(common_formats)}
- Keyword Directness Level: {keyword_directness:.2f} (0=subtle, 1=direct)
- Content Types: {', '.join(content_types)}

SITE TOLERANCE:
- Direct Approach: {site_tolerance.get('direct', 0.5):.2f}
- Subtle Integration: {site_tolerance.get('subtle', 0.5):.2f}
- Creative Freedom: {site_tolerance.get('creative', 0.5):.2f}

REQUIREMENTS:
1. Create titles that match the target site's editorial style
2. Integrate the anchor text naturally based on the directness level
3. Follow the site's preferred formats and content types
4. Optimize for SEO (50-60 characters ideal)
5. Consider the source site context for relevance
6. Make titles compelling and click-worthy
7. Ensure each title is unique and original

ADAPTATION STRATEGY:
- If directness is high (>0.7): Use anchor text prominently and directly
- If directness is medium (0.3-0.7): Integrate anchor text naturally within context
- If directness is low (<0.3): Use anchor text subtly or as supporting context

Generate {count} titles and return them as a JSON array with this exact structure:
[
    {{
        "title": "Your SEO title here",
        "reasoning": "Brief explanation of approach",
        "directness_level": "high/medium/low",
        "estimated_performance": "percentage"
    }}
]

Focus on creating titles that would genuinely fit on the target site while effectively incorporating the anchor text.
"""

        return prompt

    def _score_title_with_context(self, title: str, anchor_text: str,
                                 style_patterns: Dict[str, Any],
                                 research_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Enhanced title scoring that considers research context
        """
        # Get base scores
        base_scores = self._score_title(title, anchor_text, style_patterns)

        if not research_context:
            return base_scores

        # Add context-aware scoring
        context_scores = self._score_context_match(title, anchor_text, research_context)

        # Combine scores with weighted average
        enhanced_scores = base_scores.copy()
        enhanced_scores.update({
            'context_match': context_scores['context_match'],
            'editorial_fit': context_scores['editorial_fit'],
            'length_appropriateness': context_scores['length_appropriateness'],
            'style_consistency': context_scores['style_consistency']
        })

        # Recalculate overall score with context weighting
        enhanced_scores['seo_score'] = self._calculate_enhanced_seo_score(enhanced_scores)

        return enhanced_scores

    def _score_context_match(self, title: str, anchor_text: str,
                            research_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score how well the title matches the researched site context
        """
        scores = {
            'context_match': 0,
            'editorial_fit': 0,
            'length_appropriateness': 0,
            'style_consistency': 0
        }

        editorial_analysis = research_context.get('editorial_analysis', {})
        style_indicators = research_context.get('style_indicators', {})

        # Length appropriateness (0-100)
        target_length = editorial_analysis.get('avg_length', 55)
        length_diff = abs(len(title) - target_length)
        if length_diff <= 5:
            scores['length_appropriateness'] = 100
        elif length_diff <= 10:
            scores['length_appropriateness'] = 80
        elif length_diff <= 15:
            scores['length_appropriateness'] = 60
        else:
            scores['length_appropriateness'] = 40

        # Style consistency (0-100)
        dominant_style = style_indicators.get('dominant_style', 'professional')
        if dominant_style in self.editorial_styles:
            style_template = self.editorial_styles[dominant_style]
            style_keywords = style_template['keywords']

            title_lower = title.lower()
            matching_keywords = sum(1 for keyword in style_keywords if keyword in title_lower)
            scores['style_consistency'] = min(100, (matching_keywords / len(style_keywords)) * 150)

        # Editorial fit based on format preferences (0-100)
        dominant_format = style_indicators.get('dominant_format', 'standard')
        format_score = 50  # Base score

        if dominant_format == 'question' and title.endswith('?'):
            format_score = 100
        elif dominant_format == 'how_to' and 'how to' in title.lower():
            format_score = 100
        elif dominant_format == 'numbered' and re.search(r'\d+', title):
            format_score = 100
        elif dominant_format == 'colon_separated' and ':' in title:
            format_score = 100
        elif dominant_format == 'pipe_separated' and '|' in title:
            format_score = 100

        scores['editorial_fit'] = format_score

        # Overall context match (weighted average)
        scores['context_match'] = (
            scores['length_appropriateness'] * 0.3 +
            scores['style_consistency'] * 0.4 +
            scores['editorial_fit'] * 0.3
        )

        return scores

    def _calculate_enhanced_seo_score(self, scores: Dict[str, Any]) -> float:
        """
        Calculate enhanced SEO score incorporating context awareness
        """
        base_seo = scores.get('seo_score', 0)
        context_match = scores.get('context_match', 0)

        # Weight: 70% base SEO, 30% context match
        enhanced_score = (base_seo * 0.7) + (context_match * 0.3)

        return min(100, enhanced_score)

    def _score_title(self, title: str, anchor_text: str, style_patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Score generated title on multiple criteria"""

        # Ensure title is a string, not a dict
        if isinstance(title, dict):
            title = title.get('title', str(title))
        title = str(title)

        # Ensure anchor_text is a string
        anchor_text = str(anchor_text)

        # Initialize scoring
        scores = {
            'title': title,
            'seo_score': 0,
            'style_match': 0,
            'keyword_position': 0,
            'emotional_score': 0,
            'acceptance_probability': 0
        }

        # SEO Score (0-100)
        seo_score = 0

        # Length optimization (50-60 chars is ideal)
        length = len(title)
        if 50 <= length <= 60:
            seo_score += 30
        elif 45 <= length <= 65:
            seo_score += 20
        elif 40 <= length <= 70:
            seo_score += 10

        # Keyword presence and position
        anchor_lower = anchor_text.lower()
        title_lower = title.lower()

        if anchor_lower in title_lower:
            seo_score += 25
            # Bonus for early position
            position = title_lower.find(anchor_lower)
            if position <= 10:
                seo_score += 15
                scores['keyword_position'] = 'early'
            elif position <= 25:
                seo_score += 10
                scores['keyword_position'] = 'middle'
            else:
                seo_score += 5
                scores['keyword_position'] = 'late'
        else:
            scores['keyword_position'] = 'not found'

        # Title structure
        if title[0].isupper():  # Proper capitalization
            seo_score += 5
        if not title.endswith('.'):  # No trailing period
            seo_score += 5
        if ':' in title or '|' in title:  # Clear structure
            seo_score += 5

        scores['seo_score'] = min(seo_score, 100)

        # Style Match Score (0-100)
        style_score = 50  # Base score

        editorial_style = style_patterns.get('editorial_style', 'neutral')
        common_formats = style_patterns.get('common_formats', [])

        # Match editorial style
        if editorial_style == 'formal':
            formal_words = ['analysis', 'comprehensive', 'guide', 'ultimate']
            if any(word in title_lower for word in formal_words):
                style_score += 20
        elif editorial_style == 'casual':
            casual_words = ['awesome', 'amazing', 'cool', 'best', 'top']
            if any(word in title_lower for word in casual_words):
                style_score += 20
        elif editorial_style == 'listicle':
            if re.search(r'\d+', title) or any(word in title_lower for word in ['top', 'best', 'ways']):
                style_score += 25

        # Match common formats
        for fmt in common_formats:
            if fmt == 'question' and title.endswith('?'):
                style_score += 15
            elif fmt == 'how-to' and 'how to' in title_lower:
                style_score += 15
            elif fmt == 'numbered' and re.search(r'\d+', title):
                style_score += 15
            elif fmt == 'colon-separated' and ':' in title:
                style_score += 15

        scores['style_match'] = min(style_score, 100)

        # Emotional Score (0-10)
        emotional_words = [
            'amazing', 'incredible', 'ultimate', 'best', 'perfect', 'essential',
            'powerful', 'proven', 'secret', 'exclusive', 'revolutionary'
        ]
        emotional_count = sum(1 for word in emotional_words if word in title_lower)
        scores['emotional_score'] = min(emotional_count * 2, 10)

        # Acceptance Probability (0-1)
        # Based on combination of other scores
        acceptance_prob = (
            (scores['seo_score'] / 100) * 0.4 +
            (scores['style_match'] / 100) * 0.4 +
            (scores['emotional_score'] / 10) * 0.2
        )
        scores['acceptance_probability'] = acceptance_prob

        return scores

    def _generate_with_openai(self, prompt: str, model: str, count: int) -> List[str]:
        """Generate titles using OpenAI models"""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")

        try:
            # GPT-5 doesn't support custom temperature, only default (1.0)
            api_params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are an expert SEO title generator with deep understanding of editorial styles and link placement strategies."},
                    {"role": "user", "content": prompt}
                ],
                "response_format": {"type": "json_object"},
                "max_completion_tokens": 2000
            }

            # Only add temperature for non-GPT-5 models
            if not model.startswith("gpt-5"):
                api_params["temperature"] = 0.8

            response = self.openai_client.chat.completions.create(**api_params)

            content = response.choices[0].message.content

            # Check if content is empty or None
            if not content or content.strip() == "":
                logging.error("OpenAI returned empty content")
                return []

            try:
                titles_data = json.loads(content)
            except json.JSONDecodeError as e:
                logging.error(f"JSON parsing failed. Content: {content[:200]}...")
                return self._extract_titles_from_text(content)

            # Extract titles from response
            if isinstance(titles_data, list):
                return [item.get('title', '') for item in titles_data if item.get('title')]
            elif isinstance(titles_data, dict) and 'titles' in titles_data:
                return titles_data['titles']
            else:
                # Fallback parsing
                return self._extract_titles_from_text(content)

        except Exception as e:
            logging.error(f"OpenAI generation failed: {e}")
            return []

    def _generate_with_claude(self, prompt: str, model: str, count: int) -> List[str]:
        """Generate titles using Claude models"""
        if not self.claude_client:
            raise ValueError("Claude client not initialized")

        try:
            response = self.claude_client.messages.create(
                model=model,
                max_tokens=2000,
                temperature=0.8,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            content = response.content[0].text

            # Try to parse as JSON first
            try:
                titles_data = json.loads(content)
                if isinstance(titles_data, list):
                    return [item.get('title', '') for item in titles_data if item.get('title')]
            except json.JSONDecodeError:
                pass

            # Fallback to text extraction
            return self._extract_titles_from_text(content)

        except Exception as e:
            logging.error(f"Claude generation failed: {e}")
            return []

    def _extract_titles_from_text(self, text: str) -> List[str]:
        """Extract titles from text when JSON parsing fails"""
        titles = []

        # Look for quoted titles
        quoted_matches = re.findall(r'"([^"]{20,80})"', text)
        titles.extend(quoted_matches)

        # Look for numbered lists
        numbered_matches = re.findall(r'\d+\.\s*([^\n]{20,80})', text)
        titles.extend(numbered_matches)

        # Look for bullet points
        bullet_matches = re.findall(r'[-•]\s*([^\n]{20,80})', text)
        titles.extend(bullet_matches)

        return titles[:10]  # Limit to 10 titles

    def generate_titles_with_research(self, target_site: str, anchor_text: str,
                                     source_site: str = "", model: str = "gpt-5",
                                     count: int = 5) -> Dict[str, Any]:
        """
        Generate titles with comprehensive research context integration
        """
        result = {
            'titles': [],
            'research_context': None,
            'generation_strategy': 'fallback',
            'context_available': False,
            'recommendations': []
        }

        try:
            # Attempt to get research context
            research_context = self._get_research_context(target_site, anchor_text)

            if research_context:
                result['research_context'] = research_context
                result['context_available'] = True
                result['generation_strategy'] = 'context_aware'
                result['recommendations'] = research_context.get('recommendations', [])

                logging.info(f"Research context available: {research_context['sample_size']} titles analyzed")
            else:
                logging.info("No research context available, using fallback generation")

            # Generate titles with appropriate strategy
            style_patterns = self._extract_style_patterns_from_context(research_context) if research_context else {}

            titles = self.generate_titles(
                target_site=target_site,
                anchor_text=anchor_text,
                source_site=source_site,
                style_patterns=style_patterns,
                model=model,
                count=count,
                use_research_context=bool(research_context)
            )

            result['titles'] = titles

            # Add context-specific metadata
            if research_context:
                for title_data in result['titles']:
                    title_data['generation_strategy'] = 'context_aware'
                    title_data['context_analysis'] = self._analyze_title_context_fit(
                        title_data['title'], research_context
                    )

            return result

        except Exception as e:
            logging.error(f"Enhanced title generation failed: {e}")
            result['error'] = str(e)
            return result

    def _extract_style_patterns_from_context(self, research_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract style patterns from research context for compatibility
        """
        if not research_context:
            return {}

        editorial_analysis = research_context.get('editorial_analysis', {})
        style_indicators = research_context.get('style_indicators', {})

        return {
            'editorial_style': style_indicators.get('dominant_style', 'professional'),
            'avg_length': editorial_analysis.get('avg_length', 55),
            'common_formats': [style_indicators.get('dominant_format', 'standard')],
            'keyword_directness': editorial_analysis.get('keyword_usage_rate', 0.5),
            'site_tolerance': {
                'direct': editorial_analysis.get('keyword_usage_rate', 0.5),
                'subtle': 1 - editorial_analysis.get('keyword_usage_rate', 0.5),
                'creative': 0.7
            },
            'content_types': [style_indicators.get('dominant_tone', 'neutral')]
        }

    def _analyze_title_context_fit(self, title: str, research_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze how well a generated title fits the researched context
        """
        if not research_context:
            return {}

        analysis = {
            'fits_length_pattern': False,
            'matches_style': False,
            'appropriate_directness': False,
            'format_consistency': False,
            'overall_fit': 'unknown'
        }

        editorial_analysis = research_context.get('editorial_analysis', {})
        style_indicators = research_context.get('style_indicators', {})

        # Length pattern analysis
        target_length = editorial_analysis.get('avg_length', 55)
        length_diff = abs(len(title) - target_length)
        analysis['fits_length_pattern'] = length_diff <= 10

        # Style matching
        dominant_style = style_indicators.get('dominant_style', 'professional')
        if dominant_style in self.editorial_styles:
            style_keywords = self.editorial_styles[dominant_style]['keywords']
            title_lower = title.lower()
            matching_keywords = sum(1 for keyword in style_keywords if keyword in title_lower)
            analysis['matches_style'] = matching_keywords > 0

        # Directness appropriateness
        keyword_usage_rate = editorial_analysis.get('keyword_usage_rate', 0.5)
        keyword = research_context.get('keyword', '')
        title_has_keyword = keyword.lower() in title.lower() if keyword else False

        if keyword_usage_rate > 0.7:  # High directness expected
            analysis['appropriate_directness'] = title_has_keyword
        elif keyword_usage_rate < 0.3:  # Low directness expected
            analysis['appropriate_directness'] = not title_has_keyword
        else:  # Medium directness
            analysis['appropriate_directness'] = True

        # Format consistency
        dominant_format = style_indicators.get('dominant_format', 'standard')
        if dominant_format == 'question':
            analysis['format_consistency'] = title.endswith('?')
        elif dominant_format == 'how_to':
            analysis['format_consistency'] = 'how to' in title.lower()
        elif dominant_format == 'numbered':
            analysis['format_consistency'] = bool(re.search(r'\d+', title))
        elif dominant_format == 'colon_separated':
            analysis['format_consistency'] = ':' in title
        elif dominant_format == 'pipe_separated':
            analysis['format_consistency'] = '|' in title
        else:
            analysis['format_consistency'] = True  # Standard format is always acceptable

        # Overall fit assessment
        fit_score = sum([
            analysis['fits_length_pattern'],
            analysis['matches_style'],
            analysis['appropriate_directness'],
            analysis['format_consistency']
        ])

        if fit_score >= 3:
            analysis['overall_fit'] = 'excellent'
        elif fit_score >= 2:
            analysis['overall_fit'] = 'good'
        elif fit_score >= 1:
            analysis['overall_fit'] = 'fair'
        else:
            analysis['overall_fit'] = 'poor'

        return analysis

    def generate_batch_with_context(self, batch_data: List[Dict[str, str]],
                                   model: str = "gpt-5",
                                   titles_per_item: int = 3) -> List[Dict[str, Any]]:
        """
        Generate titles for multiple items with research context awareness
        """
        results = []

        for i, item in enumerate(batch_data):
            logging.info(f"Processing batch item {i+1}/{len(batch_data)}: {item.get('target_site', 'Unknown')}")

            try:
                result = self.generate_titles_with_research(
                    target_site=item.get('target_site', ''),
                    anchor_text=item.get('anchor_text', ''),
                    source_site=item.get('source_site', ''),
                    model=model,
                    count=titles_per_item
                )

                result['batch_index'] = i
                result['input_data'] = item
                results.append(result)

            except Exception as e:
                logging.error(f"Batch item {i+1} failed: {e}")
                results.append({
                    'batch_index': i,
                    'input_data': item,
                    'titles': [],
                    'error': str(e),
                    'context_available': False
                })

        return results

    def get_fallback_generation_strategy(self, target_site: str, anchor_text: str) -> Dict[str, Any]:
        """
        Provide fallback generation strategy when research context is unavailable
        """
        strategy = {
            'approach': 'generic_best_practices',
            'reasoning': 'No site-specific research available',
            'recommendations': [
                'Use SEO best practices for title length (50-60 characters)',
                'Include target keyword naturally in the title',
                'Follow professional editorial style as default',
                'Ensure title is compelling and click-worthy',
                'Consider the target site\'s general industry and audience'
            ],
            'style_assumptions': {
                'editorial_style': 'professional',
                'avg_length': 55,
                'keyword_directness': 0.6,
                'tone': 'authoritative'
            }
        }

        # Try to infer some basic characteristics from the domain
        domain = target_site.replace('https://', '').replace('http://', '').replace('www.', '').split('/')[0]

        # Basic domain-based style inference
        if any(term in domain for term in ['blog', 'news', 'journal']):
            strategy['style_assumptions']['editorial_style'] = 'news'
            strategy['style_assumptions']['tone'] = 'urgent'
        elif any(term in domain for term in ['tech', 'dev', 'code', 'api']):
            strategy['style_assumptions']['editorial_style'] = 'technical'
            strategy['style_assumptions']['tone'] = 'precise'
        elif any(term in domain for term in ['shop', 'store', 'buy', 'sale']):
            strategy['style_assumptions']['editorial_style'] = 'casual'
            strategy['style_assumptions']['tone'] = 'promotional'

        return strategy

    def log_context_usage(self, target_site: str, anchor_text: str,
                         context_available: bool, generation_successful: bool,
                         titles_generated: int = 0) -> None:
        """
        Log context usage for debugging and analytics
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'target_site': target_site,
            'anchor_text': anchor_text,
            'context_available': context_available,
            'generation_successful': generation_successful,
            'titles_generated': titles_generated,
            'strategy': 'context_aware' if context_available else 'fallback'
        }

        logging.info(f"Context usage: {json.dumps(log_entry)}")

        # Could be enhanced to write to a separate analytics log file
        # or database for more detailed tracking

    def get_research_summary(self, target_site: str, anchor_text: str) -> Dict[str, Any]:
        """
        Get a summary of available research context for a site/keyword combination
        """
        try:
            research_context = self._get_research_context(target_site, anchor_text)

            if not research_context:
                return {
                    'available': False,
                    'message': 'No research context available for this site/keyword combination'
                }

            summary = {
                'available': True,
                'domain': research_context.get('domain'),
                'keyword': research_context.get('keyword'),
                'sample_size': research_context.get('sample_size', 0),
                'editorial_style': research_context.get('style_indicators', {}).get('dominant_style'),
                'avg_title_length': research_context.get('editorial_analysis', {}).get('avg_length'),
                'keyword_usage_rate': research_context.get('editorial_analysis', {}).get('keyword_usage_rate'),
                'recommendations_count': len(research_context.get('recommendations', [])),
                'last_updated': 'recent'  # Could be enhanced with actual timestamp
            }

            return summary

        except Exception as e:
            return {
                'available': False,
                'error': str(e)
            }

    def clear_research_cache(self) -> bool:
        """
        Clear the research context cache
        """
        try:
            self.research_cache.clear()
            logging.info("Research context cache cleared")
            return True
        except Exception as e:
            logging.error(f"Failed to clear research cache: {e}")
            return False