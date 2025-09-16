import json
import os
import logging
from typing import List, Dict, Any, Optional
import openai
from openai import OpenAI
import anthropic
from anthropic import Anthropic
import time
import re

class AIHandler:
    def __init__(self, openai_key: str = None, claude_key: str = None):
        # the newest OpenAI model is "gpt-5" which was released August 7, 2025.
        # do not change this unless explicitly requested by the user
        self.openai_client = None
        self.claude_client = None
        
        if openai_key:
            self.openai_client = OpenAI(api_key=openai_key)
        
        if claude_key:
            self.claude_client = Anthropic(api_key=claude_key)
        
        # Available models
        self.openai_models = ["gpt-5", "gpt-5-mini"]
        self.claude_models = ["claude-sonnet-4-20250514"]
    
    def generate_titles(self, target_site: str, anchor_text: str, source_site: str,
                       style_patterns: Dict[str, Any], model: str = "gpt-5", 
                       count: int = 5) -> List[Dict[str, Any]]:
        """
        Generate SEO-optimized titles using specified AI model
        """
        try:
            # Prepare context and prompt
            prompt = self._build_generation_prompt(
                target_site, anchor_text, source_site, style_patterns, count
            )
            
            # Generate titles based on model
            if model in self.openai_models:
                titles = self._generate_with_openai(prompt, model, count)
            elif model in self.claude_models:
                titles = self._generate_with_claude(prompt, model, count)
            else:
                raise ValueError(f"Unsupported model: {model}")
            
            # Score and enhance titles
            enhanced_titles = []
            for title in titles:
                scored_title = self._score_title(title, anchor_text, style_patterns)
                enhanced_titles.append(scored_title)
            
            # Sort by SEO score
            enhanced_titles.sort(key=lambda x: x.get('seo_score', 0), reverse=True)
            
            logging.info(f"Generated {len(enhanced_titles)} titles using {model}")
            return enhanced_titles
            
        except Exception as e:
            logging.error(f"Title generation failed: {e}")
            return []
    
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
    
    def _generate_with_openai(self, prompt: str, model: str, count: int) -> List[str]:
        """Generate titles using OpenAI models"""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")
        
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert SEO title generator with deep understanding of editorial styles and link placement strategies."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.8,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            titles_data = json.loads(content)
            
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
    
    def _score_title(self, title: str, anchor_text: str, style_patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Score generated title on multiple criteria"""
        
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
    
    def score_title_advanced(self, title: str, anchor_text: str, 
                           target_style: Dict[str, Any]) -> Dict[str, float]:
        """
        Advanced scoring with machine learning-like approach
        """
        # This could be enhanced with actual ML models
        base_scores = self._score_title(title, anchor_text, target_style)
        
        # Additional advanced scoring
        advanced_scores = {
            'readability': self._calculate_readability(title),
            'click_potential': self._estimate_click_potential(title),
            'brand_safety': self._assess_brand_safety(title),
            'trending_potential': self._assess_trending_potential(title)
        }
        
        return {**base_scores, **advanced_scores}
    
    def _calculate_readability(self, title: str) -> float:
        """Calculate readability score"""
        words = title.split()
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        # Simple readability metric
        if avg_word_length <= 4:
            return 1.0
        elif avg_word_length <= 6:
            return 0.8
        else:
            return 0.6
    
    def _estimate_click_potential(self, title: str) -> float:
        """Estimate click-through potential"""
        power_words = ['ultimate', 'secret', 'proven', 'exclusive', 'breakthrough']
        curiosity_words = ['why', 'what', 'how', 'when', 'where']
        
        power_score = sum(1 for word in power_words if word in title.lower())
        curiosity_score = sum(1 for word in curiosity_words if word in title.lower())
        
        return min((power_score * 0.2 + curiosity_score * 0.3 + 0.5), 1.0)
    
    def _assess_brand_safety(self, title: str) -> float:
        """Assess brand safety of title"""
        risky_words = ['controversial', 'shocking', 'scandal', 'hate', 'dangerous']
        risky_count = sum(1 for word in risky_words if word in title.lower())
        
        return max(1.0 - (risky_count * 0.3), 0.0)
    
    def _assess_trending_potential(self, title: str) -> float:
        """Assess potential for trending/viral content"""
        trending_indicators = ['2024', '2025', 'new', 'latest', 'breaking', 'trending']
        trend_score = sum(1 for indicator in trending_indicators if indicator in title.lower())
        
        return min(trend_score * 0.25, 1.0)

    def generate_title_variations(self, base_title: str, variation_count: int = 3) -> List[str]:
        """Generate variations of a successful title"""
        if not self.openai_client:
            return []
        
        prompt = f"""
        Create {variation_count} variations of this successful title: "{base_title}"
        
        Keep the same core message and SEO value but vary:
        - Word choice and synonyms
        - Structure and format
        - Emotional appeal
        
        Return as JSON array of strings.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-5",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            variations = json.loads(content)
            
            if isinstance(variations, list):
                return variations
            elif isinstance(variations, dict) and 'variations' in variations:
                return variations['variations']
            
        except Exception as e:
            logging.error(f"Variation generation failed: {e}")
        
        return []
