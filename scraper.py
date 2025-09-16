import requests
import trafilatura
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse, urljoin
from typing import Dict, List, Any, Optional
import logging
import time
from collections import Counter
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import os

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

class WebsiteAnalyzer:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.sia = SentimentIntensityAnalyzer()
        
    def analyze_website_style(self, url: str) -> Dict[str, Any]:
        """
        Comprehensive analysis of website style and patterns
        """
        try:
            domain = self._clean_domain(url)
            
            # Extract main content
            content = self._extract_content(url)
            if not content:
                return self._get_default_analysis()
            
            # Analyze different aspects
            analysis = {
                'domain': domain,
                'title_patterns': self._extract_title_patterns(content),
                'avg_length': self._calculate_avg_title_length(content),
                'editorial_style': self._detect_editorial_style(content),
                'keyword_directness': self._assess_keyword_directness(content),
                'emotional_tone': self._analyze_emotional_tone(content),
                'common_formats': self._identify_common_formats(content),
                'content_types': self._classify_content_types(content),
                'site_tolerance': self._assess_site_tolerance(content),
                'patterns': self._extract_patterns_from_titles(content)
            }
            
            logging.info(f"Website analysis completed for {domain}")
            return analysis
            
        except Exception as e:
            logging.error(f"Website analysis failed for {url}: {e}")
            return self._get_default_analysis()
    
    def _extract_content(self, url: str) -> Optional[str]:
        """Extract clean text content from website"""
        try:
            # Use trafilatura for clean content extraction
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                return None
            
            content = trafilatura.extract(downloaded)
            
            # Also try to get titles from the page
            soup = BeautifulSoup(downloaded, 'html.parser')
            titles = []
            
            # Extract various title elements
            for tag in ['h1', 'h2', 'h3', 'title']:
                elements = soup.find_all(tag)
                for elem in elements:
                    text = elem.get_text().strip()
                    if text and len(text) > 10:  # Filter out very short titles
                        titles.append(text)
            
            # Try to get more titles from meta tags
            meta_titles = soup.find_all('meta', attrs={'name': re.compile(r'title|headline', re.I)})
            for meta in meta_titles:
                if meta.get('content'):
                    titles.append(meta['content'])
            
            # Combine content with extracted titles
            if titles:
                content = content + "\n\nTITLES:\n" + "\n".join(titles)
            
            return content
            
        except Exception as e:
            logging.error(f"Content extraction failed: {e}")
            return None
    
    def _extract_title_patterns(self, content: str) -> List[str]:
        """Extract title patterns from content"""
        if not content:
            return []
        
        lines = content.split('\n')
        potential_titles = []
        
        for line in lines:
            line = line.strip()
            # Look for title-like patterns
            if (20 <= len(line) <= 100 and 
                ':' in line or 
                line.endswith('?') or 
                any(word in line.lower() for word in ['how', 'what', 'why', 'best', 'top', 'guide'])):
                potential_titles.append(line)
        
        return potential_titles[:20]  # Return top 20 patterns
    
    def _calculate_avg_title_length(self, content: str) -> int:
        """Calculate average title length"""
        patterns = self._extract_title_patterns(content)
        if not patterns:
            return 55  # Default SEO-friendly length
        
        return int(sum(len(pattern) for pattern in patterns) / len(patterns))
    
    def _detect_editorial_style(self, content: str) -> str:
        """Detect editorial style of the website"""
        if not content:
            return "neutral"
        
        content_lower = content.lower()
        
        # Define style indicators
        formal_indicators = ['furthermore', 'however', 'therefore', 'consequently', 'analysis', 'research']
        casual_indicators = ['awesome', 'cool', 'amazing', 'love', 'hate', 'totally', 'really']
        listicle_indicators = ['top 10', 'best of', 'ways to', 'tips for', 'things you']
        news_indicators = ['breaking', 'update', 'report', 'according to', 'announced']
        
        scores = {
            'formal': sum(1 for word in formal_indicators if word in content_lower),
            'casual': sum(1 for word in casual_indicators if word in content_lower),
            'listicle': sum(1 for phrase in listicle_indicators if phrase in content_lower),
            'news': sum(1 for word in news_indicators if word in content_lower)
        }
        
        return max(scores, key=scores.get) if any(scores.values()) else "neutral"
    
    def _assess_keyword_directness(self, content: str) -> float:
        """Assess how directly keywords are used (0-1 scale)"""
        if not content:
            return 0.5
        
        # Look for direct keyword usage patterns
        direct_patterns = [
            r'\b\w+\s+for\s+\w+\b',  # "tools for developers"
            r'\bbest\s+\w+\b',        # "best practices"
            r'\btop\s+\d+\b',         # "top 10"
            r'\bhow\s+to\s+\w+\b'     # "how to do"
        ]
        
        direct_matches = 0
        total_sentences = len(content.split('.'))
        
        for pattern in direct_patterns:
            matches = len(re.findall(pattern, content, re.IGNORECASE))
            direct_matches += matches
        
        # Calculate directness score
        directness = min(direct_matches / max(total_sentences, 1), 1.0)
        return directness
    
    def _analyze_emotional_tone(self, content: str) -> Dict[str, float]:
        """Analyze emotional tone of content"""
        if not content:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        
        # Sample content for analysis (to avoid processing too much text)
        sample = content[:2000]
        scores = self.sia.polarity_scores(sample)
        
        return {
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'compound': scores['compound']
        }
    
    def _identify_common_formats(self, content: str) -> List[str]:
        """Identify common title formats"""
        patterns = self._extract_title_patterns(content)
        
        format_patterns = []
        
        for pattern in patterns:
            # Identify format types
            if pattern.endswith('?'):
                format_patterns.append('question')
            elif 'how to' in pattern.lower():
                format_patterns.append('how-to')
            elif re.search(r'\d+', pattern):
                format_patterns.append('numbered')
            elif 'vs' in pattern.lower() or 'versus' in pattern.lower():
                format_patterns.append('comparison')
            elif any(word in pattern.lower() for word in ['best', 'top']):
                format_patterns.append('superlative')
            elif pattern.count(':') > 0:
                format_patterns.append('colon-separated')
            else:
                format_patterns.append('standard')
        
        # Return most common formats
        format_counts = Counter(format_patterns)
        return [fmt for fmt, count in format_counts.most_common(5)]
    
    def _classify_content_types(self, content: str) -> List[str]:
        """Classify types of content on the site"""
        content_lower = content.lower()
        
        content_types = []
        
        # Define content type indicators
        type_indicators = {
            'blog': ['posted', 'author', 'blog', 'article'],
            'news': ['breaking', 'update', 'report', 'news'],
            'tutorial': ['step', 'tutorial', 'guide', 'how-to'],
            'review': ['review', 'rating', 'pros', 'cons'],
            'listicle': ['top', 'best', 'worst', 'list'],
            'commercial': ['buy', 'price', 'sale', 'discount']
        }
        
        for content_type, indicators in type_indicators.items():
            score = sum(1 for indicator in indicators if indicator in content_lower)
            if score > 0:
                content_types.append((content_type, score))
        
        # Sort by score and return top types
        content_types.sort(key=lambda x: x[1], reverse=True)
        return [ct[0] for ct in content_types[:3]]
    
    def _assess_site_tolerance(self, content: str) -> Dict[str, float]:
        """Assess site's tolerance for different approaches"""
        if not content:
            return {'direct': 0.5, 'subtle': 0.5, 'creative': 0.5}
        
        # Analyze content for tolerance indicators
        directness = self._assess_keyword_directness(content)
        emotional_tone = self._analyze_emotional_tone(content)
        
        # Calculate tolerance scores
        direct_tolerance = directness
        subtle_tolerance = 1.0 - directness
        creative_tolerance = abs(emotional_tone.get('compound', 0))
        
        return {
            'direct': direct_tolerance,
            'subtle': subtle_tolerance,
            'creative': creative_tolerance
        }
    
    def _extract_patterns_from_titles(self, content: str) -> List[Dict[str, Any]]:
        """Extract detailed patterns from titles"""
        patterns = self._extract_title_patterns(content)
        
        analyzed_patterns = []
        
        for pattern in patterns:
            analysis = {
                'text': pattern,
                'length': len(pattern),
                'word_count': len(pattern.split()),
                'has_numbers': bool(re.search(r'\d+', pattern)),
                'has_question': pattern.endswith('?'),
                'has_colon': ':' in pattern,
                'starts_with_action': any(pattern.lower().startswith(word) for word in ['how', 'what', 'why', 'when']),
                'emotional_words': self._count_emotional_words(pattern)
            }
            analyzed_patterns.append(analysis)
        
        return analyzed_patterns
    
    def _count_emotional_words(self, text: str) -> int:
        """Count emotional words in text"""
        emotional_words = [
            'amazing', 'awesome', 'incredible', 'fantastic', 'great', 'excellent',
            'terrible', 'awful', 'horrible', 'shocking', 'surprising', 'stunning'
        ]
        
        return sum(1 for word in emotional_words if word in text.lower())
    
    def _clean_domain(self, url: str) -> str:
        """Extract clean domain from URL"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc or parsed.path
            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except:
            return url
    
    def _get_default_analysis(self) -> Dict[str, Any]:
        """Return default analysis when extraction fails"""
        return {
            'domain': 'unknown',
            'title_patterns': [],
            'avg_length': 55,
            'editorial_style': 'neutral',
            'keyword_directness': 0.5,
            'emotional_tone': {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'compound': 0.0},
            'common_formats': ['standard'],
            'content_types': ['blog'],
            'site_tolerance': {'direct': 0.5, 'subtle': 0.5, 'creative': 0.5},
            'patterns': []
        }
    
    def get_competitor_analysis(self, domain: str, keywords: List[str]) -> Dict[str, Any]:
        """Analyze competitor sites for the same keywords"""
        # This would require search API integration
        # For now, return a placeholder structure
        return {
            'competitors': [],
            'common_patterns': [],
            'avg_performance': {}
        }
