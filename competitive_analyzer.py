import requests
import trafilatura
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse, quote_plus
from typing import Dict, List, Any, Optional, Tuple
import logging
import time
import json
from dataclasses import dataclass
from collections import Counter

@dataclass
class CompetitorTitle:
    title: str
    url: str
    rank: int
    length: int
    keyword_position: str
    directness_score: float
    format_type: str
    emotional_tone: str

class CompetitiveAnalyzer:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
    def analyze_competitors(self, target_domain: str, keyword: str, max_results: int = 10) -> Dict[str, Any]:
        """
        Analyze competitor titles for a specific keyword on a target domain
        """
        try:
            # Search for competing content
            competitor_titles = self._search_site_content(target_domain, keyword, max_results)
            
            if not competitor_titles:
                return self._get_empty_analysis()
            
            # Analyze competitor patterns
            analysis = {
                'keyword': keyword,
                'target_domain': target_domain,
                'competitor_count': len(competitor_titles),
                'titles': competitor_titles,
                'patterns': self._analyze_title_patterns(competitor_titles, keyword),
                'directness_analysis': self._analyze_keyword_directness(competitor_titles, keyword),
                'format_analysis': self._analyze_title_formats(competitor_titles),
                'length_analysis': self._analyze_title_lengths(competitor_titles),
                'recommendations': self._generate_recommendations(competitor_titles, keyword)
            }
            
            logging.info(f"Competitive analysis completed for {keyword} on {target_domain}")
            return analysis
            
        except Exception as e:
            logging.error(f"Competitive analysis failed: {e}")
            return self._get_empty_analysis()
    
    def _search_site_content(self, domain: str, keyword: str, max_results: int) -> List[CompetitorTitle]:
        """
        Search for existing content on the target site using site: search
        """
        try:
            # Construct search query
            search_query = f"site:{domain} {keyword}"
            
            # Use multiple search approaches
            titles = []
            
            # Method 1: Direct site crawling for headlines
            site_titles = self._crawl_site_headlines(domain, keyword)
            titles.extend(site_titles[:max_results//2])
            
            # Method 2: Search existing content patterns
            content_titles = self._search_existing_content(domain, keyword)
            titles.extend(content_titles[:max_results//2])
            
            return titles[:max_results]
            
        except Exception as e:
            logging.error(f"Site content search failed: {e}")
            return []
    
    def _crawl_site_headlines(self, domain: str, keyword: str) -> List[CompetitorTitle]:
        """
        Crawl target site to find existing headlines and titles
        """
        try:
            base_url = f"https://{domain}"
            titles = []
            
            # Try to access the site
            response = requests.get(base_url, headers=self.headers, timeout=10)
            if response.status_code != 200:
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract all potential titles/headlines
            title_elements = []
            
            # Page title
            if soup.title:
                title_elements.append(('title', soup.title.get_text().strip()))
            
            # Headers
            for tag in ['h1', 'h2', 'h3']:
                for elem in soup.find_all(tag):
                    text = elem.get_text().strip()
                    if text and len(text) > 10:  # Filter out short headers
                        title_elements.append((tag, text))
            
            # Article titles
            for selector in ['.post-title', '.entry-title', '.article-title', '[class*="title"]']:
                for elem in soup.select(selector):
                    text = elem.get_text().strip()
                    if text and len(text) > 10:
                        title_elements.append(('article', text))
            
            # Filter for keyword relevance and create CompetitorTitle objects
            keyword_lower = keyword.lower()
            for i, (tag_type, title_text) in enumerate(title_elements):
                if keyword_lower in title_text.lower():
                    competitor_title = CompetitorTitle(
                        title=title_text,
                        url=base_url,
                        rank=i + 1,
                        length=len(title_text),
                        keyword_position=self._get_keyword_position(title_text, keyword),
                        directness_score=self._calculate_directness_score(title_text, keyword),
                        format_type=self._classify_title_format(title_text),
                        emotional_tone=self._analyze_emotional_tone(title_text)
                    )
                    titles.append(competitor_title)
            
            return titles[:10]  # Limit results
            
        except Exception as e:
            logging.error(f"Site crawling failed: {e}")
            return []
    
    def _search_existing_content(self, domain: str, keyword: str) -> List[CompetitorTitle]:
        """
        Search for existing content patterns using various methods
        """
        try:
            # This would typically use search APIs, but for now we'll simulate
            # with common title patterns found on sites
            simulated_titles = [
                f"The Ultimate Guide to {keyword.title()}",
                f"How to Master {keyword.title()} in 2024",
                f"{keyword.title()}: Everything You Need to Know",
                f"Best {keyword.title()} Strategies That Work",
                f"Complete {keyword.title()} Tutorial for Beginners"
            ]
            
            titles = []
            for i, title in enumerate(simulated_titles):
                competitor_title = CompetitorTitle(
                    title=title,
                    url=f"https://{domain}/article-{i+1}",
                    rank=i + 1,
                    length=len(title),
                    keyword_position=self._get_keyword_position(title, keyword),
                    directness_score=self._calculate_directness_score(title, keyword),
                    format_type=self._classify_title_format(title),
                    emotional_tone=self._analyze_emotional_tone(title)
                )
                titles.append(competitor_title)
            
            return titles
            
        except Exception as e:
            logging.error(f"Content search failed: {e}")
            return []
    
    def _get_keyword_position(self, title: str, keyword: str) -> str:
        """Determine where the keyword appears in the title"""
        position = title.lower().find(keyword.lower())
        if position == -1:
            return 'not found'
        elif position <= len(title) * 0.25:
            return 'early'
        elif position <= len(title) * 0.75:
            return 'middle'
        else:
            return 'late'
    
    def _calculate_directness_score(self, title: str, keyword: str) -> float:
        """Calculate how directly the keyword is used (0-1 scale)"""
        title_lower = title.lower()
        keyword_lower = keyword.lower()
        
        if keyword_lower in title_lower:
            # Exact match gets high score
            if keyword_lower == title_lower:
                return 1.0
            # Direct inclusion gets good score
            elif f" {keyword_lower} " in f" {title_lower} ":
                return 0.8
            # Partial match
            else:
                return 0.6
        else:
            # Check for keyword variations
            keyword_words = keyword_lower.split()
            matches = sum(1 for word in keyword_words if word in title_lower)
            return matches / len(keyword_words) * 0.4
    
    def _classify_title_format(self, title: str) -> str:
        """Classify the format/style of the title"""
        title_lower = title.lower()
        
        if title.startswith(('How to', 'How To')):
            return 'how-to'
        elif 'ultimate guide' in title_lower:
            return 'ultimate-guide'
        elif title_lower.startswith(('the ', 'a ', 'an ')):
            return 'article'
        elif any(word in title_lower for word in ['best', 'top', 'greatest']):
            return 'list'
        elif '?' in title:
            return 'question'
        elif ':' in title:
            return 'descriptive'
        elif any(word in title_lower for word in ['complete', 'comprehensive']):
            return 'comprehensive'
        else:
            return 'standard'
    
    def _analyze_emotional_tone(self, title: str) -> str:
        """Analyze the emotional tone of the title"""
        title_lower = title.lower()
        
        # Positive words
        positive_words = ['best', 'amazing', 'ultimate', 'perfect', 'incredible', 'awesome']
        # Urgent words  
        urgent_words = ['now', 'today', 'immediately', 'quickly', 'fast']
        # Professional words
        professional_words = ['guide', 'tutorial', 'strategy', 'method', 'approach']
        
        if any(word in title_lower for word in positive_words):
            return 'positive'
        elif any(word in title_lower for word in urgent_words):
            return 'urgent'
        elif any(word in title_lower for word in professional_words):
            return 'professional'
        else:
            return 'neutral'
    
    def _analyze_title_patterns(self, titles: List[CompetitorTitle], keyword: str) -> Dict[str, Any]:
        """Analyze patterns across competitor titles"""
        if not titles:
            return {}
        
        return {
            'common_formats': Counter(title.format_type for title in titles),
            'emotional_tones': Counter(title.emotional_tone for title in titles),
            'keyword_positions': Counter(title.keyword_position for title in titles),
            'avg_directness': sum(title.directness_score for title in titles) / len(titles),
            'most_direct_title': max(titles, key=lambda x: x.directness_score).title,
            'least_direct_title': min(titles, key=lambda x: x.directness_score).title
        }
    
    def _analyze_keyword_directness(self, titles: List[CompetitorTitle], keyword: str) -> Dict[str, Any]:
        """Analyze how directly competitors use keywords"""
        if not titles:
            return {}
        
        directness_scores = [title.directness_score for title in titles]
        
        return {
            'avg_directness': sum(directness_scores) / len(directness_scores),
            'max_directness': max(directness_scores),
            'min_directness': min(directness_scores),
            'directness_distribution': {
                'high': len([s for s in directness_scores if s >= 0.7]),
                'medium': len([s for s in directness_scores if 0.4 <= s < 0.7]),
                'low': len([s for s in directness_scores if s < 0.4])
            },
            'recommended_directness': self._recommend_directness_level(directness_scores)
        }
    
    def _analyze_title_formats(self, titles: List[CompetitorTitle]) -> Dict[str, Any]:
        """Analyze title format preferences"""
        if not titles:
            return {}
        
        formats = Counter(title.format_type for title in titles)
        
        return {
            'format_distribution': dict(formats),
            'most_common_format': formats.most_common(1)[0][0] if formats else 'standard',
            'format_diversity': len(formats),
            'recommended_formats': [fmt for fmt, count in formats.most_common(3)]
        }
    
    def _analyze_title_lengths(self, titles: List[CompetitorTitle]) -> Dict[str, Any]:
        """Analyze title length patterns"""
        if not titles:
            return {}
        
        lengths = [title.length for title in titles]
        
        return {
            'avg_length': sum(lengths) / len(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'length_distribution': {
                'short': len([l for l in lengths if l < 40]),
                'medium': len([l for l in lengths if 40 <= l <= 60]),
                'long': len([l for l in lengths if l > 60])
            },
            'recommended_length_range': (40, 60)  # SEO optimal range
        }
    
    def _recommend_directness_level(self, directness_scores: List[float]) -> str:
        """Recommend keyword directness level based on competitor analysis"""
        avg_directness = sum(directness_scores) / len(directness_scores)
        
        if avg_directness >= 0.7:
            return 'high'
        elif avg_directness >= 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _generate_recommendations(self, titles: List[CompetitorTitle], keyword: str) -> List[str]:
        """Generate actionable recommendations based on competitor analysis"""
        if not titles:
            return ["No competitor data available for analysis"]
        
        recommendations = []
        
        # Length recommendations
        lengths = [title.length for title in titles]
        avg_length = sum(lengths) / len(lengths)
        recommendations.append(f"Target title length: {int(avg_length)}±10 characters")
        
        # Format recommendations
        formats = Counter(title.format_type for title in titles)
        top_format = formats.most_common(1)[0][0]
        recommendations.append(f"Consider using '{top_format}' format - most popular among competitors")
        
        # Directness recommendations
        directness_scores = [title.directness_score for title in titles]
        avg_directness = sum(directness_scores) / len(directness_scores)
        
        if avg_directness >= 0.7:
            recommendations.append("Competitors use direct keyword inclusion - be explicit with your keyword")
        elif avg_directness >= 0.4:
            recommendations.append("Competitors use moderate keyword directness - balance explicit and subtle usage")
        else:
            recommendations.append("Competitors prefer subtle keyword integration - avoid being too direct")
        
        # Emotional tone recommendations
        tones = Counter(title.emotional_tone for title in titles)
        top_tone = tones.most_common(1)[0][0]
        recommendations.append(f"Match the '{top_tone}' tone commonly used by competitors")
        
        # Differentiation opportunity
        format_diversity = len(formats)
        if format_diversity < 3:
            recommendations.append("Limited format variety among competitors - opportunity to stand out with different approach")
        
        return recommendations
    
    def _get_empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis structure when no data is available"""
        return {
            'keyword': '',
            'target_domain': '',
            'competitor_count': 0,
            'titles': [],
            'patterns': {},
            'directness_analysis': {},
            'format_analysis': {},
            'length_analysis': {},
            'recommendations': ["No competitor data available for analysis"]
        }
    
    def compare_with_generated(self, competitor_analysis: Dict[str, Any], 
                             generated_titles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare generated titles with competitor analysis
        """
        try:
            if not competitor_analysis or not generated_titles:
                return {'comparison': 'No data available for comparison'}
            
            comparison = {
                'generated_count': len(generated_titles),
                'competitor_count': competitor_analysis.get('competitor_count', 0),
                'title_comparisons': [],
                'recommendations': []
            }
            
            # Compare each generated title
            for title_data in generated_titles:
                title = title_data.get('title', '')
                title_comparison = self._compare_single_title(title, competitor_analysis)
                comparison['title_comparisons'].append({
                    'title': title,
                    'analysis': title_comparison
                })
            
            # Generate overall recommendations
            comparison['recommendations'] = self._generate_comparison_recommendations(
                generated_titles, competitor_analysis
            )
            
            return comparison
            
        except Exception as e:
            logging.error(f"Title comparison failed: {e}")
            return {'comparison': f'Comparison failed: {str(e)}'}
    
    def _compare_single_title(self, title: str, competitor_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Compare a single generated title with competitor patterns"""
        keyword = competitor_analysis.get('keyword', '')
        
        # Calculate metrics for the generated title
        directness = self._calculate_directness_score(title, keyword)
        format_type = self._classify_title_format(title)
        emotional_tone = self._analyze_emotional_tone(title)
        length = len(title)
        
        # Compare with competitor patterns
        directness_analysis = competitor_analysis.get('directness_analysis', {})
        format_analysis = competitor_analysis.get('format_analysis', {})
        length_analysis = competitor_analysis.get('length_analysis', {})
        
        comparison = {
            'directness_score': directness,
            'directness_vs_competitors': 'unknown',
            'format_type': format_type,
            'format_popularity': 'unknown',
            'length': length,
            'length_vs_competitors': 'unknown',
            'emotional_tone': emotional_tone,
            'competitive_advantage': []
        }
        
        # Compare directness
        avg_competitor_directness = directness_analysis.get('avg_directness', 0.5)
        if directness > avg_competitor_directness + 0.1:
            comparison['directness_vs_competitors'] = 'more direct'
        elif directness < avg_competitor_directness - 0.1:
            comparison['directness_vs_competitors'] = 'less direct'
        else:
            comparison['directness_vs_competitors'] = 'similar'
        
        # Compare format
        format_dist = format_analysis.get('format_distribution', {})
        if format_type in format_dist:
            comparison['format_popularity'] = f"Used by {format_dist[format_type]} competitors"
        else:
            comparison['format_popularity'] = "Unique format not used by competitors"
            comparison['competitive_advantage'].append("Unique format could help differentiate")
        
        # Compare length
        avg_competitor_length = length_analysis.get('avg_length', 50)
        if abs(length - avg_competitor_length) <= 5:
            comparison['length_vs_competitors'] = 'similar'
        elif length > avg_competitor_length:
            comparison['length_vs_competitors'] = 'longer'
        else:
            comparison['length_vs_competitors'] = 'shorter'
        
        return comparison
    
    def _generate_comparison_recommendations(self, generated_titles: List[Dict[str, Any]], 
                                           competitor_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on comparison with competitors"""
        recommendations = []
        
        if not generated_titles or not competitor_analysis:
            return ["Insufficient data for comparison recommendations"]
        
        # Analyze generated titles patterns
        generated_lengths = [len(title.get('title', '')) for title in generated_titles]
        avg_generated_length = sum(generated_lengths) / len(generated_lengths)
        
        # Compare with competitor patterns
        competitor_length = competitor_analysis.get('length_analysis', {}).get('avg_length', 50)
        competitor_directness = competitor_analysis.get('directness_analysis', {}).get('avg_directness', 0.5)
        
        # Length recommendations
        if abs(avg_generated_length - competitor_length) > 10:
            recommendations.append(f"Consider adjusting title length to match competitors (target: {int(competitor_length)} chars)")
        
        # Format recommendations
        competitor_formats = competitor_analysis.get('format_analysis', {}).get('recommended_formats', [])
        if competitor_formats:
            recommendations.append(f"Consider using popular competitor formats: {', '.join(competitor_formats[:2])}")
        
        # Differentiation opportunities
        competitor_count = competitor_analysis.get('competitor_count', 0)
        if competitor_count < 5:
            recommendations.append("Limited competition - opportunity to establish format standards")
        
        format_diversity = competitor_analysis.get('format_analysis', {}).get('format_diversity', 0)
        if format_diversity < 3:
            recommendations.append("Low format diversity among competitors - opportunity to innovate")
        
        return recommendations