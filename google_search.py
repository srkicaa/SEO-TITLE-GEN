import requests
import json
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus
import time

class GoogleSearchAnalyzer:
    def __init__(self, api_key: str = None, search_engine_id: str = None):
        self.api_key = api_key
        self.search_engine_id = search_engine_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"

    def search_site_content(self, domain: str, keyword: str, limit: int = 10) -> Dict[str, Any]:
        """
        Search for content on a specific domain using Google Custom Search
        """
        if not self.api_key or not self.search_engine_id:
            return self._get_fallback_results(domain, keyword)

        try:
            # Construct search query
            query = f"site:{domain} {keyword}"

            # Make API request
            params = {
                'key': self.api_key,
                'cx': self.search_engine_id,
                'q': query,
                'num': min(limit, 10),  # Google API max is 10 per request
                'safe': 'off'  # Disable SafeSearch filtering
            }

            # Debug: Log the actual query being sent
            logging.info(f"DEBUG: Google Search Query: '{query}'")
            logging.info(f"DEBUG: Search Engine ID: {self.search_engine_id}")

            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            # Debug: Log response info
            logging.info(f"DEBUG: Response status: {response.status_code}")
            logging.info(f"DEBUG: Total results found: {data.get('searchInformation', {}).get('totalResults', '0')}")
            logging.info(f"DEBUG: Items returned: {len(data.get('items', []))}")

            # Parse results
            results = self._parse_search_results(data, domain, keyword)

            logging.info(f"Found {len(results.get('titles', []))} results for {query}")
            return results

        except Exception as e:
            logging.error(f"Google search failed: {e}")
            return self._get_fallback_results(domain, keyword)

    def _parse_search_results(self, data: Dict, domain: str, keyword: str) -> Dict[str, Any]:
        """Parse Google search API response"""

        results = {
            'domain': domain,
            'keyword': keyword,
            'total_results': data.get('searchInformation', {}).get('totalResults', '0'),
            'titles': [],
            'content_analysis': {},
            'has_content': False
        }

        items = data.get('items', [])

        if not items:
            return results

        results['has_content'] = True

        # Extract titles and analyze patterns
        titles = []
        for item in items:
            title_data = {
                'title': item.get('title', ''),
                'url': item.get('link', ''),
                'snippet': item.get('snippet', ''),
                'length': len(item.get('title', '')),
            }
            titles.append(title_data)

        results['titles'] = titles

        # Analyze content patterns
        results['content_analysis'] = self._analyze_content_patterns(titles, keyword)

        return results

    def _analyze_content_patterns(self, titles: List[Dict], keyword: str) -> Dict[str, Any]:
        """Analyze patterns in the found titles"""

        if not titles:
            return {}

        analysis = {
            'avg_title_length': sum(len(t['title']) for t in titles) / len(titles),
            'keyword_positions': [],
            'common_formats': [],
            'content_themes': [],
            'title_styles': []
        }

        # Analyze each title
        for title_data in titles:
            title = title_data['title']
            title_lower = title.lower()
            keyword_lower = keyword.lower()

            # Keyword position analysis
            if keyword_lower in title_lower:
                position = title_lower.find(keyword_lower)
                if position <= len(title) * 0.25:
                    analysis['keyword_positions'].append('early')
                elif position <= len(title) * 0.75:
                    analysis['keyword_positions'].append('middle')
                else:
                    analysis['keyword_positions'].append('late')

            # Format detection
            if title.endswith('?'):
                analysis['common_formats'].append('question')
            elif 'how to' in title_lower:
                analysis['common_formats'].append('how-to')
            elif any(char.isdigit() for char in title):
                analysis['common_formats'].append('numbered')
            elif ':' in title:
                analysis['common_formats'].append('colon-separated')
            else:
                analysis['common_formats'].append('standard')

            # Style detection
            if any(word in title_lower for word in ['best', 'top', 'ultimate']):
                analysis['title_styles'].append('superlative')
            elif any(word in title_lower for word in ['guide', 'tutorial', 'tips']):
                analysis['title_styles'].append('instructional')
            elif any(word in title_lower for word in ['review', 'analysis']):
                analysis['title_styles'].append('analytical')
            else:
                analysis['title_styles'].append('neutral')

        # Count frequencies
        from collections import Counter
        analysis['keyword_positions'] = dict(Counter(analysis['keyword_positions']))
        analysis['common_formats'] = dict(Counter(analysis['common_formats']))
        analysis['title_styles'] = dict(Counter(analysis['title_styles']))

        return analysis

    def detect_content_type(self, domain: str, content_keywords: List[str]) -> Dict[str, Any]:
        """
        Detect what type of content a domain covers
        """
        content_detection = {
            'domain': domain,
            'content_types': {},
            'risk_assessment': 'unknown',
            'coverage_analysis': {}
        }

        if not self.api_key:
            return content_detection

        for keyword in content_keywords:
            try:
                results = self.search_site_content(domain, keyword, limit=5)
                has_content = results.get('has_content', False)
                total_results = int(results.get('total_results', '0'))

                content_detection['content_types'][keyword] = {
                    'has_content': has_content,
                    'result_count': total_results,
                    'sample_titles': [t['title'] for t in results.get('titles', [])[:3]]
                }

                # Small delay to respect rate limits
                time.sleep(0.5)

            except Exception as e:
                logging.error(f"Content detection failed for {keyword}: {e}")
                content_detection['content_types'][keyword] = {
                    'has_content': False,
                    'result_count': 0,
                    'error': str(e)
                }

        # Risk assessment
        content_detection['risk_assessment'] = self._assess_content_risk(content_detection['content_types'])

        return content_detection

    def _assess_content_risk(self, content_types: Dict) -> str:
        """Assess risk level based on content types found"""

        high_risk_keywords = ['casino', 'gambling', 'betting', 'poker']
        medium_risk_keywords = ['crypto', 'bitcoin', 'trading', 'investment']

        high_risk_count = 0
        medium_risk_count = 0

        for keyword, data in content_types.items():
            if data.get('has_content', False):
                if keyword.lower() in high_risk_keywords:
                    high_risk_count += 1
                elif keyword.lower() in medium_risk_keywords:
                    medium_risk_count += 1

        if high_risk_count >= 2:
            return 'high'
        elif high_risk_count >= 1 or medium_risk_count >= 3:
            return 'medium'
        elif medium_risk_count >= 1:
            return 'low'
        else:
            return 'minimal'

    def _get_fallback_results(self, domain: str, keyword: str) -> Dict[str, Any]:
        """Fallback when API is not available"""
        return {
            'domain': domain,
            'keyword': keyword,
            'total_results': '0',
            'titles': [],
            'content_analysis': {},
            'has_content': False,
            'error': 'Google Search API not configured'
        }

    def get_title_intelligence(self, domain: str, keyword: str) -> Dict[str, Any]:
        """
        Get comprehensive title intelligence for a domain/keyword combination
        """
        # Search for existing content
        search_results = self.search_site_content(domain, keyword)

        if not search_results.get('has_content'):
            return {
                'has_existing_content': False,
                'content_gap': True,
                'opportunity_level': 'high',
                'recommendations': [
                    f"No existing content found for '{keyword}' on {domain}",
                    "This represents a potential content gap opportunity",
                    "Consider creating high-quality content on this topic"
                ]
            }

        # Analyze existing content
        titles = search_results.get('titles', [])
        analysis = search_results.get('content_analysis', {})

        intelligence = {
            'has_existing_content': True,
            'content_gap': False,
            'existing_titles': [t['title'] for t in titles[:5]],
            'title_patterns': analysis.get('common_formats', {}),
            'style_preferences': analysis.get('title_styles', {}),
            'avg_title_length': analysis.get('avg_title_length', 60),
            'opportunity_level': self._assess_opportunity_level(len(titles)),
            'recommendations': self._generate_title_recommendations(analysis, domain, keyword)
        }

        return intelligence

    def _assess_opportunity_level(self, existing_content_count: int) -> str:
        """Assess opportunity level based on existing content"""
        if existing_content_count == 0:
            return 'high'
        elif existing_content_count <= 3:
            return 'medium'
        elif existing_content_count <= 7:
            return 'low'
        else:
            return 'saturated'

    def _generate_title_recommendations(self, analysis: Dict, domain: str, keyword: str) -> List[str]:
        """Generate recommendations based on title analysis"""
        recommendations = []

        # Format recommendations
        common_formats = analysis.get('common_formats', {})
        if common_formats:
            top_format = max(common_formats, key=common_formats.get)
            recommendations.append(f"Most common format on {domain}: {top_format}")

        # Length recommendations
        avg_length = analysis.get('avg_title_length', 60)
        recommendations.append(f"Target title length: {int(avg_length)}±5 characters")

        # Style recommendations
        title_styles = analysis.get('title_styles', {})
        if title_styles:
            top_style = max(title_styles, key=title_styles.get)
            recommendations.append(f"Preferred style: {top_style}")

        # Keyword positioning
        keyword_positions = analysis.get('keyword_positions', {})
        if keyword_positions:
            top_position = max(keyword_positions, key=keyword_positions.get)
            recommendations.append(f"Place '{keyword}' in {top_position} position")

        return recommendations