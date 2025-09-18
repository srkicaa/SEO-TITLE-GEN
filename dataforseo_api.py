import requests
import json
import logging
import base64
import time
from typing import List, Dict, Any, Optional
from collections import Counter

class DataForSEOAnalyzer:
    def __init__(self, username: str = None, password: str = None):
        self.username = username
        self.password = password
        self.base_url = "https://api.dataforseo.com/v3"
        self.headers = {
            'Authorization': f'Basic {self._get_auth_string()}' if username and password else None,
            'Content-Type': 'application/json'
        }

    def _get_auth_string(self) -> str:
        if not self.username or not self.password:
            return ""
        credentials = f"{self.username}:{self.password}"
        return base64.b64encode(credentials.encode()).decode()

    def get_serp_titles(self, keyword: str, location: str = "United States", language: str = "English", limit: int = 10) -> Dict[str, Any]:
        """
        Get SERP titles for a specific keyword using DataForSEO API
        """
        if not self.username or not self.password:
            return self._get_fallback_results(keyword)

        try:
            # Prepare request for Google organic results
            endpoint = f"{self.base_url}/serp/google/organic/live/advanced"

            data = [{
                "keyword": keyword,
                "location_name": location,
                "language_name": language,
                "device": "desktop",
                "os": "windows",
                "depth": min(limit, 100)  # DataForSEO supports up to 100 results
            }]

            logging.info(f"DEBUG: DataForSEO SERP Query: '{keyword}' in {location}")

            response = requests.post(
                endpoint,
                headers=self.headers,
                json=data,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()

                logging.info(f"DEBUG: DataForSEO Response status: {response.status_code}")

                if result and 'tasks' in result and result['tasks']:
                    task_result = result['tasks'][0]
                    if task_result['status_code'] == 20000:  # Success
                        return self._parse_serp_results(task_result['result'][0], keyword)
                    else:
                        logging.error(f"DataForSEO API error: {task_result.get('status_message', 'Unknown error')}")
                        return self._get_fallback_results(keyword)
                else:
                    logging.error("DataForSEO API returned empty result")
                    return self._get_fallback_results(keyword)
            else:
                logging.error(f"DataForSEO API HTTP error: {response.status_code}")
                return self._get_fallback_results(keyword)

        except Exception as e:
            logging.error(f"DataForSEO SERP request failed: {e}")
            return self._get_fallback_results(keyword)

    def _parse_serp_results(self, data: Dict, keyword: str) -> Dict[str, Any]:
        """Parse DataForSEO SERP API response"""

        results = {
            'keyword': keyword,
            'total_results': data.get('total_count', 0),
            'titles': [],
            'content_analysis': {},
            'has_content': False,
            'serp_features': []
        }

        # Extract organic results
        items = data.get('items', [])
        organic_results = [item for item in items if item.get('type') == 'organic']

        if not organic_results:
            return results

        results['has_content'] = True

        # Process organic search results
        titles = []
        for item in organic_results:
            title_data = {
                'title': item.get('title', ''),
                'url': item.get('url', ''),
                'description': item.get('description', ''),
                'position': item.get('rank_group', 0),
                'domain': item.get('domain', ''),
                'length': len(item.get('title', '')),
            }
            titles.append(title_data)

        results['titles'] = titles[:20]  # Limit to top 20 for analysis

        # Extract SERP features for additional context
        serp_features = []
        for item in items:
            if item.get('type') != 'organic':
                serp_features.append({
                    'type': item.get('type'),
                    'title': item.get('title', ''),
                    'position': item.get('rank_group', 0)
                })

        results['serp_features'] = serp_features

        # Analyze content patterns
        results['content_analysis'] = self._analyze_title_patterns(titles, keyword)

        logging.info(f"Found {len(titles)} organic results and {len(serp_features)} SERP features for '{keyword}'")
        return results

    def _analyze_title_patterns(self, titles: List[Dict], keyword: str) -> Dict[str, Any]:
        """Analyze patterns in SERP titles"""

        if not titles:
            return {}

        analysis = {
            'avg_title_length': sum(len(t['title']) for t in titles) / len(titles),
            'keyword_positions': [],
            'title_formats': [],
            'content_themes': [],
            'title_styles': [],
            'domain_diversity': len(set(t.get('domain', '') for t in titles)),
            'top_domains': []
        }

        # Analyze each title
        for title_data in titles:
            title = title_data['title']
            title_lower = title.lower()
            keyword_lower = keyword.lower()

            # Keyword position analysis
            if keyword_lower in title_lower:
                position = title_lower.find(keyword_lower)
                char_position = position / len(title) if len(title) > 0 else 0

                if char_position <= 0.25:
                    analysis['keyword_positions'].append('beginning')
                elif char_position <= 0.75:
                    analysis['keyword_positions'].append('middle')
                else:
                    analysis['keyword_positions'].append('end')
            else:
                # Check for partial keyword matches
                keyword_words = keyword_lower.split()
                if len(keyword_words) > 1:
                    matches = sum(1 for word in keyword_words if word in title_lower)
                    if matches > 0:
                        analysis['keyword_positions'].append('partial')

            # Format detection (enhanced)
            if title.endswith('?'):
                analysis['title_formats'].append('question')
            elif title.count('|') > 0:
                analysis['title_formats'].append('pipe_separated')
            elif title.count('-') >= 2:
                analysis['title_formats'].append('dash_separated')
            elif ':' in title:
                analysis['title_formats'].append('colon_format')
            elif any(char.isdigit() for char in title):
                analysis['title_formats'].append('with_numbers')
            elif len(title.split()) <= 4:
                analysis['title_formats'].append('short_form')
            else:
                analysis['title_formats'].append('standard')

            # Style detection (enhanced)
            title_words = title_lower.split()

            if any(word in title_words for word in ['best', 'top', 'ultimate', '#1', 'leading']):
                analysis['title_styles'].append('superlative')
            elif any(word in title_words for word in ['how', 'guide', 'tutorial', 'tips', 'ways']):
                analysis['title_styles'].append('instructional')
            elif any(word in title_words for word in ['review', 'reviews', 'analysis', 'comparison']):
                analysis['title_styles'].append('review')
            elif any(word in title_words for word in ['free', 'cheap', 'discount', 'deal']):
                analysis['title_styles'].append('promotional')
            elif any(word in title_words for word in ['2024', '2025', 'new', 'latest', 'updated']):
                analysis['title_styles'].append('current')
            else:
                analysis['title_styles'].append('informational')

        # Count frequencies and get top domains
        analysis['keyword_positions'] = dict(Counter(analysis['keyword_positions']).most_common())
        analysis['title_formats'] = dict(Counter(analysis['title_formats']).most_common())
        analysis['title_styles'] = dict(Counter(analysis['title_styles']).most_common())

        # Top domains analysis
        domain_counts = Counter(t.get('domain', 'unknown') for t in titles)
        analysis['top_domains'] = dict(domain_counts.most_common(5))

        return analysis

    def get_competitor_intelligence(self, keyword: str, target_domain: str = None, location: str = "United States") -> Dict[str, Any]:
        """
        Get comprehensive competitor intelligence for a keyword
        """
        serp_data = self.get_serp_titles(keyword, location)

        if not serp_data.get('has_content'):
            return {
                'keyword': keyword,
                'has_competition': False,
                'opportunity_level': 'high',
                'recommendations': [
                    f"No organic results found for '{keyword}'",
                    "This may represent a low-competition opportunity",
                    "Consider targeting this keyword with quality content"
                ]
            }

        titles = serp_data.get('titles', [])
        analysis = serp_data.get('content_analysis', {})

        # Competition analysis
        competition_level = self._assess_competition_level(titles, serp_data.get('serp_features', []))

        intelligence = {
            'keyword': keyword,
            'has_competition': True,
            'total_results': serp_data.get('total_results', 0),
            'organic_results_count': len(titles),
            'competition_level': competition_level,
            'top_competitor_titles': [t['title'] for t in titles[:5]],
            'title_patterns': analysis.get('title_formats', {}),
            'style_preferences': analysis.get('title_styles', {}),
            'keyword_positioning': analysis.get('keyword_positions', {}),
            'avg_title_length': int(analysis.get('avg_title_length', 60)),
            'domain_diversity': analysis.get('domain_diversity', 0),
            'top_domains': analysis.get('top_domains', {}),
            'serp_features': [f['type'] for f in serp_data.get('serp_features', [])],
            'opportunity_level': self._assess_opportunity_level(competition_level, titles),
            'recommendations': self._generate_title_recommendations(analysis, keyword, target_domain)
        }

        # Target domain analysis if provided
        if target_domain:
            intelligence['target_domain_presence'] = self._analyze_domain_presence(titles, target_domain)

        return intelligence

    def _assess_competition_level(self, titles: List[Dict], serp_features: List[Dict]) -> str:
        """Assess competition level based on SERP analysis"""

        # Factors indicating high competition
        high_competition_indicators = 0

        # Domain authority indicators (simplified)
        authority_domains = ['wikipedia.org', 'amazon.com', 'youtube.com', 'facebook.com', 'linkedin.com']
        authority_count = sum(1 for t in titles if any(domain in t.get('domain', '') for domain in authority_domains))

        if authority_count >= 3:
            high_competition_indicators += 2
        elif authority_count >= 1:
            high_competition_indicators += 1

        # SERP features indicate competitive keywords
        competitive_features = ['knowledge_graph', 'featured_snippet', 'local_pack', 'shopping', 'ads']
        feature_count = sum(1 for f in serp_features if f.get('type') in competitive_features)

        if feature_count >= 3:
            high_competition_indicators += 2
        elif feature_count >= 1:
            high_competition_indicators += 1

        # Title optimization level
        optimized_titles = sum(1 for t in titles if len(t.get('title', '')) > 50 and '|' in t.get('title', ''))
        if optimized_titles >= 5:
            high_competition_indicators += 1

        # Determine competition level
        if high_competition_indicators >= 4:
            return 'very_high'
        elif high_competition_indicators >= 3:
            return 'high'
        elif high_competition_indicators >= 2:
            return 'medium'
        elif high_competition_indicators >= 1:
            return 'low'
        else:
            return 'very_low'

    def _assess_opportunity_level(self, competition_level: str, titles: List[Dict]) -> str:
        """Assess opportunity level based on competition"""

        competition_map = {
            'very_low': 'excellent',
            'low': 'high',
            'medium': 'moderate',
            'high': 'challenging',
            'very_high': 'difficult'
        }

        return competition_map.get(competition_level, 'moderate')

    def _analyze_domain_presence(self, titles: List[Dict], target_domain: str) -> Dict[str, Any]:
        """Analyze if target domain is present in results"""

        domain_results = [t for t in titles if target_domain.lower() in t.get('domain', '').lower()]

        return {
            'is_ranking': len(domain_results) > 0,
            'positions': [t.get('position', 0) for t in domain_results],
            'titles': [t.get('title', '') for t in domain_results],
            'best_position': min([t.get('position', 100) for t in domain_results]) if domain_results else None
        }

    def _generate_title_recommendations(self, analysis: Dict, keyword: str, target_domain: str = None) -> List[str]:
        """Generate actionable title recommendations"""

        recommendations = []

        # Format recommendations
        top_formats = analysis.get('title_formats', {})
        if top_formats:
            top_format = max(top_formats, key=top_formats.get)
            format_percentage = (top_formats[top_format] / sum(top_formats.values())) * 100
            recommendations.append(f"Most common format: {top_format} ({format_percentage:.0f}% of results)")

        # Style recommendations
        top_styles = analysis.get('title_styles', {})
        if top_styles:
            top_style = max(top_styles, key=top_styles.get)
            recommendations.append(f"Dominant style: {top_style}")

        # Length recommendations
        avg_length = analysis.get('avg_title_length', 60)
        recommendations.append(f"Target title length: {int(avg_length)}±10 characters")

        # Keyword positioning
        keyword_positions = analysis.get('keyword_positions', {})
        if keyword_positions:
            top_position = max(keyword_positions, key=keyword_positions.get)
            recommendations.append(f"Best keyword position: {top_position} of title")

        # Domain diversity insights
        domain_diversity = analysis.get('domain_diversity', 0)
        if domain_diversity < 5:
            recommendations.append("Low domain diversity - good opportunity for new entrants")
        elif domain_diversity > 8:
            recommendations.append("High domain diversity - competitive landscape")

        return recommendations

    def _get_fallback_results(self, keyword: str) -> Dict[str, Any]:
        """Fallback when API is not available"""
        return {
            'keyword': keyword,
            'total_results': 0,
            'titles': [],
            'content_analysis': {},
            'has_content': False,
            'serp_features': [],
            'error': 'DataForSEO API not configured or unavailable'
        }

    def test_connection(self) -> Dict[str, Any]:
        """Test DataForSEO API connection"""

        if not self.username or not self.password:
            return {
                'success': False,
                'error': 'Username and password not configured'
            }

        try:
            # Test with a simple query
            test_result = self.get_serp_titles("test query", limit=1)

            if test_result.get('error'):
                return {
                    'success': False,
                    'error': test_result['error']
                }
            else:
                return {
                    'success': True,
                    'message': 'DataForSEO API connection successful'
                }

        except Exception as e:
            return {
                'success': False,
                'error': f'Connection test failed: {str(e)}'
            }