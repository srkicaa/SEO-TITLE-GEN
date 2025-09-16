import json
import logging
from typing import Dict, List, Any, Optional
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import re
import statistics
from database import DatabaseManager

class PatternLearner:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        
    def learn_from_success(self, domain: str, successful_title: str, 
                          anchor_text: str = None, context: Dict = None) -> None:
        """
        Learn patterns from successful title placements
        """
        try:
            # Extract patterns from successful title
            patterns = self._extract_title_patterns(successful_title)
            
            # Update domain patterns in database
            for pattern_type, pattern_data in patterns.items():
                self.db_manager.save_style_pattern(
                    domain=domain,
                    pattern_type=pattern_type,
                    pattern_example=pattern_data['example'],
                    success_rate=1.0  # This was successful
                )
            
            # Update site-wide statistics
            self._update_site_success_metrics(domain, successful_title)
            
            logging.info(f"Learned patterns from successful title for {domain}")
            
        except Exception as e:
            logging.error(f"Failed to learn from success: {e}")
    
    def predict_acceptance(self, domain: str, title: str, 
                          context: Dict = None) -> float:
        """
        Predict probability of webmaster accepting the title
        Based on historical data and learned patterns
        """
        try:
            # Get historical patterns for domain
            patterns = self.get_best_patterns(domain)
            if not patterns:
                return 0.5  # Default probability when no data
            
            # Calculate acceptance probability
            score = 0.0
            factors = 0
            
            # Factor 1: Title length match
            historical_length = patterns.get('avg_length', 55)
            length_diff = abs(len(title) - historical_length)
            length_score = max(0, 1 - (length_diff / 20))  # Normalize to 0-1
            score += length_score
            factors += 1
            
            # Factor 2: Format pattern match
            title_format = self._classify_title_format(title)
            common_formats = patterns.get('common_formats', [])
            if title_format in common_formats:
                score += 0.8
            else:
                score += 0.3
            factors += 1
            
            # Factor 3: Editorial style match
            editorial_style = patterns.get('editorial_style', 'neutral')
            style_match = self._calculate_style_match(title, editorial_style)
            score += style_match
            factors += 1
            
            # Factor 4: Historical acceptance rate
            acceptance_rate = patterns.get('acceptance_rate', 0.5)
            score += acceptance_rate
            factors += 1
            
            # Factor 5: Keyword integration quality
            if context and context.get('anchor_text'):
                integration_score = self._score_keyword_integration(
                    title, context['anchor_text'], patterns
                )
                score += integration_score
                factors += 1
            
            # Calculate final probability
            probability = score / factors if factors > 0 else 0.5
            return min(max(probability, 0.0), 1.0)
            
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            return 0.5
    
    def get_best_patterns(self, domain: str) -> Dict[str, Any]:
        """
        Get the most successful patterns for a domain
        """
        try:
            # Get stored patterns from database
            stored_patterns = self.db_manager.get_site_patterns(domain)
            
            if not stored_patterns:
                return self._get_default_patterns()
            
            # Enhance with recent successful titles
            successful_titles = self._get_successful_titles_for_domain(domain)
            
            if successful_titles:
                # Analyze successful titles for additional patterns
                analyzed_patterns = self._analyze_successful_titles(successful_titles)
                stored_patterns.update(analyzed_patterns)
            
            return stored_patterns
            
        except Exception as e:
            logging.error(f"Failed to get patterns for {domain}: {e}")
            return self._get_default_patterns()
    
    def _extract_title_patterns(self, title: str) -> Dict[str, Dict]:
        """Extract various patterns from a title"""
        patterns = {}
        
        # Format pattern
        format_type = self._classify_title_format(title)
        patterns['format'] = {
            'type': format_type,
            'example': title,
            'length': len(title)
        }
        
        # Structure pattern
        structure = self._analyze_title_structure(title)
        patterns['structure'] = {
            'has_colon': ':' in title,
            'has_question': title.endswith('?'),
            'starts_with_number': title[0].isdigit() if title else False,
            'word_count': len(title.split()),
            'example': title
        }
        
        # Keyword pattern
        keywords = self._extract_keywords(title)
        patterns['keywords'] = {
            'primary_keywords': keywords[:3],
            'keyword_density': len(keywords) / len(title.split()) if title.split() else 0,
            'example': title
        }
        
        # Emotional pattern
        emotional_analysis = self._analyze_emotional_pattern(title)
        patterns['emotional'] = {
            'tone': emotional_analysis['tone'],
            'intensity': emotional_analysis['intensity'],
            'power_words': emotional_analysis['power_words'],
            'example': title
        }
        
        return patterns
    
    def _classify_title_format(self, title: str) -> str:
        """Classify the format type of a title"""
        title_lower = title.lower()
        
        if title.endswith('?'):
            return 'question'
        elif 'how to' in title_lower:
            return 'how-to'
        elif re.search(r'\d+', title):
            return 'numbered'
        elif 'vs' in title_lower or 'versus' in title_lower:
            return 'comparison'
        elif any(word in title_lower for word in ['best', 'top', 'ultimate']):
            return 'superlative'
        elif ':' in title:
            return 'colon-separated'
        elif any(word in title_lower for word in ['guide', 'tutorial', 'tips']):
            return 'instructional'
        else:
            return 'standard'
    
    def _analyze_title_structure(self, title: str) -> Dict[str, Any]:
        """Analyze structural elements of a title"""
        words = title.split()
        
        return {
            'length': len(title),
            'word_count': len(words),
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'has_punctuation': any(char in title for char in '!?:;'),
            'capitalization_style': self._detect_capitalization_style(title),
            'sentence_structure': self._analyze_sentence_structure(title)
        }
    
    def _extract_keywords(self, title: str) -> List[str]:
        """Extract potential keywords from title"""
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', title.lower())
        
        # Filter out common stop words
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'man', 'end', 'few', 'got', 'own', 'say', 'she', 'too', 'use'}
        
        keywords = [word for word in words if word not in stop_words]
        return keywords
    
    def _analyze_emotional_pattern(self, title: str) -> Dict[str, Any]:
        """Analyze emotional elements in title"""
        power_words = {
            'positive': ['amazing', 'incredible', 'ultimate', 'best', 'perfect', 'outstanding', 'excellent'],
            'urgency': ['now', 'today', 'urgent', 'immediately', 'quick', 'fast', 'instant'],
            'curiosity': ['secret', 'hidden', 'unknown', 'mystery', 'surprising', 'shocking'],
            'authority': ['expert', 'professional', 'proven', 'certified', 'guaranteed', 'official']
        }
        
        title_lower = title.lower()
        found_power_words = []
        dominant_tone = 'neutral'
        intensity = 0
        
        for category, words in power_words.items():
            category_count = sum(1 for word in words if word in title_lower)
            if category_count > 0:
                found_power_words.extend([word for word in words if word in title_lower])
                if category_count > intensity:
                    intensity = category_count
                    dominant_tone = category
        
        return {
            'tone': dominant_tone,
            'intensity': intensity,
            'power_words': found_power_words
        }
    
    def _calculate_style_match(self, title: str, editorial_style: str) -> float:
        """Calculate how well title matches editorial style"""
        title_lower = title.lower()
        
        style_indicators = {
            'formal': ['analysis', 'comprehensive', 'research', 'study', 'investigation'],
            'casual': ['awesome', 'cool', 'amazing', 'love', 'hate', 'totally'],
            'listicle': ['top', 'best', 'ways', 'tips', 'things', 'reasons'],
            'news': ['breaking', 'update', 'report', 'announced', 'reveals'],
            'neutral': ['guide', 'how', 'what', 'when', 'where', 'why']
        }
        
        indicators = style_indicators.get(editorial_style, style_indicators['neutral'])
        matches = sum(1 for indicator in indicators if indicator in title_lower)
        
        return min(matches / len(indicators), 1.0)
    
    def _score_keyword_integration(self, title: str, anchor_text: str, 
                                  patterns: Dict[str, Any]) -> float:
        """Score how well keywords are integrated"""
        title_lower = title.lower()
        anchor_lower = anchor_text.lower()
        
        # Check if anchor text is present
        if anchor_lower not in title_lower:
            return 0.0
        
        # Check position (earlier is better)
        position = title_lower.find(anchor_lower)
        position_score = max(0, 1 - (position / len(title)))
        
        # Check natural integration
        words_around = self._get_words_around_anchor(title, anchor_text)
        integration_score = self._evaluate_integration_naturalness(words_around)
        
        # Combine scores
        return (position_score * 0.4 + integration_score * 0.6)
    
    def _get_words_around_anchor(self, title: str, anchor_text: str) -> List[str]:
        """Get words surrounding the anchor text"""
        title_lower = title.lower()
        anchor_lower = anchor_text.lower()
        
        position = title_lower.find(anchor_lower)
        if position == -1:
            return []
        
        # Get 2 words before and after
        words = title.split()
        anchor_words = anchor_text.split()
        
        # Find anchor position in word list
        for i, word in enumerate(words):
            if anchor_lower in word.lower():
                start = max(0, i - 2)
                end = min(len(words), i + len(anchor_words) + 2)
                return words[start:end]
        
        return []
    
    def _evaluate_integration_naturalness(self, surrounding_words: List[str]) -> float:
        """Evaluate how naturally the anchor text is integrated"""
        if not surrounding_words:
            return 0.5
        
        # Simple heuristic: check for natural connectors
        natural_connectors = ['for', 'with', 'in', 'on', 'about', 'of', 'to', 'and']
        connector_count = sum(1 for word in surrounding_words if word.lower() in natural_connectors)
        
        return min(connector_count / len(surrounding_words), 1.0)
    
    def _get_successful_titles_for_domain(self, domain: str) -> List[Dict]:
        """Get successful titles for a domain from database"""
        try:
            # This would query the successful_titles table
            # For now, return empty list
            return []
        except Exception as e:
            logging.error(f"Failed to get successful titles: {e}")
            return []
    
    def _analyze_successful_titles(self, titles: List[Dict]) -> Dict[str, Any]:
        """Analyze patterns from successful titles"""
        if not titles:
            return {}
        
        analysis = {
            'avg_length': statistics.mean([len(title['title']) for title in titles]),
            'common_formats': self._get_common_formats([title['title'] for title in titles]),
            'success_keywords': self._extract_success_keywords(titles),
            'acceptance_rate': len([t for t in titles if t.get('accepted', False)]) / len(titles)
        }
        
        return analysis
    
    def _get_common_formats(self, titles: List[str]) -> List[str]:
        """Get most common formats from title list"""
        formats = [self._classify_title_format(title) for title in titles]
        format_counts = Counter(formats)
        return [fmt for fmt, count in format_counts.most_common(3)]
    
    def _extract_success_keywords(self, titles: List[Dict]) -> List[str]:
        """Extract keywords that appear in successful titles"""
        all_keywords = []
        for title_data in titles:
            keywords = self._extract_keywords(title_data['title'])
            all_keywords.extend(keywords)
        
        keyword_counts = Counter(all_keywords)
        return [keyword for keyword, count in keyword_counts.most_common(10)]
    
    def _detect_capitalization_style(self, title: str) -> str:
        """Detect capitalization style of title"""
        words = title.split()
        if not words:
            return 'unknown'
        
        capitalized_words = sum(1 for word in words if word[0].isupper())
        
        if capitalized_words == len(words):
            return 'title_case'
        elif capitalized_words == 1:
            return 'sentence_case'
        elif capitalized_words > len(words) / 2:
            return 'mixed_case'
        else:
            return 'lower_case'
    
    def _analyze_sentence_structure(self, title: str) -> str:
        """Analyze sentence structure type"""
        if title.endswith('?'):
            return 'interrogative'
        elif title.endswith('!'):
            return 'exclamatory'
        elif ':' in title:
            return 'declarative_colon'
        else:
            return 'declarative'
    
    def _get_default_patterns(self) -> Dict[str, Any]:
        """Return default patterns when no data is available"""
        return {
            'avg_length': 55,
            'common_formats': ['standard', 'how-to', 'superlative'],
            'editorial_style': 'neutral',
            'keyword_directness': 0.5,
            'acceptance_rate': 0.5,
            'emotional_tone': {'positive': 0.3, 'negative': 0.1, 'neutral': 0.6},
            'site_tolerance': {'direct': 0.5, 'subtle': 0.5, 'creative': 0.5}
        }
    
    def update_pattern_success(self, domain: str, pattern_type: str, 
                             pattern_example: str, was_successful: bool) -> None:
        """Update success rate for a specific pattern"""
        try:
            # Get current pattern data
            # Update success rate based on outcome
            # This would interact with the database to update pattern success rates
            logging.info(f"Updated pattern success for {domain}: {pattern_type}")
        except Exception as e:
            logging.error(f"Failed to update pattern success: {e}")
    
    def get_pattern_recommendations(self, domain: str, anchor_text: str) -> List[Dict]:
        """Get pattern recommendations for new title generation"""
        try:
            patterns = self.get_best_patterns(domain)
            
            recommendations = []
            
            # Recommend based on successful formats
            for fmt in patterns.get('common_formats', []):
                rec = {
                    'pattern_type': fmt,
                    'success_rate': patterns.get('acceptance_rate', 0.5),
                    'example_structure': self._get_format_template(fmt),
                    'integration_style': self._recommend_integration_style(patterns, anchor_text)
                }
                recommendations.append(rec)
            
            return recommendations
            
        except Exception as e:
            logging.error(f"Failed to get pattern recommendations: {e}")
            return []
    
    def _get_format_template(self, format_type: str) -> str:
        """Get template for format type"""
        templates = {
            'how-to': 'How to [ACTION] [KEYWORD] in [TIMEFRAME]',
            'question': 'What is the Best [KEYWORD] for [PURPOSE]?',
            'numbered': '[NUMBER] [KEYWORD] [BENEFITS/TIPS/WAYS]',
            'superlative': 'Best [KEYWORD] for [TARGET_AUDIENCE]',
            'comparison': '[KEYWORD A] vs [KEYWORD B]: Complete Guide',
            'colon-separated': '[KEYWORD]: [BENEFIT/DESCRIPTION]'
        }
        return templates.get(format_type, '[KEYWORD] [DESCRIPTION]')
    
    def _recommend_integration_style(self, patterns: Dict, anchor_text: str) -> str:
        """Recommend how to integrate anchor text"""
        directness = patterns.get('keyword_directness', 0.5)
        
        if directness > 0.7:
            return 'direct'  # Use anchor text prominently
        elif directness > 0.3:
            return 'natural'  # Integrate naturally
        else:
            return 'subtle'  # Use subtly or as context
