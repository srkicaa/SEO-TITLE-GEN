import requests
import json
import time
import os
import hashlib
import re
import pandas as pd
from typing import List, Optional, Dict, Callable, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict, Counter
import sqlite3

class DataForSEOSimple:
    """
    DataForSEO integration for content research and competitive analysis.

    Primary use case: Site-specific content research to understand editorial patterns
    before generating new titles.

    Example workflow:
    1. Research existing content: site:mopoga.net casino
    2. Analyze editorial patterns and content types
    3. Generate titles that match the site's style

    This helps content creators understand a domain's editorial voice and content
    strategy before creating new content.
    """
    def __init__(self, login: str, password: str, cache_db_path: str = "dataforseo_cache.db"):
        self.login = login
        self.password = password
        self.base_url = "https://api.dataforseo.com/"
        self.cache = {}
        self.cache_duration = timedelta(hours=24)  # Cache for 24 hours
        self.cache_db_path = cache_db_path

        # Cost tracking
        self.cost_per_request = 0.003  # $0.003 per SERP request
        self.total_cost = 0.0
        self.request_count = 0

        # Progress callback for UI updates
        self.progress_callback = None
        self.current_operation = ""

        # Thread safety
        self.cache_lock = threading.Lock()

        self.session = requests.Session()
        self.session.auth = (self.login, self.password)
        self.session.headers.update({"Content-Type": "application/json"})

        # Initialize persistent cache
        self._init_cache_db()

    def _init_cache_db(self):
        """Initialize persistent cache database"""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS research_cache (
                    cache_key TEXT PRIMARY KEY,
                    keyword TEXT,
                    domain TEXT,
                    titles TEXT,
                    editorial_patterns TEXT,
                    timestamp TEXT,
                    cost_tracked REAL DEFAULT 0.0
                )
            ''')
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Warning: Could not initialize cache database: {e}")

    def _get_cache_key(self, keyword: str) -> str:
        """Generate cache key for keyword"""
        return hashlib.md5(keyword.lower().encode()).hexdigest()

    def _get_cached_research(self, keyword: str) -> Optional[Dict]:
        """Get research from persistent cache"""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT titles, editorial_patterns, timestamp FROM research_cache WHERE cache_key = ?",
                (self._get_cache_key(keyword),)
            )
            result = cursor.fetchone()
            conn.close()

            if result:
                cache_time = datetime.fromisoformat(result[2])
                if datetime.now() - cache_time < self.cache_duration:
                    return {
                        'titles': json.loads(result[0]) if result[0] else [],
                        'editorial_patterns': json.loads(result[1]) if result[1] else {},
                        'timestamp': result[2]
                    }
            return None
        except Exception as e:
            print(f"Cache read error: {e}")
            return None

    def _save_to_cache(self, keyword: str, domain: str, titles: List[str], editorial_patterns: Dict, cost: float = 0.0):
        """Save research to persistent cache"""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO research_cache
                (cache_key, keyword, domain, titles, editorial_patterns, timestamp, cost_tracked)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                self._get_cache_key(keyword),
                keyword,
                domain,
                json.dumps(titles),
                json.dumps(editorial_patterns),
                datetime.now().isoformat(),
                cost
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Cache save error: {e}")

    def set_progress_callback(self, callback: Callable[[str, float], None]):
        """Set progress callback for UI updates"""
        self.progress_callback = callback

    def _update_progress(self, message: str, progress: float = 0.0):
        """Update progress if callback is set"""
        if self.progress_callback:
            self.progress_callback(message, progress)

    def get_cost_summary(self) -> Dict:
        """Get cost tracking summary"""
        return {
            'total_requests': self.request_count,
            'total_cost': self.total_cost,
            'cost_per_request': self.cost_per_request,
            'last_operation': self.current_operation
        }

    def _is_cache_valid(self, cache_entry: dict) -> bool:
        """Check if cache entry is still valid"""
        if 'timestamp' not in cache_entry:
            return False
        cache_time = datetime.fromisoformat(cache_entry['timestamp'])
        return datetime.now() - cache_time < self.cache_duration

    def get_serp_titles(self, keyword: str) -> List[str]:
        """Get SERP titles for keyword with persistent caching and Task Handed retry logic"""
        self.current_operation = f"SERP search: {keyword[:30]}..."
        self._update_progress(f"Searching for '{keyword}'", 0.1)

        # Check persistent cache first
        cached_result = self._get_cached_research(keyword)
        if cached_result:
            print(f"Using cached results for '{keyword}'")
            self._update_progress(f"Using cached data for '{keyword}'", 1.0)
            return cached_result['titles']

        print(f"Fetching fresh SERP data for '{keyword}'...")
        self._update_progress(f"Fetching SERP data for '{keyword}'", 0.3)

        # Retry logic for Task Handed and retryable errors
        max_retries = 3
        for retry_attempt in range(max_retries):
            try:
                if retry_attempt > 0:
                    print(f"Retry attempt {retry_attempt + 1}/{max_retries} for '{keyword}'")
                    self._update_progress(f"Retrying '{keyword}' (attempt {retry_attempt + 1})", 0.2)

                # Step 1: Post task
                task_id = self._post_task(keyword)
                if not task_id:
                    if retry_attempt < max_retries - 1:
                        print(f"Task posting failed, retrying in 5 seconds...")
                        time.sleep(5)
                        continue
                    return []

                self._update_progress(f"Processing SERP results for '{keyword}'", 0.6)

                # Step 2: Allow task to initialize in DataForSEO system
                init_delay = 3 + (retry_attempt * 2)  # Longer delays on retries
                print(f"Task {task_id} posted, waiting {init_delay} seconds for initialization...")
                time.sleep(init_delay)

                # Step 3: Get results
                titles = self._get_task_results(task_id)

                # If we got results, success!
                if titles:
                    self.request_count += 1
                    self.total_cost += self.cost_per_request

                    # Save to persistent cache with editorial patterns
                    editorial_patterns = self.analyze_editorial_patterns(titles, keyword)
                    self._save_to_cache(keyword, "", titles, editorial_patterns, self.cost_per_request)

                    self._update_progress(f"Completed SERP search for '{keyword}'", 1.0)
                    return titles

                # No results - check if we should retry
                if retry_attempt < max_retries - 1:
                    retry_delay = 10 + (retry_attempt * 5)  # Increasing delays
                    print(f"No results obtained, retrying in {retry_delay} seconds...")
                    self._update_progress(f"Retrying '{keyword}' in {retry_delay}s", 0.4)
                    time.sleep(retry_delay)
                    continue

            except Exception as e:
                print(f"Error during attempt {retry_attempt + 1}: {e}")
                if retry_attempt < max_retries - 1:
                    retry_delay = 10 + (retry_attempt * 5)
                    print(f"Exception occurred, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    print(f"All retry attempts failed for '{keyword}'")
                    break

        # All attempts failed
        self._update_progress(f"Failed to get results for '{keyword}' after {max_retries} attempts", 0)
        return []

    def _post_task(self, keyword: str) -> Optional[str]:
        """Post SERP task - cheapest settings with enhanced error handling"""
        url = self.base_url + "v3/serp/google/organic/task_post"

        # Minimal, cost-effective settings
        data = [{
            "keyword": keyword,
            "location_name": "United States",
            "language_code": "en",
            "depth": 10,  # Only 10 results to minimize cost
            "tag": f"title_check_{keyword.replace(' ', '_')[:20]}"
        }]

        max_retries = 3
        retry_delay = 1  # Start with 1 second

        for attempt in range(max_retries):
            try:
                response = self.session.post(url, data=json.dumps(data), timeout=30)
                response.raise_for_status()  # Raise exception for HTTP errors
                result = response.json()

                if result.get("status_code") == 20000 and result.get("tasks_count") > 0:
                    task_id = result["tasks"][0]["id"]
                    print(f"Task posted: {task_id}")
                    return task_id
                else:
                    error_msg = result.get('status_message', 'Unknown error')
                    print(f"Task post failed: {error_msg}")

                    # Check for specific error types
                    if "insufficient funds" in error_msg.lower():
                        raise Exception(f"DataForSEO API error: {error_msg} - Please check your account balance")
                    elif "authentication" in error_msg.lower():
                        raise Exception(f"DataForSEO API error: {error_msg} - Please check your credentials")
                    elif attempt < max_retries - 1:  # Retry on other errors
                        print(f"Retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        return None

            except requests.exceptions.Timeout as e:
                print(f"Request timeout on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    raise Exception("DataForSEO API timeout after multiple attempts")

            except requests.exceptions.ConnectionError as e:
                print(f"Connection error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    raise Exception("DataForSEO API connection failed after multiple attempts")

            except requests.exceptions.RequestException as e:
                print(f"Request error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    raise Exception(f"DataForSEO API request failed: {str(e)}")

            except (ValueError, KeyError) as e:
                print(f"Invalid response format: {e}")
                raise Exception(f"DataForSEO API returned invalid response: {str(e)}")

            except Exception as e:
                print(f"Unexpected error posting task: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    raise Exception(f"DataForSEO API error: {str(e)}")

        return None

    def _get_task_results(self, task_id: str) -> List[str]:
        """Get task results and extract only titles with enhanced error handling and immediate task retrieval"""
        url = self.base_url + f"v3/serp/google/organic/task_get/regular/{task_id}"

        # Enhanced polling strategy with shorter intervals initially, then longer
        max_attempts = 30  # Increased attempts
        base_poll_interval = 2  # Start with 2 seconds
        max_poll_interval = 15  # Max 15 seconds between polls

        print(f"Polling task {task_id} for results...")

        # Status codes that indicate task is still processing
        pending_status_codes = {20100, 20200, 20201, 20300, 20400, 20401}

        for attempt in range(max_attempts):
            try:
                # Calculate poll interval with exponential backoff (but capped)
                poll_interval = min(base_poll_interval * (1.2 ** attempt), max_poll_interval)

                print(f"Attempt {attempt + 1}/{max_attempts}: Checking task status...")
                response = self.session.get(url, timeout=30)

                if response.status_code == 404:
                    print(f"Task {task_id} not found (404) - task may have been processed or expired")
                    if attempt < 3:  # Try a few more times for 404s initially
                        time.sleep(2)
                        continue
                    else:
                        print(f"Task consistently returning 404 - giving up")
                        return []

                response.raise_for_status()
                result = response.json()

                print(f"API Response status: {result.get('status_code')} - {result.get('status_message', 'No message')}")

                # Check API-level errors
                if result.get("status_code") != 20000:
                    api_error = result.get("status_message", "Unknown API error")
                    print(f"API error: {api_error}")

                    if "insufficient funds" in api_error.lower():
                        raise Exception(f"DataForSEO API error: {api_error}")
                    if "authentication" in api_error.lower():
                        raise Exception(f"DataForSEO API error: {api_error}")
                    if "not found" in api_error.lower():
                        print(f"Task not found in API - may have expired or been processed")
                        return []

                    # For other API errors just stop trying
                    print(f"Stopping due to API error: {api_error}")
                    return []

                tasks = result.get("tasks")
                if not tasks:
                    print(f"No tasks in response - waiting {poll_interval:.1f}s...")
                    if attempt < max_attempts - 1:
                        time.sleep(poll_interval)
                        continue
                    print("No tasks found after maximum attempts")
                    return []

                task_info = tasks[0]
                task_status_code = task_info.get("status_code")
                task_message = task_info.get("status_message", "")
                task_result = task_info.get("result")

                print(f"Task status: {task_status_code} - {task_message}")

                # Task completed successfully
                if task_status_code == 20000 and task_result:
                    print("Task completed successfully - extracting titles...")
                    titles = []

                    # Extract titles from the result structure
                    for item in task_result:
                        if not isinstance(item, dict) or "items" not in item:
                            continue

                        items = item["items"]
                        if not isinstance(items, list):
                            continue

                        for serp_item in items:
                            if (isinstance(serp_item, dict) and
                                serp_item.get("type") == "organic" and
                                serp_item.get("title")):

                                title = serp_item["title"].strip()
                                if title and len(title) > 10:  # Reasonable minimum length
                                    titles.append(title)

                    print(f"Successfully extracted {len(titles)} titles")
                    return titles[:10]  # Return max 10 titles

                # Task still processing
                if (task_status_code in pending_status_codes or
                    task_result in (None, [], {}) or
                    not task_result):

                    remaining_attempts = max_attempts - attempt - 1
                    print(f"Task still processing, waiting {poll_interval:.1f}s (attempt {attempt + 1}/{max_attempts}, {remaining_attempts} remaining)")

                    if attempt < max_attempts - 1:
                        time.sleep(poll_interval)
                        continue
                    else:
                        print("Maximum polling attempts reached")
                        break

                # Handle specific error codes first
                if task_status_code == 40601:
                    # Task Handed - received but not yet enqueued
                    print(f"Task handed (40601) - waiting longer for queue processing...")
                    if attempt < max_attempts - 1:
                        # Use longer delay for Task Handed scenarios
                        extended_delay = min(poll_interval * 3, 30)  # Triple delay, max 30s
                        print(f"Using extended delay of {extended_delay:.1f}s for Task Handed")
                        time.sleep(extended_delay)
                        continue
                    else:
                        print("Task Handed - maximum attempts reached")
                        return []

                elif task_status_code == 40602:
                    # Task in Queue - being processed
                    print(f"Task in queue (40602) - continuing to wait...")
                    if attempt < max_attempts - 1:
                        time.sleep(poll_interval)
                        continue
                    else:
                        print("Task in queue - maximum attempts reached")
                        return []

                elif task_status_code == 40103:
                    # Task execution failed - may be retryable
                    print(f"Task execution failed (40103): {task_message}")
                    return []  # Return empty to trigger retry at higher level

                # Task failed or completed with error
                elif task_status_code != 20000:
                    print(f"Task failed: {task_message or 'Unknown task error'} (code: {task_status_code})")
                    return []

                # Unexpected case - task says success but no results
                print(f"Unexpected task state: status={task_status_code}, result={type(task_result)}")
                return []

            except requests.exceptions.Timeout as e:
                print(f"Request timeout (attempt {attempt + 1}): {e}")
                if attempt < max_attempts - 1:
                    time.sleep(poll_interval)
                    continue
                else:
                    raise Exception("DataForSEO API timeout during result polling")

            except requests.exceptions.ConnectionError as e:
                print(f"Connection error (attempt {attempt + 1}): {e}")
                if attempt < max_attempts - 1:
                    time.sleep(poll_interval)
                    continue
                else:
                    raise Exception("DataForSEO API connection failed during result polling")

            except requests.exceptions.RequestException as e:
                print(f"Request error (attempt {attempt + 1}): {e}")
                if attempt < max_attempts - 1:
                    time.sleep(poll_interval)
                    continue
                else:
                    raise Exception(f"DataForSEO API request failed: {str(e)}")

            except (ValueError, KeyError) as e:
                print(f"Invalid response format (attempt {attempt + 1}): {e}")
                if attempt < max_attempts - 1:
                    time.sleep(poll_interval)
                    continue
                else:
                    raise Exception(f"DataForSEO API returned invalid response: {str(e)}")

            except Exception as e:
                print(f"Unexpected error (attempt {attempt + 1}): {e}")
                if attempt < max_attempts - 1:
                    time.sleep(poll_interval)
                    continue
                else:
                    raise Exception(f"DataForSEO API error: {str(e)}")

        print(f"Task polling exhausted after {max_attempts} attempts")
        raise Exception("DataForSEO task timed out - no results returned within expected time")

    def analyze_editorial_patterns(self, titles: List[str], keyword: str) -> Dict:
        """
        Analyze editorial patterns from title data for AI context.

        This function extracts comprehensive patterns that can help AI understand:
        - Writing style and tone
        - Content structure preferences
        - Keyword usage patterns
        - Editorial voice characteristics

        Args:
            titles: List of titles to analyze
            keyword: The keyword context for analysis

        Returns:
            Dictionary of editorial patterns for AI context
        """
        if not titles:
            return {
                'error': 'No titles to analyze',
                'patterns': {},
                'summary_for_ai': 'No editorial data available'
            }

        patterns = {
            # Length analysis
            'length_analysis': {
                'avg_length': sum(len(t) for t in titles) / len(titles),
                'min_length': min(len(t) for t in titles),
                'max_length': max(len(t) for t in titles),
                'preferred_range': self._get_preferred_length_range(titles)
            },

            # Keyword usage patterns
            'keyword_patterns': {
                'usage_rate': sum(1 for t in titles if keyword.lower() in t.lower()) / len(titles),
                'positions': self._analyze_keyword_positions(titles, keyword),
                'variations': self._find_keyword_variations(titles, keyword)
            },

            # Structural patterns
            'structure_patterns': {
                'questions': sum(1 for t in titles if t.strip().endswith('?')) / len(titles),
                'numbered_lists': sum(1 for t in titles if any(char.isdigit() for char in t[:15])) / len(titles),
                'how_to_format': sum(1 for t in titles if 'how to' in t.lower()) / len(titles),
                'colon_separated': sum(1 for t in titles if ':' in t) / len(titles),
                'parenthetical': sum(1 for t in titles if '(' in t and ')' in t) / len(titles)
            },

            # Editorial tone
            'tone_analysis': {
                'tone': self._detect_editorial_tone(titles),
                'formality_level': self._analyze_formality(titles),
                'emotional_indicators': self._find_emotional_words(titles)
            },

            # Content themes
            'content_themes': self._extract_content_themes(titles, keyword),

            # Common word patterns
            'vocabulary_patterns': {
                'common_words': self._get_most_common_content_words(titles),
                'power_words': self._find_power_words(titles),
                'domain_specific_terms': self._find_domain_terms(titles, keyword)
            }
        }

        # Create AI-friendly summary
        patterns['summary_for_ai'] = self._create_ai_summary(patterns, keyword)
        patterns['confidence_score'] = len(titles) / 10.0  # Confidence based on sample size

        return patterns

    def _get_preferred_length_range(self, titles: List[str]) -> Tuple[int, int]:
        """Get the preferred length range based on title distribution"""
        lengths = [len(t) for t in titles]
        lengths.sort()
        # Get range covering middle 80% of titles
        low_idx = int(len(lengths) * 0.1)
        high_idx = int(len(lengths) * 0.9)
        return (lengths[low_idx], lengths[high_idx])

    def _analyze_keyword_positions(self, titles: List[str], keyword: str) -> Dict:
        """Analyze where keywords typically appear in titles"""
        positions = {'beginning': 0, 'middle': 0, 'end': 0, 'not_found': 0}

        for title in titles:
            title_lower = title.lower()
            keyword_lower = keyword.lower()

            if keyword_lower in title_lower:
                pos = title_lower.find(keyword_lower)
                title_length = len(title)

                if pos < title_length * 0.3:
                    positions['beginning'] += 1
                elif pos > title_length * 0.7:
                    positions['end'] += 1
                else:
                    positions['middle'] += 1
            else:
                positions['not_found'] += 1

        return positions

    def _find_keyword_variations(self, titles: List[str], keyword: str) -> List[str]:
        """Find variations of the keyword used in titles"""
        variations = set()
        keyword_lower = keyword.lower()

        for title in titles:
            words = re.findall(r'\b\w+\b', title.lower())
            for word in words:
                if keyword_lower in word or word in keyword_lower:
                    if word != keyword_lower and len(word) > 2:
                        variations.add(word)

        return list(variations)[:10]  # Top 10 variations

    def _detect_editorial_tone(self, titles: List[str]) -> str:
        """Detect overall editorial tone"""
        formal_indicators = ['comprehensive', 'complete', 'guide', 'ultimate', 'definitive', 'professional']
        casual_indicators = ['awesome', 'cool', 'amazing', 'best', 'top', 'great', 'super']
        technical_indicators = ['review', 'analysis', 'comparison', 'vs', 'features', 'specs']
        urgent_indicators = ['now', 'today', 'urgent', 'breaking', 'latest', 'new']

        counts = {
            'formal': sum(1 for title in titles for ind in formal_indicators if ind in title.lower()),
            'casual': sum(1 for title in titles for ind in casual_indicators if ind in title.lower()),
            'technical': sum(1 for title in titles for ind in technical_indicators if ind in title.lower()),
            'urgent': sum(1 for title in titles for ind in urgent_indicators if ind in title.lower())
        }

        return max(counts.items(), key=lambda x: x[1])[0] if max(counts.values()) > 0 else 'neutral'

    def _analyze_formality(self, titles: List[str]) -> str:
        """Analyze formality level of titles"""
        formal_score = 0
        informal_score = 0

        for title in titles:
            # Formal indicators
            if any(word in title.lower() for word in ['complete', 'comprehensive', 'guide', 'analysis']):
                formal_score += 1
            if title[0].isupper() and not title.isupper():  # Proper capitalization
                formal_score += 0.5

            # Informal indicators
            if any(word in title.lower() for word in ['awesome', 'cool', 'wow', 'omg']):
                informal_score += 1
            if '!' in title:
                informal_score += 0.5

        if formal_score > informal_score * 1.5:
            return 'formal'
        elif informal_score > formal_score * 1.5:
            return 'informal'
        else:
            return 'balanced'

    def _find_emotional_words(self, titles: List[str]) -> List[str]:
        """Find emotional trigger words in titles"""
        emotional_words = [
            'amazing', 'incredible', 'stunning', 'shocking', 'unbelievable',
            'ultimate', 'best', 'worst', 'epic', 'fantastic', 'awesome',
            'terrible', 'brilliant', 'perfect', 'horrible', 'wonderful'
        ]

        found_emotions = []
        for title in titles:
            for word in emotional_words:
                if word in title.lower() and word not in found_emotions:
                    found_emotions.append(word)

        return found_emotions[:5]  # Top 5

    def _extract_content_themes(self, titles: List[str], keyword: str) -> List[str]:
        """Extract main content themes from titles"""
        themes = []

        theme_patterns = {
            'reviews': r'\breview\b|\breviews\b|\brating\b',
            'guides': r'\bguide\b|\bguides\b|\bhow to\b|\btips\b',
            'comparisons': r'\bvs\b|\bcompare\b|\bcomparison\b|\bbetter\b',
            'lists': r'\btop\b|\bbest\b|\d+\s+(things|ways|tips)',
            'news': r'\bnews\b|\blatest\b|\bnew\b|\bupdate\b',
            'tutorials': r'\btutorial\b|\blearn\b|\bstep\b|\bbeginners\b',
            'problems': r'\bfix\b|\bsolve\b|\bproblem\b|\bissue\b|\berror\b'
        }

        for theme, pattern in theme_patterns.items():
            if any(re.search(pattern, title, re.IGNORECASE) for title in titles):
                themes.append(theme)

        return themes

    def _get_most_common_content_words(self, titles: List[str]) -> List[str]:
        """Get most common content words (excluding stop words)"""
        stop_words = {
            'the', 'and', 'for', 'are', 'with', 'you', 'can', 'how', 'what',
            'this', 'that', 'will', 'your', 'from', 'they', 'know', 'want',
            'been', 'good', 'much', 'some', 'time', 'very', 'when', 'come',
            'here', 'just', 'like', 'long', 'make', 'many', 'over', 'such',
            'take', 'than', 'them', 'well', 'were'
        }

        all_words = []
        for title in titles:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', title.lower())
            all_words.extend([w for w in words if w not in stop_words])

        return [word for word, count in Counter(all_words).most_common(10)]

    def _find_power_words(self, titles: List[str]) -> List[str]:
        """Find power words used in titles"""
        power_words = [
            'ultimate', 'best', 'top', 'amazing', 'incredible', 'exclusive',
            'secret', 'proven', 'guaranteed', 'instant', 'complete', 'essential',
            'perfect', 'powerful', 'effective', 'professional', 'advanced'
        ]

        found_power_words = []
        for title in titles:
            for word in power_words:
                if word in title.lower() and word not in found_power_words:
                    found_power_words.append(word)

        return found_power_words

    def _find_domain_terms(self, titles: List[str], keyword: str) -> List[str]:
        """Find domain-specific terminology"""
        # Extract words that appear frequently and are related to the keyword domain
        all_words = []
        for title in titles:
            words = re.findall(r'\b[a-zA-Z]{4,}\b', title.lower())
            all_words.extend(words)

        # Filter to words that might be domain-specific (appear multiple times)
        word_counts = Counter(all_words)
        domain_terms = [word for word, count in word_counts.items()
                       if count >= 2 and len(word) > 3 and word != keyword.lower()]

        return domain_terms[:8]

    def _create_ai_summary(self, patterns: Dict, keyword: str) -> str:
        """Create an AI-friendly summary of editorial patterns"""
        length_info = patterns['length_analysis']
        tone = patterns['tone_analysis']['tone']
        themes = patterns['content_themes']
        structure = patterns['structure_patterns']

        summary_parts = [
            f"Editorial Style Analysis for '{keyword}':",
            f"- Average title length: {length_info['avg_length']:.0f} characters",
            f"- Editorial tone: {tone}",
            f"- Content themes: {', '.join(themes[:3]) if themes else 'general'}",
        ]

        # Add structural insights
        if structure['questions'] > 0.3:
            summary_parts.append("- Frequently uses question format")
        if structure['numbered_lists'] > 0.3:
            summary_parts.append("- Often includes numbered lists/counts")
        if structure['how_to_format'] > 0.2:
            summary_parts.append("- Commonly uses how-to format")

        # Add keyword usage insight
        keyword_usage = patterns['keyword_patterns']['usage_rate']
        if keyword_usage > 0.7:
            summary_parts.append(f"- High keyword usage rate ({keyword_usage:.1%})")
        elif keyword_usage < 0.3:
            summary_parts.append(f"- Low direct keyword usage, uses variations")

        return "\n".join(summary_parts)

    def analyze_titles(self, titles: List[str], keyword: str) -> dict:
        """Quick title analysis"""
        if not titles:
            return {"error": "No titles found"}

        analysis = {
            "total_titles": len(titles),
            "avg_length": sum(len(t) for t in titles) / len(titles),
            "keyword_usage": sum(1 for t in titles if keyword.lower() in t.lower()),
            "titles": titles
        }

        return analysis

    def bulk_research_domains(self, domain_keyword_pairs: List[Tuple[str, str]],
                             max_workers: int = 3) -> Dict[str, Dict]:
        """
        Efficiently research multiple domains in bulk with intelligent caching and cost optimization.

        Args:
            domain_keyword_pairs: List of (domain, keyword) tuples to research
            max_workers: Number of concurrent workers for API requests

        Returns:
            Dictionary mapping domain_keyword -> research results
        """
        self.current_operation = "Bulk domain research"
        results = {}
        total_pairs = len(domain_keyword_pairs)

        if total_pairs == 0:
            return results

        print(f"Starting bulk research for {total_pairs} domain-keyword pairs...")
        self._update_progress(f"Initializing bulk research for {total_pairs} pairs", 0.05)

        # Check cache first to minimize API calls
        cached_count = 0
        uncached_pairs = []

        for domain, keyword in domain_keyword_pairs:
            site_query = f"site:{domain} {keyword}"
            cached_result = self._get_cached_research(site_query)

            if cached_result:
                results[f"{domain}_{keyword}"] = {
                    'domain': domain,
                    'keyword': keyword,
                    'titles': cached_result['titles'],
                    'editorial_patterns': cached_result['editorial_patterns'],
                    'from_cache': True,
                    'timestamp': cached_result['timestamp']
                }
                cached_count += 1
            else:
                uncached_pairs.append((domain, keyword))

        if cached_count > 0:
            print(f"Found {cached_count} cached results, {len(uncached_pairs)} need fresh data")
            self._update_progress(f"Found {cached_count} cached results", 0.2)

        if not uncached_pairs:
            print("All results found in cache!")
            self._update_progress("All results from cache", 1.0)
            return results

        # Process uncached pairs with controlled concurrency
        estimated_cost = len(uncached_pairs) * self.cost_per_request
        print(f"Estimated API cost for {len(uncached_pairs)} requests: ${estimated_cost:.3f}")

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_pair = {
                    executor.submit(self._research_single_domain, domain, keyword): (domain, keyword)
                    for domain, keyword in uncached_pairs
                }

                completed = 0
                # Process completed tasks
                for future in as_completed(future_to_pair):
                    domain, keyword = future_to_pair[future]
                    completed += 1

                    progress = 0.2 + (0.7 * completed / len(uncached_pairs))
                    self._update_progress(f"Completed {completed}/{len(uncached_pairs)}: {domain}", progress)

                    try:
                        result = future.result()
                        results[f"{domain}_{keyword}"] = result
                        print(f"✅ Completed: {domain} + '{keyword}'")
                    except Exception as e:
                        print(f"❌ Failed: {domain} + '{keyword}': {str(e)}")
                        # Add error result so we don't lose track
                        results[f"{domain}_{keyword}"] = {
                            'domain': domain,
                            'keyword': keyword,
                            'titles': [],
                            'editorial_patterns': {'error': str(e)},
                            'from_cache': False,
                            'error': str(e),
                            'timestamp': datetime.now().isoformat()
                        }

        except Exception as e:
            print(f"Bulk research failed: {e}")
            self._update_progress(f"Bulk research failed: {e}", 0.0)
            return results

        # Final summary
        successful = sum(1 for r in results.values() if 'error' not in r)
        total_cost = self.total_cost

        summary = {
            'total_researched': total_pairs,
            'from_cache': cached_count,
            'new_requests': len(uncached_pairs),
            'successful': successful,
            'failed': total_pairs - successful,
            'total_cost': total_cost,
            'cost_per_pair': total_cost / total_pairs if total_pairs > 0 else 0
        }

        print(f"Bulk research complete: {successful}/{total_pairs} successful, ${total_cost:.3f} total cost")
        self._update_progress(f"Bulk research complete: {successful}/{total_pairs} successful", 1.0)

        results['_summary'] = summary
        return results

    def _research_single_domain(self, domain: str, keyword: str) -> Dict:
        """Research a single domain-keyword pair (for use in threading)"""
        try:
            site_query = f"site:{domain} {keyword}"

            # Get titles
            titles = self.get_serp_titles(site_query)

            # Analyze editorial patterns
            if titles:
                editorial_patterns = self.analyze_editorial_patterns(titles, keyword)
            else:
                editorial_patterns = {
                    'error': 'No titles found',
                    'summary_for_ai': f'No content found for {keyword} on {domain}'
                }

            return {
                'domain': domain,
                'keyword': keyword,
                'titles': titles,
                'editorial_patterns': editorial_patterns,
                'from_cache': False,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            raise Exception(f"Research failed for {domain}: {str(e)}")

    def get_research_summary_for_ai(self, research_results: Dict[str, Dict]) -> str:
        """
        Create a comprehensive summary of research results optimized for AI context.

        Args:
            research_results: Results from bulk_research_domains()

        Returns:
            String summary formatted for AI consumption
        """
        if not research_results or '_summary' not in research_results:
            return "No research data available for AI context."

        summary_parts = []
        summary_parts.append("=== EDITORIAL RESEARCH SUMMARY ===\n")

        # Overall statistics
        summary_info = research_results['_summary']
        summary_parts.append(f"Research Coverage: {summary_info['successful']}/{summary_info['total_researched']} domains")
        summary_parts.append(f"Cache Efficiency: {summary_info['from_cache']}/{summary_info['total_researched']} from cache")
        summary_parts.append("")

        # Domain-specific insights
        successful_domains = 0
        for key, result in research_results.items():
            if key.startswith('_'):  # Skip summary
                continue

            if 'error' in result or not result.get('titles'):
                continue

            successful_domains += 1
            domain = result['domain']
            keyword = result['keyword']
            patterns = result.get('editorial_patterns', {})

            summary_parts.append(f"Domain: {domain} (keyword: '{keyword}')")

            if 'summary_for_ai' in patterns:
                summary_parts.append(f"  {patterns['summary_for_ai']}")

            # Add key insights
            if 'tone_analysis' in patterns:
                tone_info = patterns['tone_analysis']
                summary_parts.append(f"  - Editorial tone: {tone_info.get('tone', 'unknown')}")
                summary_parts.append(f"  - Formality: {tone_info.get('formality_level', 'unknown')}")

            if 'content_themes' in patterns:
                themes = patterns['content_themes'][:3]  # Top 3 themes
                if themes:
                    summary_parts.append(f"  - Content themes: {', '.join(themes)}")

            summary_parts.append("")

        if successful_domains == 0:
            summary_parts.append("No successful domain research available.")
        else:
            summary_parts.append(f"=== END RESEARCH SUMMARY ({successful_domains} domains analyzed) ===")

        return "\n".join(summary_parts)

# Convenience functions for app.py
def get_competitor_titles(keyword: str, domain: str = None) -> List[str]:
    """Get titles from DataForSEO - supports both general and site-specific searches"""
    login = os.getenv("DATAFORSEO_LOGIN")
    password = os.getenv("DATAFORSEO_PASSWORD")

    if not login or not password:
        print("DataForSEO credentials not found")
        return []

    try:
        analyzer = DataForSEOSimple(login, password)

        # If domain is provided, do site-specific search
        if domain:
            # Clean domain (remove http/https/www)
            clean_domain = domain.replace('https://', '').replace('http://', '').replace('www.', '').split('/')[0]
            site_query = f"site:{clean_domain} {keyword}"
            print(f"Site-specific search: {site_query}")
            return analyzer.get_serp_titles(site_query)
        else:
            # General SERP search
            return analyzer.get_serp_titles(keyword)
    except Exception as e:
        print(f"DataForSEO error: {e}")
        return []

def get_site_titles(domain: str, keyword: str) -> List[str]:
    """
    Get existing titles from a specific domain for a keyword.

    This is the primary function for content research workflow:

    Args:
        domain: Target domain (e.g., 'mopoga.net')
        keyword: Search keyword (e.g., 'casino')

    Returns:
        List of existing titles found on the domain for that keyword

    Example:
        titles = get_site_titles('mopoga.net', 'casino')
        # Returns titles from pages like site:mopoga.net casino

    Use case:
        Research existing content styles before generating new titles
        that match the site's editorial approach.
    """
    return get_competitor_titles(keyword, domain)

def bulk_research_for_dataframe(df, login: str, password: str, progress_callback=None) -> Dict[str, Dict]:
    """
    Convenience function to perform bulk research for a DataFrame with research_keyword column.

    Args:
        df: DataFrame with columns including 'target_website' and 'research_keyword'
        login: DataForSEO login
        password: DataForSEO password
        progress_callback: Optional callback function for progress updates

    Returns:
        Dictionary of research results suitable for AI context
    """
    if not login or not password:
        print("DataForSEO credentials not provided")
        return {}

    # Extract unique domain-keyword pairs
    domain_keyword_pairs = []
    for _, row in df.iterrows():
        if pd.notna(row.get('research_keyword', '')) and row.get('research_keyword', '').strip():
            from utils import clean_domain  # Import here to avoid circular imports
            domain = clean_domain(row['target_website'])
            keyword = row['research_keyword'].strip()
            if (domain, keyword) not in domain_keyword_pairs:
                domain_keyword_pairs.append((domain, keyword))

    if not domain_keyword_pairs:
        print("No research keywords found in DataFrame")
        return {}

    # Initialize analyzer with progress callback
    analyzer = DataForSEOSimple(login, password)
    if progress_callback:
        analyzer.set_progress_callback(progress_callback)

    # Perform bulk research
    results = analyzer.bulk_research_domains(domain_keyword_pairs, max_workers=3)

    # Create AI summary
    ai_summary = analyzer.get_research_summary_for_ai(results)
    results['_ai_summary'] = ai_summary

    return results