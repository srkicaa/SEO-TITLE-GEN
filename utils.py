import re
import csv
import io
import logging
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime

def validate_url(url: str) -> bool:
    """Validate URL format"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

def clean_domain(url: str) -> str:
    """Extract clean domain from URL"""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path
        # Remove www. prefix
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain.lower()
    except Exception:
        return url.lower()

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file system usage"""
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove excessive whitespace
    filename = re.sub(r'\s+', '_', filename)
    # Limit length
    return filename[:100]

def export_to_csv(data: pd.DataFrame) -> str:
    """Export DataFrame to CSV string"""
    try:
        output = io.StringIO()
        data.to_csv(output, index=False)
        return output.getvalue()
    except Exception as e:
        logging.error(f"CSV export failed: {e}")
        return ""

def format_timestamp(timestamp: datetime = None) -> str:
    """Format timestamp for display"""
    if timestamp is None:
        timestamp = datetime.now()
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")

def truncate_text(text: str, max_length: int = 50) -> str:
    """Truncate text with ellipsis if too long"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

def extract_keywords_from_text(text: str, max_keywords: int = 10) -> List[str]:
    """Extract potential keywords from text"""
    # Simple keyword extraction
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Filter out common stop words
    stop_words = {
        'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 
        'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 
        'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 
        'boy', 'did', 'man', 'end', 'few', 'got', 'own', 'say', 'she', 'too', 
        'use', 'from', 'they', 'know', 'want', 'been', 'good', 'much', 'some', 
        'time', 'very', 'when', 'come', 'here', 'just', 'like', 'long', 'make', 
        'many', 'over', 'such', 'take', 'than', 'them', 'well', 'were'
    }
    
    keywords = [word for word in words if word not in stop_words and len(word) > 3]
    
    # Count frequency and return most common
    from collections import Counter
    keyword_counts = Counter(keywords)
    return [keyword for keyword, count in keyword_counts.most_common(max_keywords)]

def calculate_title_metrics(title: str) -> Dict[str, Any]:
    """Calculate various metrics for a title"""
    words = title.split()
    
    metrics = {
        'character_count': len(title),
        'word_count': len(words),
        'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
        'has_numbers': bool(re.search(r'\d+', title)),
        'has_question_mark': title.endswith('?'),
        'has_exclamation': title.endswith('!'),
        'has_colon': ':' in title,
        'starts_with_capital': title[0].isupper() if title else False,
        'punctuation_count': len(re.findall(r'[^\w\s]', title)),
        'readability_score': calculate_simple_readability(title)
    }
    
    return metrics

def calculate_simple_readability(text: str) -> float:
    """Calculate simple readability score (0-1, higher is more readable)"""
    words = text.split()
    if not words:
        return 0.0
    
    # Simple metrics
    avg_word_length = sum(len(word) for word in words) / len(words)
    sentence_count = len(re.split(r'[.!?]+', text))
    words_per_sentence = len(words) / max(sentence_count, 1)
    
    # Readability formula (simplified)
    # Shorter words and sentences = higher readability
    readability = max(0, min(1, 1 - (avg_word_length - 4) / 10 - (words_per_sentence - 15) / 20))
    
    return readability

def parse_csv_safely(file_content: str) -> Optional[pd.DataFrame]:
    """Safely parse CSV content with error handling"""
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                if isinstance(file_content, bytes):
                    content = file_content.decode(encoding)
                else:
                    content = file_content
                
                # Parse CSV
                df = pd.read_csv(io.StringIO(content))
                return df
                
            except UnicodeDecodeError:
                continue
        
        return None
        
    except Exception as e:
        logging.error(f"CSV parsing failed: {e}")
        return None

def validate_csv_structure(df: pd.DataFrame, required_columns: List[str]) -> Dict[str, Any]:
    """Validate CSV structure and return validation results"""
    validation_result = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'row_count': len(df),
        'column_count': len(df.columns)
    }
    
    # Check required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        validation_result['is_valid'] = False
        validation_result['errors'].append(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Check for empty rows
    empty_rows = df.isnull().all(axis=1).sum()
    if empty_rows > 0:
        validation_result['warnings'].append(f"Found {empty_rows} empty rows")
    
    # Check for duplicate URLs
    if 'target_website' in df.columns:
        duplicates = df['target_website'].duplicated().sum()
        if duplicates > 0:
            validation_result['warnings'].append(f"Found {duplicates} duplicate target websites")
    
    # Validate URLs
    if 'target_website' in df.columns:
        invalid_urls = 0
        for url in df['target_website'].dropna():
            if not validate_url(str(url)):
                invalid_urls += 1
        
        if invalid_urls > 0:
            validation_result['warnings'].append(f"Found {invalid_urls} invalid URLs")
    
    return validation_result

def generate_batch_id() -> str:
    """Generate unique batch ID"""
    return f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

def format_seo_score(score: float) -> str:
    """Format SEO score for display"""
    if score >= 80:
        return f"🟢 {score:.1f}"
    elif score >= 60:
        return f"🟡 {score:.1f}"
    else:
        return f"🔴 {score:.1f}"

def format_acceptance_probability(prob: float) -> str:
    """Format acceptance probability for display"""
    percentage = prob * 100
    if percentage >= 75:
        return f"🔥 {percentage:.1f}%"
    elif percentage >= 50:
        return f"✅ {percentage:.1f}%"
    elif percentage >= 25:
        return f"⚠️ {percentage:.1f}%"
    else:
        return f"❌ {percentage:.1f}%"

def clean_text_for_analysis(text: str) -> str:
    """Clean text for analysis by removing extra whitespace and special characters"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\!\?\:\;\,\-\(\)]', '', text)
    
    return text.strip()

def extract_domain_from_email(email: str) -> str:
    """Extract domain from email address"""
    try:
        return email.split('@')[1].lower()
    except (IndexError, AttributeError):
        return ""

def generate_title_variations(base_title: str, variations: List[str]) -> List[str]:
    """Generate title variations based on patterns"""
    generated = []
    
    for variation in variations:
        # Simple template replacement
        if '{title}' in variation:
            new_title = variation.replace('{title}', base_title)
            generated.append(new_title)
    
    return generated

def is_mobile_friendly_title(title: str) -> bool:
    """Check if title is mobile-friendly (shorter length)"""
    return len(title) <= 50

def calculate_keyword_density(text: str, keyword: str) -> float:
    """Calculate keyword density in text"""
    if not text or not keyword:
        return 0.0
    
    text_lower = text.lower()
    keyword_lower = keyword.lower()
    
    # Count keyword occurrences
    keyword_count = text_lower.count(keyword_lower)
    
    # Count total words
    word_count = len(text.split())
    
    if word_count == 0:
        return 0.0
    
    return (keyword_count / word_count) * 100

def suggest_title_improvements(title: str, target_length: int = 55) -> List[str]:
    """Suggest improvements for a title"""
    suggestions = []
    
    title_length = len(title)
    
    # Length suggestions
    if title_length > target_length + 10:
        suggestions.append(f"Consider shortening title (currently {title_length} chars, target ~{target_length})")
    elif title_length < target_length - 10:
        suggestions.append(f"Consider expanding title (currently {title_length} chars, target ~{target_length})")
    
    # Structure suggestions
    if not title[0].isupper():
        suggestions.append("Consider capitalizing the first letter")
    
    if title.endswith('.'):
        suggestions.append("Consider removing trailing period for better SEO")
    
    if not any(char.isdigit() for char in title):
        suggestions.append("Consider adding numbers for better engagement")
    
    # SEO suggestions
    power_words = ['best', 'top', 'ultimate', 'complete', 'essential', 'proven']
    if not any(word in title.lower() for word in power_words):
        suggestions.append("Consider adding power words (best, top, ultimate, etc.)")
    
    return suggestions

def log_user_action(action: str, details: Dict[str, Any] = None):
    """Log user actions for analytics"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        'timestamp': timestamp,
        'action': action,
        'details': details or {}
    }
    
    logging.info(f"User Action: {json.dumps(log_entry)}")

def create_download_filename(prefix: str, extension: str = 'csv') -> str:
    """Create standardized download filename"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"

def validate_anchor_text(anchor_text: str) -> Dict[str, Any]:
    """Validate anchor text quality"""
    validation = {
        'is_valid': True,
        'warnings': [],
        'length': len(anchor_text),
        'word_count': len(anchor_text.split())
    }
    
    # Length check
    if len(anchor_text) < 2:
        validation['is_valid'] = False
        validation['warnings'].append("Anchor text too short")
    elif len(anchor_text) > 100:
        validation['warnings'].append("Anchor text might be too long")
    
    # Word count check
    word_count = len(anchor_text.split())
    if word_count > 10:
        validation['warnings'].append("Consider using fewer words in anchor text")
    
    # Special character check
    if re.search(r'[^\w\s\-]', anchor_text):
        validation['warnings'].append("Anchor text contains special characters")
    
    return validation

def format_processing_time(start_time: datetime, end_time: datetime = None) -> str:
    """Format processing time for display"""
    if end_time is None:
        end_time = datetime.now()
    
    duration = end_time - start_time
    seconds = duration.total_seconds()
    
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"
