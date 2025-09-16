import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd

class DatabaseManager:
    def __init__(self, db_path: str = "database.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Sites table - learns patterns over time
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sites (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        domain TEXT UNIQUE NOT NULL,
                        style_patterns TEXT,
                        typical_title_length INTEGER,
                        common_formats TEXT,
                        acceptance_rate REAL DEFAULT 0.0,
                        last_analyzed DATETIME,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Orders table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS orders (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        batch_id TEXT NOT NULL,
                        target_website TEXT NOT NULL,
                        target_anchor TEXT NOT NULL,
                        source_website TEXT,
                        status TEXT DEFAULT 'completed',
                        generated_titles TEXT,
                        selected_title TEXT,
                        webmaster_accepted BOOLEAN,
                        ai_model_used TEXT,
                        seo_score REAL,
                        style_match REAL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        completed_at DATETIME
                    )
                """)
                
                # Successful titles table - for learning
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS successful_titles (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        site_id INTEGER,
                        title TEXT NOT NULL,
                        anchor_text TEXT,
                        character_count INTEGER,
                        format_type TEXT,
                        acceptance_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                        seo_score REAL,
                        FOREIGN KEY (site_id) REFERENCES sites(id)
                    )
                """)
                
                # Style patterns table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS style_patterns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        domain TEXT NOT NULL,
                        pattern_type TEXT NOT NULL,
                        pattern_example TEXT,
                        frequency INTEGER DEFAULT 1,
                        success_rate REAL DEFAULT 0.0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                logging.info("Database initialized successfully")
                
        except Exception as e:
            logging.error(f"Database initialization failed: {e}")
            raise
    
    def save_order(self, batch_id: str, target_website: str, target_anchor: str,
                   source_website: str, generated_titles: List[Dict], 
                   ai_model_used: str) -> int:
        """Save a new order to the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Calculate best scores
                best_title = max(generated_titles, key=lambda x: x.get('seo_score', 0)) if generated_titles else {}
                seo_score = best_title.get('seo_score', 0)
                style_match = best_title.get('style_match', 0)
                selected_title = best_title.get('title', '')
                
                cursor.execute("""
                    INSERT INTO orders (
                        batch_id, target_website, target_anchor, source_website,
                        generated_titles, selected_title, ai_model_used,
                        seo_score, style_match, completed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    batch_id, target_website, target_anchor, source_website,
                    json.dumps(generated_titles), selected_title, ai_model_used,
                    seo_score, style_match, datetime.now()
                ))
                
                order_id = cursor.lastrowid
                conn.commit()
                
                logging.info(f"Order saved with ID: {order_id}")
                return order_id
                
        except Exception as e:
            logging.error(f"Failed to save order: {e}")
            raise
    
    def update_site_patterns(self, domain: str, patterns: Dict[str, Any]) -> None:
        """Update or insert site patterns"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO sites (
                        domain, style_patterns, typical_title_length,
                        common_formats, last_analyzed
                    ) VALUES (?, ?, ?, ?, ?)
                """, (
                    domain,
                    json.dumps(patterns),
                    patterns.get('avg_length', 0),
                    json.dumps(patterns.get('common_formats', [])),
                    datetime.now()
                ))
                
                conn.commit()
                logging.info(f"Site patterns updated for {domain}")
                
        except Exception as e:
            logging.error(f"Failed to update site patterns: {e}")
            raise
    
    def get_site_patterns(self, domain: str) -> Optional[Dict[str, Any]]:
        """Get stored patterns for a domain"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT style_patterns, typical_title_length, common_formats, last_analyzed
                    FROM sites WHERE domain = ?
                """, (domain,))
                
                result = cursor.fetchone()
                if result:
                    patterns = json.loads(result[0]) if result[0] else {}
                    patterns.update({
                        'typical_title_length': result[1],
                        'common_formats': json.loads(result[2]) if result[2] else [],
                        'last_analyzed': result[3]
                    })
                    return patterns
                
                return None
                
        except Exception as e:
            logging.error(f"Failed to get site patterns: {e}")
            return None
    
    def save_style_pattern(self, domain: str, pattern_type: str, 
                          pattern_example: str, success_rate: float = 0.0) -> None:
        """Save a style pattern"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if pattern exists
                cursor.execute("""
                    SELECT id, frequency FROM style_patterns 
                    WHERE domain = ? AND pattern_type = ? AND pattern_example = ?
                """, (domain, pattern_type, pattern_example))
                
                result = cursor.fetchone()
                
                if result:
                    # Update frequency and success rate
                    cursor.execute("""
                        UPDATE style_patterns 
                        SET frequency = frequency + 1, success_rate = ?
                        WHERE id = ?
                    """, (success_rate, result[0]))
                else:
                    # Insert new pattern
                    cursor.execute("""
                        INSERT INTO style_patterns (
                            domain, pattern_type, pattern_example, success_rate
                        ) VALUES (?, ?, ?, ?)
                    """, (domain, pattern_type, pattern_example, success_rate))
                
                conn.commit()
                
        except Exception as e:
            logging.error(f"Failed to save style pattern: {e}")
            raise
    
    def get_orders_by_date_range(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Get orders within a date range"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM orders 
                    WHERE DATE(created_at) BETWEEN DATE(?) AND DATE(?)
                    ORDER BY created_at DESC
                """, (start_date, end_date))
                
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                
                return [dict(zip(columns, row)) for row in rows]
                
        except Exception as e:
            logging.error(f"Failed to get orders by date range: {e}")
            return []
    
    def get_recent_orders(self, limit: int = 50) -> List[Dict]:
        """Get recent orders"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM orders 
                    ORDER BY created_at DESC 
                    LIMIT ?
                """, (limit,))
                
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                
                return [dict(zip(columns, row)) for row in rows]
                
        except Exception as e:
            logging.error(f"Failed to get recent orders: {e}")
            return []
    
    def update_order_feedback(self, order_id: int, accepted: bool) -> None:
        """Update order feedback"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE orders 
                    SET webmaster_accepted = ?
                    WHERE id = ?
                """, (accepted, order_id))
                
                conn.commit()
                
                # If accepted, add to successful titles
                if accepted:
                    cursor.execute("""
                        SELECT target_website, selected_title, target_anchor, seo_score
                        FROM orders WHERE id = ?
                    """, (order_id,))
                    
                    result = cursor.fetchone()
                    if result:
                        self._add_successful_title(
                            result[0], result[1], result[2], result[3]
                        )
                
        except Exception as e:
            logging.error(f"Failed to update order feedback: {e}")
            raise
    
    def _add_successful_title(self, domain: str, title: str, 
                             anchor_text: str, seo_score: float) -> None:
        """Add a successful title for learning"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get or create site
                cursor.execute("SELECT id FROM sites WHERE domain = ?", (domain,))
                site_result = cursor.fetchone()
                
                if not site_result:
                    cursor.execute("""
                        INSERT INTO sites (domain) VALUES (?)
                    """, (domain,))
                    site_id = cursor.lastrowid
                else:
                    site_id = site_result[0]
                
                # Add successful title
                cursor.execute("""
                    INSERT INTO successful_titles (
                        site_id, title, anchor_text, character_count, seo_score
                    ) VALUES (?, ?, ?, ?, ?)
                """, (site_id, title, anchor_text, len(title), seo_score))
                
                conn.commit()
                
        except Exception as e:
            logging.error(f"Failed to add successful title: {e}")
            raise
    
    def get_unique_domains(self) -> List[str]:
        """Get list of unique domains"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT DISTINCT target_website FROM orders
                    WHERE target_website IS NOT NULL
                    ORDER BY target_website
                """)
                
                return [row[0] for row in cursor.fetchall()]
                
        except Exception as e:
            logging.error(f"Failed to get unique domains: {e}")
            return []
    
    def get_learning_stats(self) -> Dict[str, int]:
        """Get learning statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Sites count
                cursor.execute("SELECT COUNT(*) FROM sites")
                sites_count = cursor.fetchone()[0]
                
                # Successful titles count
                cursor.execute("SELECT COUNT(*) FROM successful_titles")
                successful_titles = cursor.fetchone()[0]
                
                # Patterns count
                cursor.execute("SELECT COUNT(*) FROM style_patterns")
                patterns_count = cursor.fetchone()[0]
                
                return {
                    'sites_count': sites_count,
                    'successful_titles': successful_titles,
                    'patterns_count': patterns_count
                }
                
        except Exception as e:
            logging.error(f"Failed to get learning stats: {e}")
            return {}
    
    def export_all_data(self) -> Dict[str, Any]:
        """Export all database data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                data = {}
                
                # Export each table
                tables = ['sites', 'orders', 'successful_titles', 'style_patterns']
                
                for table in tables:
                    cursor = conn.cursor()
                    cursor.execute(f"SELECT * FROM {table}")
                    columns = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()
                    data[table] = [dict(zip(columns, row)) for row in rows]
                
                return data
                
        except Exception as e:
            logging.error(f"Failed to export data: {e}")
            return {}
    
    def reset_database(self) -> None:
        """Reset database by dropping and recreating tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Drop all tables
                tables = ['sites', 'orders', 'successful_titles', 'style_patterns']
                for table in tables:
                    cursor.execute(f"DROP TABLE IF EXISTS {table}")
                
                conn.commit()
                
            # Reinitialize
            self.init_database()
            logging.info("Database reset successfully")
            
        except Exception as e:
            logging.error(f"Failed to reset database: {e}")
            raise
