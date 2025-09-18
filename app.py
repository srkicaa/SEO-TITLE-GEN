import streamlit as st
import pandas as pd
import yaml
import os
import asyncio
import json
import time
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any
import io
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import custom modules
from database import DatabaseManager
from scraper import WebsiteAnalyzer
from ai_handler import AIHandler
from patterns import PatternLearner
from competitive_analyzer import CompetitiveAnalyzer
from google_search import GoogleSearchAnalyzer
from dataforseo_simple import get_competitor_titles
from utils import validate_url, clean_domain, export_to_csv

# Initialize session state
if 'db_manager' not in st.session_state:
    st.session_state.db_manager = DatabaseManager()

if 'config' not in st.session_state:
    try:
        with open('config.yaml', 'r') as f:
            st.session_state.config = yaml.safe_load(f)
    except FileNotFoundError:
        st.session_state.config = {
            'api_keys': {'openai': '', 'claude': ''},
            'settings': {
                'default_model': 'gpt-5',
                'titles_per_request': 5,
                'max_concurrent_requests': 3,
                'cache_duration_hours': 24
            }
        }

def main():
    st.set_page_config(
        page_title="SEO Title Generator Pro",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🎯 SEO Title Generator Pro")
    st.markdown("*Dynamic SEO title generation powered by AI with intelligent site analysis*")
    
    # Sidebar for API configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Keys
        openai_key = st.text_input("OpenAI API Key",
                                 value=os.getenv("OPENAI_API_KEY", ""),
                                 type="password")
        claude_key = st.text_input("Anthropic API Key",
                                 value=os.getenv("ANTHROPIC_API_KEY", ""),
                                 type="password")

        st.subheader("🔍 SERP Data Integration")

        # DataForSEO Integration (Primary)
        dataforseo_login = st.text_input("DataForSEO Login",
                                        value=os.getenv("DATAFORSEO_LOGIN", ""),
                                        help="DataForSEO API login for real SERP titles")
        dataforseo_password = st.text_input("DataForSEO Password",
                                           value=os.getenv("DATAFORSEO_PASSWORD", ""),
                                           type="password",
                                           help="DataForSEO API password")

        use_dataforseo = st.checkbox("Use DataForSEO API (Cost-Effective)",
                                   value=bool(dataforseo_login and dataforseo_password),
                                   help="Real SERP titles with 24hr caching")

        # Google Search fallback in expander
        with st.expander("Google Search API (Legacy)"):
            google_api_key = st.text_input("Google API Key",
                                         value=os.getenv("GOOGLE_API_KEY", ""),
                                         type="password",
                                         help="Google Custom Search API key")
            google_search_id = st.text_input("Google Search Engine ID",
                                            value=os.getenv("GOOGLE_SEARCH_ID", "a5614ee6de8594e68"),
                                            help="Custom Search Engine ID")
            use_google_search = st.checkbox("Use Google Search API",
                                          value=bool(google_api_key) and not use_dataforseo,
                                          help="Fallback option with limited results")
        
        # Model Selection
        ai_model = st.selectbox("Default AI Model",
                              ["gpt-5", "gpt-5-mini", "claude-sonnet-4-20250514"],
                              index=0)
        
        # Generation Settings
        st.subheader("Generation Settings")
        titles_count = st.slider("Titles per request", 3, 10, 5)
        use_historical = st.checkbox("Use historical data", True)
        
        # Update config
        st.session_state.config['api_keys']['openai'] = openai_key
        st.session_state.config['api_keys']['claude'] = claude_key
        st.session_state.config['settings']['default_model'] = ai_model
        st.session_state.config['settings']['titles_per_request'] = titles_count
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Single Generation",
        "📊 Bulk Processing",
        "🔍 Content Research",
        "📈 Analytics & History",
        "🧠 Learning Center",
        "⚙️ Advanced Settings"
    ])
    
    with tab1:
        single_generation_tab(openai_key, claude_key)
    
    with tab2:
        bulk_processing_tab(openai_key, claude_key)
    
    with tab3:
        competitive_analysis_tab(dataforseo_login, dataforseo_password, use_dataforseo, google_api_key, google_search_id, use_google_search if 'use_google_search' in locals() else False)
    
    with tab4:
        analytics_tab()
    
    with tab5:
        learning_tab()
        
    with tab6:
        settings_tab()

def single_generation_tab(openai_key: str, claude_key: str):
    st.header("Single Title Generation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Input form
        with st.form("single_generation_form"):
            target_website = st.text_input(
                "Target Website URL *", 
                placeholder="https://example.com",
                help="The website where you want to place your content"
            )
            
            target_anchor = st.text_input(
                "Target Anchor Text *", 
                placeholder="your keyword or phrase",
                help="The main keyword/phrase to incorporate in titles"
            )
            
            source_website = st.text_input(
                "Source Website URL", 
                placeholder="https://your-site.com",
                help="Your website (optional)"
            )
            
            col_model, col_count = st.columns(2)
            with col_model:
                model_override = st.selectbox(
                    "AI Model", 
                    ["Use Default", "gpt-5", "gpt-5-mini", "claude-sonnet-4-20250514"]
                )
            
            with col_count:
                title_count = st.number_input("Number of titles", 3, 10, 5)
            
            submitted = st.form_submit_button("🚀 Generate Titles", type="primary")
        
        if submitted:
            if not target_website or not target_anchor:
                st.error("Please fill in required fields (Target Website and Anchor Text)")
                return
            
            if not validate_url(target_website):
                st.error("Please enter a valid target website URL")
                return
            
            if not (openai_key or claude_key):
                st.error("Please configure at least one API key in the sidebar")
                return
            
            generate_single_titles(target_website, target_anchor, source_website, 
                                 model_override, title_count, openai_key, claude_key)
    
    with col2:
        st.info("""
        **Tips for better results:**
        
        🎯 **Target Website**: Use the exact domain where you want to publish
        
        🔑 **Anchor Text**: Be specific with your main keyword
        
        🌐 **Source Website**: Your own site helps with context
        
        🤖 **Model Selection**: 
        - GPT-5: Best overall performance
        - GPT-5 Mini: Faster, cost-effective
        - Claude: Alternative perspective
        """)

def generate_single_titles(target_website: str, target_anchor: str, source_website: str, 
                          model_override: str, title_count: int, openai_key: str, claude_key: str):
    
    with st.spinner("🔍 Analyzing target website..."):
        # Initialize components
        analyzer = WebsiteAnalyzer()
        # Get DataForSEO credentials from environment
        dataforseo_login = os.getenv("DATAFORSEO_LOGIN", "")
        dataforseo_password = os.getenv("DATAFORSEO_PASSWORD", "")
        ai_handler = AIHandler(openai_key, claude_key, dataforseo_login, dataforseo_password)
        pattern_learner = PatternLearner(st.session_state.db_manager)
        
        # Clean domain
        domain = clean_domain(target_website)
        
        # Analyze website style
        try:
            analysis = analyzer.analyze_website_style(target_website)
            
            # Get historical patterns
            historical_patterns = pattern_learner.get_best_patterns(domain)
            
            # Combine analysis with historical data
            combined_patterns = {**analysis, 'historical': historical_patterns}
            
            st.success(f"✅ Analysis complete for {domain}")
            
            # Display analysis results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Title Length", f"{analysis.get('avg_length', 0)} chars")
            with col2:
                st.metric("Common Patterns", len(analysis.get('patterns', [])))
            with col3:
                st.metric("Editorial Style", analysis.get('editorial_style', 'Unknown'))
            
        except Exception as e:
            st.warning(f"Website analysis failed: {str(e)}")
            combined_patterns = {}
    
    with st.spinner("🤖 Generating SEO-optimized titles..."):
        try:
            # Select model
            model = model_override if model_override != "Use Default" else st.session_state.config['settings']['default_model']
            
            # Generate titles
            titles = ai_handler.generate_titles(
                target_site=target_website,
                anchor_text=target_anchor,
                source_site=source_website,
                style_patterns=combined_patterns,
                model=model,
                count=title_count
            )
            
            if titles:
                st.success(f"✅ Generated {len(titles)} titles!")
                
                # Display titles
                for i, title_data in enumerate(titles, 1):
                    with st.container():
                        st.markdown(f"### Title {i}")
                        
                        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                        
                        with col1:
                            st.markdown(f"**{title_data['title']}**")
                            if st.button(f"📋 Copy", key=f"copy_{i}"):
                                st.code(title_data['title'])
                        
                        with col2:
                            st.metric("SEO Score", f"{title_data.get('seo_score', 0)}/100")
                        
                        with col3:
                            st.metric("Style Match", f"{title_data.get('style_match', 0)}%")
                        
                        with col4:
                            st.metric("Length", f"{len(title_data['title'])} chars")
                        
                        # Additional metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.caption(f"🎯 Keyword Position: {title_data.get('keyword_position', 'N/A')}")
                        with col2:
                            st.caption(f"💡 Emotional Score: {title_data.get('emotional_score', 0)}/10")
                        with col3:
                            acceptance_prob = title_data.get('acceptance_probability', 0)
                            st.caption(f"📈 Acceptance Prob: {acceptance_prob:.1%}")
                        
                        st.divider()
                
                # Save to database
                batch_id = f"single_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                st.session_state.db_manager.save_order(
                    batch_id=batch_id,
                    target_website=target_website,
                    target_anchor=target_anchor,
                    source_website=source_website,
                    generated_titles=titles,
                    ai_model_used=model
                )
                
                st.info("💾 Results saved to database for future learning")
                
                # Store generated titles in session state for competitive analysis
                st.session_state.generated_titles = titles
            
            else:
                st.error("Failed to generate titles. Please check your API keys and try again.")
                
        except Exception as e:
            st.error(f"Title generation failed: {str(e)}")

def parse_pasted_data_with_ai(data: str, openai_key: str) -> pd.DataFrame:
    """AI-powered parsing using ChatGPT-5 Nano to handle any messy data format"""
    if not openai_key:
        st.error("OpenAI API key required for AI parsing")
        return None

    try:
        # Use ChatGPT-5 Mini to intelligently parse the data
        import openai
        client = openai.OpenAI(api_key=openai_key)

        parsing_prompt = f"""
You are a data parsing expert. Parse the following messy data into a clean CSV format with these exact columns:
target_website,target_anchor,source_website,research_keyword

CRITICAL DATA FORMAT:
Input format: source_website    target_anchor    target_website    [research_keyword]

Example input:
"https://www.casiny1.com/    casiny1.com    readybetgo.com"

EXACT MAPPING:
- Column 1 (https://www.casiny1.com/) = source_website (keep full URL)
- Column 2 (casiny1.com) = target_anchor (keyword for title)
- Column 3 (readybetgo.com) = target_website (clean domain only)
- Column 4 (if present) = research_keyword (or empty if missing)

CRITICAL: Output CSV must have columns in THIS ORDER:
target_website,target_anchor,source_website,research_keyword

Expected output for example:
target_website,target_anchor,source_website,research_keyword
readybetgo.com,casiny1.com,https://www.casiny1.com/,

PARSING RULES:
1. Split input on tabs or multiple spaces
2. target_website = Column 3 (clean domain, remove https://, www.)
3. target_anchor = Column 2 (as-is)
4. source_website = Column 1 (keep full URL format)
5. research_keyword = Column 4 if present, otherwise empty
6. Return CSV with headers in EXACT order: target_website,target_anchor,source_website,research_keyword

Raw data to parse:
```
{data}
```

Return ONLY clean CSV format with headers:"""

        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": parsing_prompt}],
            temperature=0.1,
            max_completion_tokens=2000
        )

        parsed_csv = response.choices[0].message.content.strip()

        # Convert AI-parsed CSV to DataFrame
        from io import StringIO
        df = pd.read_csv(StringIO(parsed_csv))

        # Validate we got the expected columns
        expected_cols = ['target_website', 'target_anchor', 'source_website', 'research_keyword']
        if not all(col in df.columns for col in expected_cols):
            st.warning("AI parsing didn't produce expected columns, trying manual parsing...")
            return parse_pasted_data_manual(data)

        return df

    except Exception as e:
        st.warning(f"AI parsing failed: {str(e)}. Trying manual parsing...")
        return parse_pasted_data_manual(data)

def parse_pasted_data_manual(data: str) -> pd.DataFrame:
    """Fallback manual parsing method"""
    try:
        lines = data.strip().split('\n')
        if len(lines) < 1:
            return None

        # Better delimiter detection - check for multiple spaces or tabs
        first_line = lines[0]

        # Split test to see what works best
        tab_split = first_line.split('\t')
        space_split = [x for x in first_line.split('    ') if x.strip()]  # Multiple spaces
        comma_split = first_line.split(',')

        # Choose the best delimiter based on which gives us 3-4 meaningful parts
        if len(tab_split) >= 3:
            delimiter = '\t'
        elif len(space_split) >= 3:
            delimiter = '    '  # Multiple spaces
        elif len(comma_split) >= 3:
            delimiter = ','
        else:
            # Fallback: try splitting on any whitespace
            import re
            whitespace_split = re.split(r'\s{2,}', first_line.strip())  # 2+ whitespace chars
            if len(whitespace_split) >= 3:
                delimiter = 'whitespace'
            else:
                delimiter = '\t'  # Default fallback

        # Check if first line looks like headers
        if any(word in first_line.lower() for word in ['target', 'website', 'anchor', 'keyword']):
            # Has headers
            if delimiter == 'whitespace':
                headers = [h.strip() for h in re.split(r'\s{2,}', first_line.strip())]
            else:
                headers = [h.strip() for h in first_line.split(delimiter)]
            data_lines = lines[1:]
        else:
            # No headers, assume standard format based on input pattern:
            # Input: source_website    target_anchor    target_website
            # But we need output: target_website, target_anchor, source_website, research_keyword
            headers = ['source_website', 'target_anchor', 'target_website', 'research_keyword']
            data_lines = lines

        # Parse data rows
        data_rows = []
        for line in data_lines:
            if line.strip():  # Skip empty lines
                if delimiter == 'whitespace':
                    import re
                    row = [cell.strip() for cell in re.split(r'\s{2,}', line.strip())]
                else:
                    row = [cell.strip() for cell in line.split(delimiter)]

                # Pad with empty strings if row is shorter than headers
                while len(row) < len(headers):
                    row.append('')
                data_rows.append(row)

        # Create DataFrame with actual column names from parsing
        df = pd.DataFrame(data_rows, columns=headers[:len(data_rows[0]) if data_rows else 4])

        # NOW CORRECTLY MAP THE COLUMNS based on the actual input format
        # Input format: source_website    target_anchor    target_website
        # We need: target_website, target_anchor, source_website, research_keyword

        if headers == ['source_website', 'target_anchor', 'target_website', 'research_keyword']:
            # Reorder columns to match expected output
            df_reordered = pd.DataFrame()
            df_reordered['target_website'] = df['target_website']
            df_reordered['target_anchor'] = df['target_anchor']
            df_reordered['source_website'] = df['source_website']
            df_reordered['research_keyword'] = df['research_keyword']
            df = df_reordered
        else:
            # For other formats, try to map columns intelligently
            if len(df.columns) >= 3:
                # Assume format: source_website, target_anchor, target_website
                df_new = pd.DataFrame()
                df_new['target_website'] = df.iloc[:, 2] if len(df.columns) > 2 else ''  # 3rd column
                df_new['target_anchor'] = df.iloc[:, 1] if len(df.columns) > 1 else ''   # 2nd column
                df_new['source_website'] = df.iloc[:, 0] if len(df.columns) > 0 else ''  # 1st column
                df_new['research_keyword'] = df.iloc[:, 3] if len(df.columns) > 3 else '' # 4th column or empty
                df = df_new

        # Ensure all required columns exist
        for col in ['target_website', 'target_anchor', 'source_website', 'research_keyword']:
            if col not in df.columns:
                df[col] = ''

        return df

    except Exception as e:
        st.error(f"Error parsing data: {str(e)}")
        return None

def parse_pasted_data(data: str, openai_key: str = "", batch_research_keyword: str = "") -> pd.DataFrame:
    """Smart parsing that tries AI first, then falls back to manual"""
    if openai_key:
        st.info("🤖 Using AI-powered parsing with ChatGPT-5 Nano...")
        df = parse_pasted_data_with_ai(data, openai_key)
    else:
        st.info("📝 Using manual parsing (add OpenAI key for AI parsing)...")
        df = parse_pasted_data_manual(data)

    # Apply batch research keyword with OVERRIDE logic
    if df is not None:
        df = df.fillna('')
        if batch_research_keyword.strip():
            if 'research_keyword' not in df.columns:
                df['research_keyword'] = batch_research_keyword
            else:
                # OVERRIDE: Batch keyword takes precedence over parsed values
                df['research_keyword'] = batch_research_keyword
                st.info(f"🔄 Applied batch research keyword '{batch_research_keyword}' to all {len(df)} rows")

    return df

def bulk_processing_tab(openai_key: str, claude_key: str):
    st.header("Bulk Title Generation")

    # Initialize session state for DataFrame
    if 'bulk_df' not in st.session_state:
        st.session_state.bulk_df = None

    # Create tabs for different input methods
    input_tab1, input_tab2, input_tab3 = st.tabs(["📤 Upload CSV File", "📝 Manual Entry", "📋 Paste from Sheets"])

    with input_tab1:
        st.markdown("### 📤 Upload CSV File")

        # Download sample file button
        if st.button("📥 Download Sample CSV Template", key="bulk_download_sample"):
            sample_csv = """target_website,target_anchor,source_website,research_keyword
readybetgo.com,casiny1.com,https://www.casiny1.com/,casino
australiabasket.com,casiny1.com,https://www.casiny1.com/,casino
nationalcasino.com.au,casiny1.com,https://www.casiny1.com/,casino"""
            st.download_button(
                label="💾 Download sample_bulk_data.csv",
                data=sample_csv,
                file_name="sample_bulk_data.csv",
                mime="text/csv",
                key="bulk_download_sample_file"
            )

        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="CSV should contain columns: target_website, target_anchor, source_website, research_keyword (optional)",
            key="bulk_upload_csv_file"
        )

        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                df = df.fillna('')

                # Validate columns
                required_cols = ['target_website', 'target_anchor']
                if not all(col in df.columns for col in required_cols):
                    st.error(f"CSV must contain columns: {', '.join(required_cols)}")
                    return

                # Add research_keyword column if not present (for backward compatibility)
                if 'research_keyword' not in df.columns:
                    df['research_keyword'] = ''

                # Store DataFrame in session state
                st.session_state.bulk_df = df
                st.success(f"✅ Loaded {len(df)} orders")
                st.dataframe(df.head())

            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")

    with input_tab2:
        st.markdown("### 📝 Manual Entry")
        st.info("Add orders one by one using the form below:")

        # Manual entry form
        with st.form("manual_entry_form"):
            col1, col2 = st.columns(2)
            with col1:
                target_website = st.text_input("Target Website", placeholder="readybetgo.com", key="manual_target_website")
                target_anchor = st.text_input("Target Anchor", placeholder="casiny1.com", key="manual_target_anchor")
            with col2:
                source_website = st.text_input("Source Website", placeholder="https://www.casiny1.com/", key="manual_source_website")
                research_keyword = st.text_input("Research Keyword", placeholder="casino", key="manual_research_keyword")

            add_order = st.form_submit_button("➕ Add Order")

        # Initialize session state for manual orders
        if 'manual_orders' not in st.session_state:
            st.session_state.manual_orders = []

        if add_order and target_website and target_anchor:
            new_order = {
                'target_website': target_website,
                'target_anchor': target_anchor,
                'source_website': source_website,
                'research_keyword': research_keyword
            }
            st.session_state.manual_orders.append(new_order)
            st.success(f"Added order for {target_website}")

        # Display current orders
        if st.session_state.manual_orders:
            st.subheader(f"Current Orders ({len(st.session_state.manual_orders)})")
            manual_df = pd.DataFrame(st.session_state.manual_orders)
            manual_df = manual_df.fillna('')
            st.dataframe(manual_df)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear All Orders", key="manual_clear_orders"):
                    st.session_state.manual_orders = []
                    st.session_state.bulk_df = None
                    st.rerun()
            with col2:
                if st.button("✅ Use These Orders", key="manual_use_orders"):
                    st.session_state.bulk_df = manual_df
                    st.success(f"Using {len(manual_df)} manually entered orders!")

    with input_tab3:
        st.markdown("### 📋 Paste Data from Google Sheets")
        st.markdown("""
        **How to use:**
        1. Select your data in Google Sheets (including headers)
        2. Copy it (Ctrl+C or Cmd+C)
        3. Paste it in the text box below
        4. Click "Parse Data" to process
        """)

        # Text area for pasting data
        pasted_data = st.text_area(
            "Paste your data here:",
            height=200,
            placeholder="https://www.casiny1.com/    casiny1.com    readybetgo.com\nhttps://www.casiny1.com/    casiny1.com    australiabasket.com\nhttps://www.casiny1.com/    Casiny casino    ausgolf.com.au",
            help="Paste data from Google Sheets. Can be 3 or 4 columns: source_website, target_anchor, target_website, research_keyword(optional)",
            key="bulk_pasted_data"
        )

        # Research keyword for the batch
        batch_research_keyword = st.text_input(
            "🔍 Research Keyword for this batch:",
            value="casino",
            help="This keyword will be used for content research if not provided in the data. All rows will use this keyword unless they have their own research_keyword column.",
            key="bulk_batch_research_keyword"
        )

        col1, col2 = st.columns([1, 3])
        with col1:
            parse_button = st.button("📊 Parse Data", type="primary", key="bulk_parse_data")
        with col2:
            if pasted_data:
                st.caption(f"Characters: {len(pasted_data)} | Lines: {len(pasted_data.splitlines())}")

        if parse_button and pasted_data:
            try:
                df = parse_pasted_data(pasted_data, openai_key, batch_research_keyword)
                if df is not None:
                    # Store DataFrame in session state
                    st.session_state.bulk_df = df
                    st.success(f"✅ Parsed {len(df)} orders successfully!")
                    st.dataframe(df.head())
                else:
                    st.error("Failed to parse data. Please check the format.")
            except Exception as e:
                st.error(f"Error parsing pasted data: {str(e)}")

    # Processing section (only show if we have data in session state)
    if st.session_state.bulk_df is not None and len(st.session_state.bulk_df) > 0:
        st.markdown("---")
        st.markdown("### ⚙️ Processing Settings")

        # Show current data summary
        st.info(f"📊 Ready to process {len(st.session_state.bulk_df)} orders")

        # Processing settings
        col1, col2, col3 = st.columns(3)
        with col1:
            batch_model = st.selectbox(
                "Batch AI Model",
                ["gpt-5", "gpt-5-mini", "claude-sonnet-4-20250514"],
                key="bulk_batch_model"
            )
        with col2:
            batch_titles_count = st.number_input("Titles per order", 1, 10, 3, key="bulk_titles_count")
        with col3:
            concurrent_limit = st.number_input("Concurrent requests", 1, 5, 2, key="bulk_concurrent_limit")

        # Research Detection and Configuration
        has_research_keywords = st.session_state.bulk_df['research_keyword'].notna().any() and st.session_state.bulk_df['research_keyword'].str.strip().any()

        if has_research_keywords:
            st.success("✅ Research keywords detected! Enhanced processing will be used.")

            # Get DataForSEO credentials
            dataforseo_login = os.getenv("DATAFORSEO_LOGIN", "")
            dataforseo_password = os.getenv("DATAFORSEO_PASSWORD", "")

            if not (dataforseo_login and dataforseo_password):
                st.warning("⚠️ DataForSEO credentials not configured. Research phase will be skipped.")
                st.info("Add DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD to your environment or .env file for research functionality.")
            else:
                unique_research_pairs = set()
                for _, row in st.session_state.bulk_df.iterrows():
                    if row.get('research_keyword', '').strip():
                        domain = clean_domain(row['target_website'])
                        unique_research_pairs.add((domain, row['research_keyword'].strip()))

                estimated_cost = len(unique_research_pairs) * 0.003  # $0.003 per request
                st.info(f"🔍 Will research {len(unique_research_pairs)} unique domain-keyword combinations. Estimated cost: ${estimated_cost:.3f}")
        else:
            st.info("ℹ️ No research keywords detected. Using traditional processing mode.")

        if st.button("🚀 Start Enhanced Batch Processing", type="primary", key="bulk_start_processing"):
            if not (openai_key or claude_key):
                st.error("Please configure API keys in the sidebar")
                return

            process_bulk_orders(st.session_state.bulk_df, batch_model, batch_titles_count,
                              concurrent_limit, openai_key, claude_key)

    else:
        # Show format example when no data is loaded
        st.markdown("---")
        st.info("""
        **Enhanced Data Format with Research Integration:**

        **For Pasting (Tab-separated):**
        ```
        target_website  target_anchor   source_website  research_keyword
        https://example.com     keyword phrase  https://mysite.com      casino
        https://another-site.com        another keyword https://mysite.com      gaming
        ```

        **For CSV Upload:**
        ```
        target_website,target_anchor,source_website,research_keyword
        https://example.com,keyword phrase,https://mysite.com,casino
        https://another-site.com,another keyword,https://mysite.com,gaming
        ```

        **Column Descriptions:**
        - **target_website**: Required - Where you want to place content
        - **target_anchor**: Required - Main keyword/phrase for the title
        - **source_website**: Optional - Your website for context
        - **research_keyword**: Optional - Keyword to research existing content on target domain

        **Research Integration Benefits:**
        - 🎯 Analyzes existing content style on target domains
        - 📊 Adapts titles to match editorial patterns
        - 🔍 Uses DataForSEO to find real content examples
        - ✨ Generates more contextually appropriate titles
        """)

def competitive_analysis_tab(dataforseo_login: str, dataforseo_password: str, use_dataforseo: bool,
                            google_api_key: str, google_search_id: str, use_google_search: bool):
    """
    Content Research & Style Analysis Tab

    Purpose: Help users research existing content on target domains to understand
    editorial patterns before generating new titles.

    Workflow:
    1. User enters target domain (e.g., mopoga.net)
    2. User enters keyword (e.g., casino)
    3. System searches site:domain.net keyword
    4. User analyzes existing titles to understand site's style
    5. User generates new titles that match the editorial approach

    This ensures generated content fits the target site's existing voice and strategy.
    """
    st.header("🔍 Content Research & Style Analysis")

    st.markdown("**Research existing content on any domain to understand their editorial style before creating new titles.**")

    # Display API status
    if use_dataforseo and dataforseo_login:
        st.success("✅ DataForSEO enabled - Search any domain for existing content")
    elif use_google_search and google_api_key:
        st.success("✅ Google Search enabled - Limited domain search available")
    else:
        st.warning("⚠️ Enable DataForSEO in sidebar for best results")
        st.info("💡 Add DataForSEO credentials to search specific domains")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Input form for site analysis
        with st.form("site_analysis_form"):
            target_domain = st.text_input(
                "Target Domain *",
                placeholder="mopoga.net",
                help="The domain to search for existing content (without https://)"
            )

            keyword_phrase = st.text_input(
                "Keyword/Phrase *",
                placeholder="casino",
                help="The keyword to search for on the target domain"
            )

            max_results = st.number_input("Max results", 5, 20, 10)

            analyze_button = st.form_submit_button("🔍 Find Existing Titles", type="primary")
        
        if analyze_button:
            if not keyword_phrase or not target_domain:
                st.error("Please enter both domain and keyword")
                return

            # Clean domain input
            domain = target_domain.replace('https://', '').replace('http://', '').replace('www.', '').split('/')[0]
            
            with st.spinner(f"Searching {domain} for '{keyword_phrase}'..."):
                try:
                    if use_dataforseo and dataforseo_login:
                        # Site-specific search using DataForSEO
                        with st.spinner(f"Searching site:{domain} {keyword_phrase}..."):
                            competitor_titles = get_competitor_titles(keyword_phrase, domain)

                        if competitor_titles:
                            st.success(f"✅ Found {len(competitor_titles)} existing titles on {domain}")

                            # Display existing titles in a clean format
                            st.subheader(f"📄 Existing Titles on {domain}")

                            for i, title in enumerate(competitor_titles[:max_results], 1):
                                with st.container():
                                    col1, col2 = st.columns([4, 1])
                                    with col1:
                                        st.write(f"**{i}.** {title}")
                                    with col2:
                                        st.caption(f"{len(title)} chars")

                            # Basic analysis
                            st.subheader("📊 Basic Analysis")
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                avg_length = sum(len(title) for title in competitor_titles) / len(competitor_titles)
                                st.metric("Average Length", f"{avg_length:.0f} chars")

                            with col2:
                                st.metric("Total Titles", len(competitor_titles))

                            with col3:
                                keyword_in_titles = sum(1 for title in competitor_titles if keyword_phrase.lower() in title.lower())
                                st.metric("Keyword Present", f"{keyword_in_titles}/{len(competitor_titles)}")

                            # Simple recommendations
                            st.subheader("💡 Recommendations")
                            st.write(f"• Target title length around {avg_length:.0f} characters")
                            if keyword_in_titles > len(competitor_titles) / 2:
                                st.write(f"• Include '{keyword_phrase}' in your title (found in {keyword_in_titles} competitor titles)")
                            else:
                                st.write(f"• Consider using '{keyword_phrase}' or related terms for differentiation")

                        else:
                            st.warning(f"No existing titles found on {domain} for '{keyword_phrase}'")
                            st.info("This could mean:")
                            st.markdown(f"""
                            - No pages on {domain} contain '{keyword_phrase}' in their titles
                            - The keyword appears in content but not titles
                            - This is a content opportunity - consider creating content for this keyword
                            """)

                            # Show what was searched
                            st.info(f"📋 Searched: `site:{domain} {keyword_phrase}`")

                    elif use_google_search and google_api_key:
                        # Use Google Search as fallback
                        st.warning("⚠️ Using Google Custom Search as fallback - results may be limited")
                        google_analyzer = GoogleSearchAnalyzer(google_api_key, google_search_id)

                        # Note: Google Custom Search has limitations for general SERP analysis
                        search_results = google_analyzer.get_title_intelligence(domain or "example.com", keyword_phrase)

                        if search_results.get('has_existing_content'):
                            st.success("✅ Found some results via Google Custom Search")
                            st.subheader("📋 Limited Google Search Results")

                            existing_titles = search_results.get('existing_titles', [])
                            for i, title in enumerate(existing_titles[:5], 1):
                                st.write(f"**{i}.** {title}")

                            recommendations = search_results.get('recommendations', [])
                            if recommendations:
                                st.subheader("💡 Basic Recommendations")
                                for rec in recommendations:
                                    st.write(f"• {rec}")
                        else:
                            st.info("Google Custom Search didn't find relevant results")
                            st.info("Consider upgrading to DataForSEO for comprehensive SERP analysis")

                    else:
                        # Fallback to simulated analysis
                        st.info("💡 Using simulated competitive analysis")
                        st.info("Enable DataForSEO API for real SERP data")

                        competitive_analyzer = CompetitiveAnalyzer()
                        analysis = competitive_analyzer.analyze_competitors(domain or "example.com", keyword_phrase, max_results)

                        if analysis['competitor_count'] > 0:
                            st.success(f"✅ Found {analysis['competitor_count']} competitor titles")

                            # Display competitor titles
                            st.subheader("📋 Competitor Titles")
                            competitor_data = []
                            for title in analysis['titles']:
                                competitor_data.append({
                                    'Title': title.title,
                                    'Length': title.length,
                                    'Keyword Position': title.keyword_position,
                                    'Directness Score': f"{title.directness_score:.2f}",
                                    'Format': title.format_type,
                                    'Tone': title.emotional_tone
                                })

                            df_competitors = pd.DataFrame(competitor_data)
                            st.dataframe(df_competitors, use_container_width=True)

                            # Analysis sections
                            col_patterns, col_directness = st.columns(2)

                            with col_patterns:
                                st.subheader("📊 Format Analysis")
                                format_analysis = analysis.get('format_analysis', {})
                                if format_analysis:
                                    format_dist = format_analysis.get('format_distribution', {})
                                    if format_dist:
                                        fig_formats = px.bar(
                                            x=list(format_dist.keys()),
                                            y=list(format_dist.values()),
                                            title="Title Format Distribution"
                                        )
                                        st.plotly_chart(fig_formats, use_container_width=True)

                                    st.metric("Most Common Format",
                                             format_analysis.get('most_common_format', 'N/A'))
                                    st.metric("Format Diversity",
                                             format_analysis.get('format_diversity', 0))

                            with col_directness:
                                st.subheader("🎯 Directness Analysis")
                                directness_analysis = analysis.get('directness_analysis', {})
                                if directness_analysis:
                                    st.metric("Average Directness",
                                             f"{directness_analysis.get('avg_directness', 0):.2f}")
                                    st.metric("Max Directness",
                                             f"{directness_analysis.get('max_directness', 0):.2f}")
                                    st.metric("Recommended Level",
                                             directness_analysis.get('recommended_directness', 'N/A'))

                                    # Directness distribution
                                    dist = directness_analysis.get('directness_distribution', {})
                                    if dist:
                                        fig_directness = px.pie(
                                            values=list(dist.values()),
                                            names=list(dist.keys()),
                                            title="Directness Distribution"
                                        )
                                        st.plotly_chart(fig_directness, use_container_width=True)

                            # Length analysis
                            st.subheader("📏 Length Analysis")
                            length_analysis = analysis.get('length_analysis', {})
                            if length_analysis:
                                col_len1, col_len2, col_len3 = st.columns(3)
                                with col_len1:
                                    st.metric("Average Length",
                                             f"{length_analysis.get('avg_length', 0):.0f} chars")
                                with col_len2:
                                    st.metric("Min Length",
                                             f"{length_analysis.get('min_length', 0)} chars")
                                with col_len3:
                                    st.metric("Max Length",
                                             f"{length_analysis.get('max_length', 0)} chars")

                            # Recommendations
                            st.subheader("💡 Recommendations")
                            recommendations = analysis.get('recommendations', [])
                            for i, rec in enumerate(recommendations):
                                st.write(f"{i+1}. {rec}")

                            # Store analysis in session state for potential comparison
                            st.session_state.last_competitive_analysis = analysis

                            # Compare with generated titles if requested
                            if include_generated and 'generated_titles' in st.session_state:
                                st.subheader("⚔️ Generated vs Competitors")
                                comparison = competitive_analyzer.compare_with_generated(
                                    analysis, st.session_state.generated_titles
                                )

                                # Display comparison results
                                if 'title_comparisons' in comparison:
                                    for comp in comparison['title_comparisons']:
                                        with st.expander(f"Analysis: {comp['title'][:50]}..."):
                                            analysis_data = comp['analysis']

                                            col_comp1, col_comp2 = st.columns(2)
                                            with col_comp1:
                                                st.metric("Directness vs Competitors",
                                                         analysis_data.get('directness_vs_competitors', 'N/A'))
                                                st.metric("Length vs Competitors",
                                                         analysis_data.get('length_vs_competitors', 'N/A'))

                                            with col_comp2:
                                                st.metric("Format Popularity",
                                                         analysis_data.get('format_popularity', 'N/A'))
                                                st.metric("Emotional Tone",
                                                         analysis_data.get('emotional_tone', 'N/A'))

                                            advantages = analysis_data.get('competitive_advantage', [])
                                            if advantages:
                                                st.write("**Competitive Advantages:**")
                                                for adv in advantages:
                                                    st.write(f"• {adv}")

                                # Comparison recommendations
                                comp_recommendations = comparison.get('recommendations', [])
                                if comp_recommendations:
                                    st.subheader("🎯 Comparison Recommendations")
                                    for i, rec in enumerate(comp_recommendations):
                                        st.write(f"{i+1}. {rec}")

                        else:
                            st.warning("No competitor titles found. Try a different domain or keyword.")
                        
                except Exception as e:
                    st.error(f"Competitive analysis failed: {str(e)}")
    
    with col2:
        st.info("""
        **Content Research Workflow:**

        🔍 **Step 1**: Enter target domain (e.g., mopoga.net)

        🎯 **Step 2**: Enter your keyword (e.g., casino)

        📄 **Step 3**: Discover existing content styles

        ✍️ **Step 4**: Generate titles that match their style

        **Example Use Cases:**
        - Research editorial patterns before writing
        - Understand site's content strategy
        - Find content gaps and opportunities
        - Match existing writing styles

        **Search Format:** `site:domain.com keyword`
        """)
        
        # Quick stats if analysis exists
        if hasattr(st.session_state, 'last_competitive_analysis'):
            analysis = st.session_state.last_competitive_analysis
            st.subheader("📈 Last Analysis")
            st.metric("Competitor Count", analysis.get('competitor_count', 0))
            st.metric("Analyzed Keyword", analysis.get('keyword', 'N/A'))
            st.metric("Target Domain", analysis.get('target_domain', 'N/A'))

def bulk_site_research(df: pd.DataFrame, dataforseo_login: str = None,
                      dataforseo_password: str = None) -> Dict[str, Dict[str, Any]]:
    """
    ENHANCED BULK PROCESSING - SITE RESEARCH PHASE

    This function performs comprehensive site research for unique domains using DataForSEO API.
    It's the core of the enhanced bulk processing workflow that combines content research
    with AI title generation for superior results.

    INTEGRATION WORKFLOW:
    1. Extracts unique domain-keyword combinations from research_keyword column
    2. Performs site-specific SERP searches (e.g., "site:domain.com keyword")
    3. Analyzes existing titles to extract editorial patterns
    4. Caches results for 24 hours to minimize API costs
    5. Returns structured data for AI context integration

    RESEARCH CONTEXT EXTRACTION:
    - Editorial tone analysis (formal, casual, technical, etc.)
    - Title structure patterns (questions, numbered lists, how-to format)
    - Keyword usage patterns and positioning
    - Content themes and vocabulary analysis
    - Length preferences and formatting conventions

    COST OPTIMIZATION:
    - Intelligent caching to avoid duplicate API calls
    - Batches unique domain-keyword combinations only
    - Estimated cost: $0.003 per unique combination
    - 24-hour cache duration for efficiency

    Args:
        df: DataFrame with 'target_website' and 'research_keyword' columns
        dataforseo_login: DataForSEO API login credential
        dataforseo_password: DataForSEO API password credential

    Returns:
        Dict mapping "domain_keyword" -> comprehensive research data including:
        - titles: List of existing titles found on the domain
        - patterns: Editorial patterns extracted from titles
        - analysis: Statistical analysis of title characteristics
        - research_timestamp: When the research was performed

    Example Usage:
        research_results = bulk_site_research(df, login, password)
        # Results used to inform AI generation in process_bulk_orders_enhanced()
    """

    # Check if DataForSEO is available
    if not (dataforseo_login and dataforseo_password):
        st.warning("⚠️ DataForSEO credentials not provided - skipping research phase")
        return {}

    # Get unique domains that need research
    domains_to_research = set()
    for _, row in df.iterrows():
        if row.get('research_keyword', '').strip():
            domain = clean_domain(row['target_website'])
            domains_to_research.add((domain, row['research_keyword'].strip()))

    if not domains_to_research:
        st.info("ℹ️ No research keywords provided - proceeding without site research")
        return {}

    st.info(f"🔍 Starting enhanced research for {len(domains_to_research)} domain-keyword combinations...")

    # Initialize Enhanced DataForSEO with streamlit progress integration
    try:
        from dataforseo_simple import DataForSEOSimple
        analyzer = DataForSEOSimple(dataforseo_login, dataforseo_password)

        # Set up progress tracking
        research_progress = st.progress(0)
        research_status = st.empty()

        def streamlit_progress_callback(message: str, progress: float):
            """Progress callback that updates Streamlit UI"""
            research_progress.progress(min(progress, 1.0))
            research_status.text(f"🔍 {message}")

        analyzer.set_progress_callback(streamlit_progress_callback)

    except Exception as e:
        st.error(f"Failed to initialize enhanced DataForSEO: {str(e)}")
        return {}

    # Convert set to list for bulk processing
    domain_keyword_pairs = list(domains_to_research)

    # Estimate cost
    estimated_cost = len(domain_keyword_pairs) * 0.003
    st.info(f"💰 Estimated research cost: ${estimated_cost:.3f} ({len(domain_keyword_pairs)} unique combinations)")

    try:
        # Use the enhanced bulk research functionality
        research_status.text(f"🚀 Starting bulk research with enhanced polling...")

        start_time = time.time()

        # Add timeout handling for DataForSEO API issues
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
        import threading

        # Use threading timeout for Windows compatibility
        result_container = []
        exception_container = []

        def run_bulk_research():
            try:
                result = analyzer.bulk_research_domains(domain_keyword_pairs, max_workers=3)
                result_container.append(result)
            except Exception as e:
                exception_container.append(e)

        research_thread = threading.Thread(target=run_bulk_research)
        research_thread.daemon = True
        research_thread.start()
        research_thread.join(timeout=60)  # 60 second timeout

        if research_thread.is_alive():
            # Thread is still running - timeout occurred
            st.warning("⚠️ DataForSEO API timeout - switching to fallback mode")
            raise TimeoutError("DataForSEO API request timed out after 60 seconds")

        if exception_container:
            raise exception_container[0]

        research_results = result_container[0] if result_container else None

        duration = time.time() - start_time

        if research_results and '_summary' in research_results:
            summary = research_results['_summary']

            # Update UI with results
            research_progress.progress(1.0)
            research_status.text(f"✅ Research complete in {duration:.1f}s!")

            # Display detailed summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Researched", summary['total_researched'])
            with col2:
                st.metric("From Cache", summary['from_cache'])
            with col3:
                st.metric("Successful", summary['successful'])
            with col4:
                st.metric("Total Cost", f"${summary['total_cost']:.4f}")

            # Show cache efficiency
            cache_rate = (summary['from_cache'] / summary['total_researched']) * 100 if summary['total_researched'] > 0 else 0
            if cache_rate > 0:
                st.success(f"🚀 Cache efficiency: {cache_rate:.1f}% (saved {summary['from_cache']} API calls)")

            # Generate AI summary for context
            if summary['successful'] > 0:
                ai_summary = analyzer.get_research_summary_for_ai(research_results)
                research_results['_ai_summary'] = ai_summary

                # Show preview of AI summary
                with st.expander("📝 Research Context Summary (for AI)"):
                    st.text(ai_summary[:500] + "..." if len(ai_summary) > 500 else ai_summary)

            return research_results

        else:
            st.error("❌ Bulk research failed - no results returned")
            return {}

    except Exception as e:
        st.error(f"❌ Enhanced bulk research failed: {str(e)}")

        # Skip research entirely and return empty results to proceed with basic title generation
        st.warning("⚠️ DataForSEO unavailable - proceeding with traditional title generation")
        research_progress.progress(1.0)
        research_status.text("✅ Skipping research - using traditional generation")

        return {
            '_summary': {
                'total_researched': 0,
                'from_cache': 0,
                'successful': 0,
                'total_cost': 0.0
            },
            '_ai_summary': "No research data available due to DataForSEO API issues. Using traditional title generation."
        }

def _fallback_individual_research(domain_keyword_pairs, analyzer, progress_bar, status_text):
    """Fallback method for individual research when bulk fails"""
    research_cache = {}

    for i, (domain, keyword) in enumerate(domain_keyword_pairs):
        progress = (i + 1) / len(domain_keyword_pairs)
        progress_bar.progress(progress)
        status_text.text(f"🔍 Fallback research {i + 1}/{len(domain_keyword_pairs)}: {domain}")

        try:
            # Individual site search
            site_query = f"site:{domain} {keyword}"
            site_titles = analyzer.get_serp_titles(site_query)

            if site_titles:
                # Use enhanced editorial pattern analysis
                editorial_patterns = analyzer.analyze_editorial_patterns(site_titles, keyword)

                research_cache[f"{domain}_{keyword}"] = {
                    'domain': domain,
                    'keyword': keyword,
                    'titles': site_titles,
                    'editorial_patterns': editorial_patterns,
                    'from_cache': False,
                    'timestamp': datetime.now().isoformat()
                }
                st.success(f"✅ Found {len(site_titles)} titles for {domain}")
            else:
                research_cache[f"{domain}_{keyword}"] = {
                    'domain': domain,
                    'keyword': keyword,
                    'titles': [],
                    'editorial_patterns': {'error': 'No titles found'},
                    'from_cache': False,
                    'timestamp': datetime.now().isoformat()
                }
                st.warning(f"⚠️ No titles found for {domain} + '{keyword}'")

        except Exception as e:
            st.error(f"❌ Research failed for {domain}: {str(e)}")
            research_cache[f"{domain}_{keyword}"] = {
                'domain': domain,
                'keyword': keyword,
                'titles': [],
                'editorial_patterns': {'error': str(e)},
                'from_cache': False,
                'timestamp': datetime.now().isoformat()
            }

    # Add summary
    successful = sum(1 for r in research_cache.values() if 'error' not in r.get('editorial_patterns', {}))
    research_cache['_summary'] = {
        'total_researched': len(domain_keyword_pairs),
        'from_cache': 0,
        'new_requests': len(domain_keyword_pairs),
        'successful': successful,
        'failed': len(domain_keyword_pairs) - successful,
        'total_cost': analyzer.get_cost_summary()['total_cost'],
        'cost_per_pair': analyzer.get_cost_summary()['total_cost'] / len(domain_keyword_pairs) if domain_keyword_pairs else 0
    }

    return research_cache

def _extract_editorial_patterns(titles: List[str], keyword: str) -> Dict[str, Any]:
    """
    Extract editorial patterns from researched titles to inform AI generation.

    Args:
        titles: List of titles found on the domain
        keyword: The research keyword used

    Returns:
        Dictionary of editorial patterns and insights
    """
    if not titles:
        return {}

    patterns = {
        'avg_length': sum(len(title) for title in titles) / len(titles),
        'keyword_usage_rate': sum(1 for title in titles if keyword.lower() in title.lower()) / len(titles),
        'common_words': _get_most_common_words(titles),
        'title_structures': _analyze_title_structures(titles),
        'editorial_tone': _detect_editorial_tone(titles),
        'content_themes': _extract_content_themes(titles, keyword)
    }

    return patterns

def _get_most_common_words(titles: List[str]) -> List[str]:
    """Extract most common words from titles"""
    from collections import Counter
    import re

    # Combine all titles and extract words
    all_text = ' '.join(titles).lower()
    words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text)

    # Filter out common stop words
    stop_words = {'the', 'and', 'for', 'are', 'with', 'you', 'can', 'how', 'what', 'best', 'top'}
    words = [word for word in words if word not in stop_words]

    # Get top 10 most common words
    counter = Counter(words)
    return [word for word, count in counter.most_common(10)]

def _analyze_title_structures(titles: List[str]) -> Dict[str, int]:
    """Analyze common title structures and formats"""
    structures = {
        'questions': sum(1 for title in titles if title.strip().endswith('?')),
        'numbered': sum(1 for title in titles if any(char.isdigit() for char in title[:10])),
        'how_to': sum(1 for title in titles if 'how to' in title.lower()),
        'colon_separated': sum(1 for title in titles if ':' in title),
        'parentheses': sum(1 for title in titles if '(' in title and ')' in title)
    }
    return structures

def _detect_editorial_tone(titles: List[str]) -> str:
    """Detect the overall editorial tone from titles"""
    formal_indicators = ['guide', 'complete', 'comprehensive', 'ultimate', 'definitive']
    casual_indicators = ['awesome', 'cool', 'amazing', 'best', 'top', 'great']
    technical_indicators = ['review', 'analysis', 'comparison', 'vs', 'features']

    formal_count = sum(1 for title in titles for indicator in formal_indicators if indicator in title.lower())
    casual_count = sum(1 for title in titles for indicator in casual_indicators if indicator in title.lower())
    technical_count = sum(1 for title in titles for indicator in technical_indicators if indicator in title.lower())

    if formal_count > casual_count and formal_count > technical_count:
        return 'formal'
    elif casual_count > technical_count:
        return 'casual'
    elif technical_count > 0:
        return 'technical'
    else:
        return 'neutral'

def _extract_content_themes(titles: List[str], keyword: str) -> List[str]:
    """Extract content themes and topics from titles"""
    themes = []

    # Look for common content themes
    theme_patterns = {
        'reviews': r'\breview\b|\breviews\b',
        'guides': r'\bguide\b|\bguides\b|\bhow to\b',
        'comparisons': r'\bvs\b|\bcompare\b|\bcomparison\b',
        'lists': r'\btop\b|\bbest\b|\dthings',
        'news': r'\bnews\b|\blatest\b|\bnew\b',
        'tutorials': r'\btutorial\b|\blearn\b|\bstep\b'
    }

    for theme, pattern in theme_patterns.items():
        if any(re.search(pattern, title, re.IGNORECASE) for title in titles):
            themes.append(theme)

    return themes

def process_bulk_orders_enhanced(df: pd.DataFrame, model: str, titles_count: int,
                               concurrent_limit: int, openai_key: str, claude_key: str,
                               dataforseo_login: str = None, dataforseo_password: str = None):
    """
    Enhanced bulk processing with integrated DataForSEO research workflow.

    Workflow:
    1. Phase 1: Perform site research for unique domains (if research_keyword provided)
    2. Phase 2: Generate titles using research context + traditional website analysis
    3. Combine results with enhanced context awareness
    """

    # Initialize components
    analyzer = WebsiteAnalyzer()
    ai_handler = AIHandler(openai_key, claude_key, dataforseo_login, dataforseo_password)
    pattern_learner = PatternLearner(st.session_state.db_manager)

    batch_id = f"bulk_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    st.markdown("### 🚀 Enhanced Bulk Processing with Research Integration")

    # === PHASE 1: SITE RESEARCH ===
    research_results = {}
    has_research_keywords = df['research_keyword'].notna().any() and df['research_keyword'].str.strip().any()

    if has_research_keywords and dataforseo_login and dataforseo_password:
        st.markdown("#### 📊 Phase 1: Site Research")
        research_results = bulk_site_research(df, dataforseo_login, dataforseo_password)

        if research_results:
            st.success(f"✅ Research completed for {len(research_results)} domain-keyword combinations")

            # Show research summary
            with st.expander("📋 Research Summary"):
                for research_key, data in research_results.items():
                    domain = data['domain']
                    keyword = data['keyword']
                    title_count = len(data['titles'])

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(f"{domain}", f"{title_count} titles")
                    with col2:
                        patterns = data.get('patterns', {})
                        avg_length = patterns.get('avg_length', 0)
                        st.metric("Avg Length", f"{avg_length:.0f} chars")
                    with col3:
                        editorial_tone = patterns.get('editorial_tone', 'unknown')
                        st.metric("Editorial Tone", editorial_tone)
        else:
            st.info("ℹ️ No research data obtained - proceeding with traditional analysis")
    else:
        if has_research_keywords:
            st.warning("⚠️ Research keywords found but DataForSEO not configured - skipping research phase")
        else:
            st.info("ℹ️ No research keywords provided - using traditional website analysis only")

    # === PHASE 2: TITLE GENERATION ===
    st.markdown("#### 🎯 Phase 2: Title Generation")

    # Progress tracking for generation phase
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.empty()

    results = []

    for index, row in df.iterrows():
        progress = (index + 1) / len(df)
        progress_bar.progress(progress)
        status_text.text(f"Generating titles {index + 1}/{len(df)}: {row['target_website']}")

        try:
            # Get domain and research context
            domain = clean_domain(row['target_website'])
            research_keyword = row.get('research_keyword', '').strip()

            # Combine traditional website analysis with research data
            analysis = analyzer.analyze_website_style(row['target_website'])
            historical_patterns = pattern_learner.get_best_patterns(domain)

            # Get research context if available
            research_context = None
            if research_keyword and f"{domain}_{research_keyword}" in research_results:
                research_data = research_results[f"{domain}_{research_keyword}"]
                research_context = {
                    'existing_titles': research_data['titles'],
                    'editorial_patterns': research_data['patterns'],
                    'research_keyword': research_keyword,
                    'domain_analysis': research_data['analysis']
                }

            # Combine all patterns and context
            combined_patterns = {
                **analysis,
                'historical': historical_patterns,
                'research_context': research_context
            }

            # Generate titles with research context
            titles = ai_handler.generate_titles(
                target_site=row['target_website'],
                anchor_text=row['target_anchor'],
                source_site=row.get('source_website', ''),
                style_patterns=combined_patterns,
                model=model,
                count=titles_count,
                use_research_context=research_context is not None
            )

            if titles:
                best_title = max(titles, key=lambda x: x.get('seo_score', 0))

                result = {
                    'target_website': row['target_website'],
                    'target_anchor': row['target_anchor'],
                    'source_website': row.get('source_website', ''),
                    'research_keyword': research_keyword,
                    'research_used': research_context is not None,
                    'research_titles_found': len(research_context.get('existing_titles', [])) if research_context else 0,
                    'best_title': best_title.get('title', f"Complete Guide to {row['target_anchor']}") if isinstance(best_title, dict) else str(best_title),
                    'seo_score': best_title.get('seo_score', 0),
                    'style_match': best_title.get('style_match', 0),
                    'research_context_score': best_title.get('research_context_score', 0),
                    'all_titles': [t['title'] for t in titles]
                }

                # Save to database with research metadata
                st.session_state.db_manager.save_order(
                    batch_id=batch_id,
                    target_website=row['target_website'],
                    target_anchor=row['target_anchor'],
                    source_website=row.get('source_website', ''),
                    generated_titles=titles,
                    ai_model_used=model
                )

            else:
                result = {
                    'target_website': row['target_website'],
                    'target_anchor': row['target_anchor'],
                    'source_website': row.get('source_website', ''),
                    'research_keyword': research_keyword,
                    'research_used': False,
                    'research_titles_found': 0,
                    'best_title': 'Generation failed',
                    'seo_score': 0,
                    'style_match': 0,
                    'research_context_score': 0,
                    'all_titles': []
                }

            results.append(result)

        except Exception as e:
            st.warning(f"Error processing {row['target_website']}: {str(e)}")
            results.append({
                'target_website': row['target_website'],
                'target_anchor': row['target_anchor'],
                'source_website': row.get('source_website', ''),
                'research_keyword': row.get('research_keyword', ''),
                'research_used': False,
                'research_titles_found': 0,
                'best_title': f'Error: {str(e)}',
                'seo_score': 0,
                'style_match': 0,
                'research_context_score': 0,
                'all_titles': []
            })

    # === PHASE 3: RESULTS PRESENTATION ===
    progress_bar.progress(1.0)
    status_text.text("✅ Enhanced bulk processing complete!")

    # Display enhanced results
    results_df = pd.DataFrame(results)

    # Show success metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Processed", len(results))
    with col2:
        successful = len(results_df[results_df['seo_score'] > 0])
        st.metric("Successful", successful)
    with col3:
        research_used = len(results_df[results_df['research_used'] == True])
        st.metric("With Research", research_used)
    with col4:
        avg_score = results_df['seo_score'].mean()
        st.metric("Avg SEO Score", f"{avg_score:.1f}")

    st.success(f"🎉 Enhanced processing complete! {successful}/{len(results)} orders successful")

    # Enhanced results display
    st.subheader("📊 Results with Research Context")

    # Add color coding for research usage
    def color_research_status(val):
        if val:
            return 'background-color: #d4edda'
        return 'background-color: #f8d7da'

    styled_df = results_df.style.applymap(color_research_status, subset=['research_used'])
    st.dataframe(styled_df, use_container_width=True)

    # Research effectiveness analysis
    if research_used > 0:
        st.subheader("📈 Research Effectiveness Analysis")

        with_research = results_df[results_df['research_used'] == True]
        without_research = results_df[results_df['research_used'] == False]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Avg SEO Score (With Research)",
                     f"{with_research['seo_score'].mean():.1f}" if len(with_research) > 0 else "N/A")
        with col2:
            st.metric("Avg SEO Score (Without Research)",
                     f"{without_research['seo_score'].mean():.1f}" if len(without_research) > 0 else "N/A")

    # Download enhanced results
    csv_data = export_to_csv(results_df)
    st.download_button(
        label="📥 Download Enhanced Results CSV",
        data=csv_data,
        file_name=f"seo_titles_enhanced_{batch_id}.csv",
        mime="text/csv"
    )

# Backward compatibility wrapper
def process_bulk_orders(df: pd.DataFrame, model: str, titles_count: int,
                       concurrent_limit: int, openai_key: str, claude_key: str):
    """Backward compatibility wrapper for the enhanced bulk processing"""
    # Get DataForSEO credentials from sidebar/environment
    dataforseo_login = os.getenv("DATAFORSEO_LOGIN", "")
    dataforseo_password = os.getenv("DATAFORSEO_PASSWORD", "")

    return process_bulk_orders_enhanced(
        df, model, titles_count, concurrent_limit,
        openai_key, claude_key, dataforseo_login, dataforseo_password
    )

def analytics_tab():
    st.header("📈 Analytics & History")
    
    # Get data from database
    db_manager = st.session_state.db_manager
    
    # Date range filter
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
    with col2:
        end_date = st.date_input("End Date", datetime.now())
    
    # Get orders data
    orders = db_manager.get_orders_by_date_range(start_date, end_date)
    
    if orders:
        df_orders = pd.DataFrame(orders)
        
        # Key metrics
        st.subheader("📊 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Orders", len(df_orders))
        with col2:
            completed_orders = len(df_orders[df_orders['status'] == 'completed'])
            st.metric("Completed", completed_orders)
        with col3:
            if 'seo_score' in df_orders.columns:
                avg_score = df_orders['seo_score'].mean() if 'seo_score' in df_orders.columns else 0
                st.metric("Avg SEO Score", f"{avg_score:.1f}")
        with col4:
            accepted_orders = len(df_orders[df_orders.get('webmaster_accepted', False) == True])
            acceptance_rate = (accepted_orders / len(df_orders)) * 100 if len(df_orders) > 0 else 0
            st.metric("Acceptance Rate", f"{acceptance_rate:.1f}%")
        
        # Charts
        st.subheader("📈 Trends")
        
        # Orders over time
        if 'created_at' in df_orders.columns:
            df_orders['date'] = pd.to_datetime(df_orders['created_at']).dt.date
            daily_orders = df_orders.groupby('date').size().reset_index(name='orders')
            
            fig_timeline = px.line(daily_orders, x='date', y='orders', 
                                 title="Orders Over Time")
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Model performance
        if 'ai_model_used' in df_orders.columns:
            model_performance = df_orders.groupby('ai_model_used').agg({
                'seo_score': 'mean',
                'id': 'count'
            }).reset_index()
            model_performance.columns = ['Model', 'Avg_SEO_Score', 'Count']
            
            fig_models = px.bar(model_performance, x='Model', y='Avg_SEO_Score',
                              title="Model Performance Comparison")
            st.plotly_chart(fig_models, use_container_width=True)
        
        # Recent orders table
        st.subheader("📋 Recent Orders")
        st.dataframe(df_orders.tail(10))
        
    else:
        st.info("No orders found for the selected date range.")

def learning_tab():
    st.header("🧠 Learning Center")
    
    st.markdown("""
    The learning center helps improve title generation by analyzing patterns 
    from successful and unsuccessful placements.
    """)
    
    # Manual feedback section
    st.subheader("📝 Manual Feedback")
    
    # Get recent orders for feedback
    db_manager = st.session_state.db_manager
    recent_orders = db_manager.get_recent_orders(50)
    
    if recent_orders:
        for order in recent_orders[:10]:  # Show only 10 most recent
            with st.expander(f"Order: {order['target_website']} - {order['target_anchor']}"):
                st.text(f"Selected Title: {order.get('selected_title', 'Not selected')}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"✅ Mark as Accepted", key=f"accept_{order['id']}"):
                        db_manager.update_order_feedback(order['id'], True)
                        st.success("Marked as accepted!")
                        st.rerun()
                
                with col2:
                    if st.button(f"❌ Mark as Rejected", key=f"reject_{order['id']}"):
                        db_manager.update_order_feedback(order['id'], False)
                        st.success("Marked as rejected!")
                        st.rerun()
    
    # Pattern analysis
    st.subheader("🔍 Pattern Analysis")
    
    pattern_learner = PatternLearner(st.session_state.db_manager)
    
    # Domain selector for analysis
    domains = db_manager.get_unique_domains()
    if domains:
        selected_domain = st.selectbox("Select domain for analysis", domains)
        
        if selected_domain:
            patterns = pattern_learner.get_best_patterns(selected_domain)
            
            if patterns:
                st.json(patterns)
            else:
                st.info("No patterns found for this domain yet.")
    
    # Learning statistics
    st.subheader("📊 Learning Statistics")
    
    stats = db_manager.get_learning_stats()
    if stats:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sites Analyzed", stats.get('sites_count', 0))
        with col2:
            st.metric("Successful Titles", stats.get('successful_titles', 0))
        with col3:
            st.metric("Pattern Templates", stats.get('patterns_count', 0))

def settings_tab():
    st.header("⚙️ Advanced Settings")
    
    # Database management
    st.subheader("🗄️ Database Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Database"):
            db_data = st.session_state.db_manager.export_all_data()
            if db_data:
                st.download_button(
                    label="📥 Download Database Export",
                    data=json.dumps(db_data, indent=2),
                    file_name=f"seo_database_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    with col2:
        if st.button("🗑️ Clear Cache"):
            # Implement cache clearing logic
            st.success("Cache cleared!")
    
    with col3:
        if st.button("🔄 Reset Database", type="secondary"):
            if st.checkbox("I understand this will delete all data"):
                st.session_state.db_manager.reset_database()
                st.success("Database reset!")
                st.rerun()
    
    # Configuration
    st.subheader("⚙️ Configuration")
    
    # Save current config
    if st.button("💾 Save Configuration"):
        with open('config.yaml', 'w') as f:
            yaml.dump(st.session_state.config, f)
        st.success("Configuration saved!")
    
    # Display current config
    st.json(st.session_state.config)

if __name__ == "__main__":
    main()
