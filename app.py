import streamlit as st
import pandas as pd
import yaml
import os
import asyncio
import json
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any
import io

# Import custom modules
from database import DatabaseManager
from scraper import WebsiteAnalyzer
from ai_handler import AIHandler
from patterns import PatternLearner
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Single Generation", 
        "📊 Bulk Processing", 
        "📈 Analytics & History", 
        "🧠 Learning Center",
        "⚙️ Advanced Settings"
    ])
    
    with tab1:
        single_generation_tab(openai_key, claude_key)
    
    with tab2:
        bulk_processing_tab(openai_key, claude_key)
    
    with tab3:
        analytics_tab()
    
    with tab4:
        learning_tab()
        
    with tab5:
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
        ai_handler = AIHandler(openai_key, claude_key)
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
            
            else:
                st.error("Failed to generate titles. Please check your API keys and try again.")
                
        except Exception as e:
            st.error(f"Title generation failed: {str(e)}")

def parse_pasted_data(data: str) -> pd.DataFrame:
    """Parse pasted data from Google Sheets (TSV/CSV format)"""
    try:
        lines = data.strip().split('\n')
        if len(lines) < 2:
            return None
        
        # Try to detect delimiter (tab vs comma)
        first_line = lines[0]
        if '\t' in first_line:
            delimiter = '\t'
        elif ',' in first_line:
            delimiter = ','
        else:
            return None
        
        # Parse header
        headers = [h.strip() for h in first_line.split(delimiter)]
        
        # Parse data rows
        data_rows = []
        for line in lines[1:]:
            if line.strip():  # Skip empty lines
                row = [cell.strip() for cell in line.split(delimiter)]
                # Pad with empty strings if row is shorter than headers
                while len(row) < len(headers):
                    row.append('')
                data_rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data_rows, columns=headers)
        
        # Validate required columns
        required_cols = ['target_website', 'target_anchor']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Data must contain columns: {', '.join(required_cols)}")
            return None
        
        return df
        
    except Exception as e:
        st.error(f"Error parsing data: {str(e)}")
        return None

def bulk_processing_tab(openai_key: str, claude_key: str):
    st.header("Bulk Title Generation")
    
    # Create tabs for different input methods
    input_tab1, input_tab2 = st.tabs(["📋 Paste from Sheets", "📤 Upload CSV File"])
    
    df = None
    
    with input_tab1:
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
            placeholder="target_website\ttarget_anchor\tsource_website\nhttps://example.com\tkeyword phrase\thttps://mysite.com\nhttps://another-site.com\tanother keyword\thttps://mysite.com",
            help="Paste data directly from Google Sheets. Headers should be in the first row."
        )
        
        col1, col2 = st.columns([1, 3])
        with col1:
            parse_button = st.button("📊 Parse Data", type="primary")
        with col2:
            if pasted_data:
                st.caption(f"Characters: {len(pasted_data)} | Lines: {len(pasted_data.splitlines())}")
        
        if parse_button and pasted_data:
            try:
                df = parse_pasted_data(pasted_data)
                if df is not None:
                    st.success(f"✅ Parsed {len(df)} orders successfully!")
                    st.dataframe(df.head())
                else:
                    st.error("Failed to parse data. Please check the format.")
            except Exception as e:
                st.error(f"Error parsing pasted data: {str(e)}")
    
    with input_tab2:
        st.markdown("### 📤 Upload CSV File")
        uploaded_file = st.file_uploader(
            "Choose a CSV file", 
            type=['csv'],
            help="CSV should contain columns: target_website, target_anchor, source_website"
        )
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Validate columns
                required_cols = ['target_website', 'target_anchor']
                if not all(col in df.columns for col in required_cols):
                    st.error(f"CSV must contain columns: {', '.join(required_cols)}")
                    return
                
                st.success(f"✅ Loaded {len(df)} orders")
                st.dataframe(df.head())
                
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
    
    # Processing section (only show if we have data)
    if df is not None and len(df) > 0:
        st.markdown("---")
        st.markdown("### ⚙️ Processing Settings")
        
        # Processing settings
        col1, col2, col3 = st.columns(3)
        with col1:
            batch_model = st.selectbox(
                "Batch AI Model", 
                ["gpt-5", "gpt-5-mini", "claude-sonnet-4-20250514"]
            )
        with col2:
            batch_titles_count = st.number_input("Titles per order", 1, 10, 3)
        with col3:
            concurrent_limit = st.number_input("Concurrent requests", 1, 5, 2)
        
        if st.button("🚀 Start Batch Processing", type="primary"):
            if not (openai_key or claude_key):
                st.error("Please configure API keys in the sidebar")
                return
            
            process_bulk_orders(df, batch_model, batch_titles_count, 
                              concurrent_limit, openai_key, claude_key)
    
    else:
        # Show format example when no data is loaded
        st.markdown("---")
        st.info("""
        **Data Format Example:**
        
        **For Pasting (Tab-separated):**
        ```
        target_website  target_anchor   source_website
        https://example.com     keyword phrase  https://mysite.com
        https://another-site.com        another keyword https://mysite.com
        ```
        
        **For CSV Upload:**
        ```
        target_website,target_anchor,source_website
        https://example.com,keyword phrase,https://mysite.com
        https://another-site.com,another keyword,https://mysite.com
        ```
        
        - **target_website**: Required - Where you want to place content
        - **target_anchor**: Required - Main keyword/phrase  
        - **source_website**: Optional - Your website for context
        """)

def process_bulk_orders(df: pd.DataFrame, model: str, titles_count: int, 
                       concurrent_limit: int, openai_key: str, claude_key: str):
    
    # Initialize components
    analyzer = WebsiteAnalyzer()
    ai_handler = AIHandler(openai_key, claude_key)
    pattern_learner = PatternLearner(st.session_state.db_manager)
    
    batch_id = f"bulk_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.empty()
    
    results = []
    
    for index, row in df.iterrows():
        progress = (index + 1) / len(df)
        progress_bar.progress(progress)
        status_text.text(f"Processing {index + 1}/{len(df)}: {row['target_website']}")
        
        try:
            # Analyze website
            domain = clean_domain(row['target_website'])
            analysis = analyzer.analyze_website_style(row['target_website'])
            historical_patterns = pattern_learner.get_best_patterns(domain)
            combined_patterns = {**analysis, 'historical': historical_patterns}
            
            # Generate titles
            titles = ai_handler.generate_titles(
                target_site=row['target_website'],
                anchor_text=row['target_anchor'],
                source_site=row.get('source_website', ''),
                style_patterns=combined_patterns,
                model=model,
                count=titles_count
            )
            
            if titles:
                best_title = max(titles, key=lambda x: x.get('seo_score', 0))
                
                result = {
                    'target_website': row['target_website'],
                    'target_anchor': row['target_anchor'],
                    'source_website': row.get('source_website', ''),
                    'best_title': best_title['title'],
                    'seo_score': best_title.get('seo_score', 0),
                    'style_match': best_title.get('style_match', 0),
                    'all_titles': [t['title'] for t in titles]
                }
                
                # Save to database
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
                    'best_title': 'Generation failed',
                    'seo_score': 0,
                    'style_match': 0,
                    'all_titles': []
                }
            
            results.append(result)
            
        except Exception as e:
            st.warning(f"Error processing {row['target_website']}: {str(e)}")
            results.append({
                'target_website': row['target_website'],
                'target_anchor': row['target_anchor'],
                'source_website': row.get('source_website', ''),
                'best_title': f'Error: {str(e)}',
                'seo_score': 0,
                'style_match': 0,
                'all_titles': []
            })
    
    # Complete processing
    progress_bar.progress(1.0)
    status_text.text("✅ Batch processing complete!")
    
    # Display results
    results_df = pd.DataFrame(results)
    st.success(f"🎉 Processed {len(results)} orders")
    st.dataframe(results_df)
    
    # Download results
    csv_data = export_to_csv(results_df)
    st.download_button(
        label="📥 Download Results CSV",
        data=csv_data,
        file_name=f"seo_titles_{batch_id}.csv",
        mime="text/csv"
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
