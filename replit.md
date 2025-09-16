# SEO Title Generator Pro

## Overview

SEO Title Generator Pro is a Streamlit-based application that generates SEO-optimized titles for bulk orders by analyzing website styles and matching them with target sites. The application uses AI models (GPT-5 and Claude) to generate contextually appropriate titles while learning from successful placements to improve future recommendations.

The system analyzes target websites to understand their editorial style, title patterns, and content preferences, then generates titles that are more likely to be accepted by webmasters. It maintains a learning database that tracks successful title placements and uses this data to refine its pattern recognition and generation algorithms.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Streamlit Web Interface**: Single-page application with sidebar configuration and main dashboard
- **Real-time Analytics**: Interactive charts and metrics using Plotly for visualizing success rates and patterns
- **Batch Processing UI**: Interface for bulk title generation with progress tracking
- **Export Functionality**: CSV export capabilities for generated titles and analytics

### Backend Architecture
- **Modular Design**: Separated concerns across specialized modules (database, scraping, AI, patterns, utilities)
- **AI Model Abstraction**: Handler class supporting both OpenAI GPT-5 and Claude Sonnet models with unified interface
- **Pattern Learning System**: Machine learning component that analyzes successful title patterns and improves recommendations
- **Web Scraping Engine**: Content extraction and analysis using trafilatura and BeautifulSoup for website style detection

### Data Storage
- **SQLite Database**: Local database with two main tables:
  - Sites table: Stores learned patterns, acceptance rates, and style analysis for each domain
  - Orders table: Tracks individual title generation requests, selections, and webmaster acceptance feedback
- **JSON Configuration**: YAML-based configuration for API keys and application settings
- **File-based Exports**: CSV export functionality for data portability

### AI Integration
- **Dual AI Provider Support**: Integration with both OpenAI (GPT-5, GPT-5-mini) and Anthropic (Claude Sonnet) APIs
- **Contextual Prompting**: Dynamic prompt generation based on target site analysis and learned patterns
- **Model Selection**: User-configurable AI model selection with fallback options
- **Response Processing**: Structured parsing and scoring of AI-generated titles

### Learning and Analytics
- **Pattern Recognition**: Automated extraction of title patterns, formats, and style preferences from successful placements
- **Success Rate Tracking**: Statistical analysis of acceptance rates per domain and title characteristics
- **Predictive Scoring**: Algorithm to predict likelihood of title acceptance based on historical data
- **Continuous Learning**: Feedback loop that improves recommendations based on webmaster acceptance/rejection

## External Dependencies

### AI Services
- **OpenAI API**: Primary AI service for GPT-5 and GPT-5-mini model access
- **Anthropic API**: Secondary AI service for Claude Sonnet model access

### Python Libraries
- **Streamlit**: Web application framework for the user interface
- **Plotly**: Interactive charting and visualization library
- **Pandas**: Data manipulation and analysis
- **Requests**: HTTP client for web scraping and API calls
- **BeautifulSoup**: HTML parsing for website content analysis
- **Trafilatura**: Content extraction from web pages
- **NLTK**: Natural language processing for sentiment analysis
- **SQLite3**: Built-in Python database interface
- **PyYAML**: Configuration file parsing

### Web Services
- **Target Website Analysis**: Scrapes and analyzes target websites to understand their content style and preferences
- **Content Extraction Services**: Uses trafilatura for clean content extraction from various website formats

### File System Dependencies
- **Local Database**: SQLite database file for persistent storage
- **Configuration Files**: YAML files for API keys and settings
- **Export Directory**: Local folder structure for CSV file exports
- **Log Files**: Application logging for debugging and monitoring