# Content Research Workflow Documentation

## Overview
The Content Research feature is designed to help content creators understand a target domain's editorial style and content strategy before generating new titles. This ensures generated titles match the site's existing approach and editorial voice.

## Core Use Case
**Problem**: You have a keyword (e.g., "casino") and want to create content for a specific domain (e.g., mopoga.net), but you don't know what type of content or title style would fit their existing approach.

**Solution**: Research existing content on that domain for your keyword to understand their editorial patterns.

## How It Works

### Step 1: Content Research
1. Go to **🔍 Content Research** tab
2. Enter target domain (e.g., `mopoga.net`)
3. Enter your keyword (e.g., `casino`)
4. Click **"🔍 Find Existing Titles"**

### Step 2: Analysis
The system performs a site-specific search using the format:
```
site:mopoga.net casino
```

This finds all pages on mopoga.net that contain "casino" in their titles.

### Step 3: Style Understanding
From the results, you can analyze:
- **Title formats** (How-to, Listicles, Reviews, etc.)
- **Editorial tone** (Professional, Casual, Technical)
- **Content types** (Guides, News, Analysis, etc.)
- **Keyword usage patterns**
- **Title length preferences**

### Step 4: Title Generation
Use insights from Step 3 to generate titles that:
- Match the site's existing style
- Follow their editorial patterns
- Fit their content strategy

## Example Workflow

### Input:
- **Domain**: mopoga.net
- **Keyword**: casino

### DataForSEO Query:
```
site:mopoga.net casino
```

### Example Results:
1. "Exploring Casino-Themed Games on Mobile - Mopoga"
2. "Rise of Social Casino Games in Ireland: From Free Play to Real ..."
3. "HOW IS MOBILE GAMING AFFECTING THE CASINO ..."
4. "Gaming - Mopoga"

### Style Analysis:
- Uses descriptive, explanatory titles
- Focuses on mobile/gaming angle
- Includes geographic specificity (Ireland)
- Mixes sentence case and ALL CAPS
- Often includes site branding (- Mopoga)

### Generated Title Ideas:
Based on the analysis, create titles like:
- "Mobile Casino Gaming Trends in [Location] - Mopoga"
- "THE FUTURE OF CASINO APPS IN 2024"
- "Exploring Real Money Casino Games on Mobile Devices"

## Technical Implementation

### DataForSEO Integration
```python
def get_site_titles(domain: str, keyword: str) -> List[str]:
    """Get existing titles from a specific domain for a keyword"""
    return get_competitor_titles(keyword, domain)
```

### Search Query Format
```python
site_query = f"site:{clean_domain} {keyword}"
```

### API Endpoint
```
POST https://api.dataforseo.com/v3/serp/google/organic/task_post
```

### Request Payload
```json
[{
    "keyword": "site:mopoga.net casino",
    "location_name": "United States",
    "language_code": "en",
    "depth": 10
}]
```

## Benefits

1. **Style Matching**: Generate titles that fit the site's editorial voice
2. **Content Strategy**: Understand what types of content work on the domain
3. **Gap Identification**: Find content opportunities (no results = opportunity)
4. **Editorial Consistency**: Maintain brand voice across content
5. **Competitive Intelligence**: Learn from successful content patterns

## Use Cases

### 1. Guest Posting
Research the target site's style before pitching article ideas.

### 2. Content Planning
Understand what content gaps exist on a domain.

### 3. Brand Consistency
Ensure new content matches existing editorial standards.

### 4. SEO Strategy
Identify successful title patterns for specific keywords.

### 5. Editorial Guidelines
Develop style guides based on existing successful content.

## API Costs
- **DataForSEO**: ~$0.003 per search query
- **Caching**: 24-hour cache reduces repeat costs
- **Efficient**: Only searches for titles, not full content

## Error Scenarios

### No Results Found
```
No existing titles found on mopoga.net for 'casino'
```
**Meaning**: Content opportunity - the keyword isn't well-covered on this domain.

### API Error
```
DataForSEO API Error: Authentication failed
```
**Solution**: Check API credentials at https://app.dataforseo.com/api-access

## Integration with Title Generation

After research, use the **🎯 Single Generation** tab with:
- **Target Website**: The researched domain
- **Target Anchor**: Your keyword
- **Style Context**: Insights from the content research

The AI will generate titles that match the discovered editorial patterns.

## File Locations

- **Main Implementation**: `app.py` (competitive_analysis_tab function)
- **DataForSEO Integration**: `dataforseo_simple.py`
- **API Functions**: `get_site_titles()`, `get_competitor_titles()`

## Configuration

Required environment variables:
```
DATAFORSEO_LOGIN=your-api-login
DATAFORSEO_PASSWORD=your-api-password
```

## Future Enhancements

1. **Content Type Classification**: Automatically categorize content types
2. **Sentiment Analysis**: Analyze editorial tone
3. **Competitive Comparison**: Compare multiple domains
4. **Trend Analysis**: Track content patterns over time
5. **Export Features**: Save research results for reference