# WordPress Migration Script

A Python script that migrates articles from a JSON file to WordPress with AI-powered content rewriting while preserving the exact structure and position of images.

## Features

- **Structure-Preserving Rewrite**: Images remain in their exact original positions
- **AI Content Rewriting**: Uses OpenRouter AI models to improve content
- **Multi-Model Voting System**: Multiple AI models grade content quality
- **Automatic Image Migration**: Downloads images from source CMS and uploads to WordPress
- **Duplicate Detection**: Skips articles that already exist in WordPress
- **Comprehensive Error Handling**: Retries with exponential backoff
- **Detailed Logging**: All operations logged with timestamps

## Architecture

The script follows a 3-step structure-preserving approach:

1. **PARSE**: Analyze HTML and extract structure template
   - Identify all components (images, tables, lists, headings, paragraphs)
   - Record exact position/order of each component
   - Extract only text content for rewriting

2. **REWRITE**: Send only text blocks to AI
   - Images are never sent to AI
   - AI rewrites/improves text content
   - AI generates FAQ (10 questions) and table if needed

3. **RECONSTRUCT**: Rebuild final HTML
   - Original structure template preserved
   - New rewritten text from AI inserted
   - FAQ section appended
   - Table added if original didn't have one

## Installation

```bash
# Clone the repository
git clone https://github.com/samansohani78/wordpress-migration. git
cd wordpress-migration

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment variables
cp .env.example . env
# Edit .env with your credentials
```

## Configuration

Edit the `.env` file with your credentials:

```bash
# WordPress Configuration
WORDPRESS_URL=https://your-wordpress-site.com
WORDPRESS_AUTH_TOKEN=your_base64_encoded_auth_token

# Source CMS URL
SOURCE_CMS_URL=https://your-source-cms-url.com

# OpenRouter API Keys
EDITOR_API_KEY=your_editor_api_key
VOTER_API_KEY=your_voter_api_key

# Input file path
INPUT_JSON_PATH=articles.json
```

### Generating WordPress Auth Token

```bash
echo -n "username:application_password" | base64
```

## Input JSON Format

The script expects a JSON file with articles in this format:

```json
[
  {
    "contentItemId": "unique-id-123",
    "displayText": "Article Title in Persian",
    "htmlBody": {
      "html": "<p>Article HTML content... </p>"
    },
    "image":  {
      "urls": ["/media/image-name.jpg"]
    }
  }
]
```

## Usage

```bash
# Run the migration
python migrate.py

# Check logs
tail -f migration.log
```

## AI Models Used

### Editor Model
- `openai/gpt-4.1-nano` - For content rewriting

### Voter Models (Free tier)
- `google/gemma-3-27b-it: free`
- `xiaomi/mimo-v2-flash:free`
- `nex-agi/deepseek-v3.1-nex-n1:free`

## Voting System

- Each voter grades content on a 0-100 scale
- Pass threshold: 70
- Majority vote required to accept
- Maximum 3 iterations if failed
- Best output used if max iterations reached

## Output

- Posts created as drafts in WordPress
- Debug files saved in `debug/` directory
- Summary displayed at end: 
  - Success count
  - Failed count
  - Created post IDs

## Debug Files

The script saves debug files for troubleshooting: 

- `editor_prompt_*. txt` - AI prompts sent to editor
- `editor_response_*.txt` - Raw AI responses
- `structure_*.json` - HTML structure analysis

## Error Handling

- HTTP requests retry 3 times with exponential backoff
- JSON parsing failures handled gracefully
- Migration continues to next article if one fails
- All errors logged with timestamps

## License

MIT License