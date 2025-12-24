#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WordPress Migration Script with AI Content Rewriting
Migrates articles from JSON to WordPress while preserving image structure. 
"""

import json
import os
import re
import time
import logging
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from urllib.parse import urlparse, unquote
from pathlib import Path

import requests
from bs4 import BeautifulSoup, Tag, NavigableString
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging. INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('migration.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def ensure_url_scheme(url: str) -> str:
    """Ensure URL has a scheme (https://)."""
    url = url.strip() if url else ''
    if url and not url.startswith(('http://', 'https://')):
        return f'https://{url}'
    return url


# Configuration
CONFIG = {
    'WORDPRESS_URL': ensure_url_scheme(os.getenv('WORDPRESS_URL', '')),
    'WORDPRESS_AUTH_TOKEN': os. getenv('WORDPRESS_AUTH_TOKEN', '').strip(),
    'SOURCE_CMS_URL': ensure_url_scheme(os.getenv('SOURCE_CMS_URL', '')),
    'EDITOR_API_KEY': os.getenv('EDITOR_API_KEY', '').strip(),
    'VOTER_API_KEY': os.getenv('VOTER_API_KEY', '').strip(),
    'INPUT_JSON_PATH': os.getenv('INPUT_JSON_PATH', 'articles.json').strip(),
    'DEBUG_DIR': os.getenv('DEBUG_DIR', 'debug').strip(),
    'MAX_RETRIES': 3,
    'RETRY_BACKOFF': 2,
    'PASS_THRESHOLD': 70,
    'MAX_ITERATIONS': 3,
}

# AI Models
EDITOR_MODEL = 'openai/gpt-4. 1-nano'
VOTER_MODELS = [
    'google/gemma-3-27b-it: free',
    'xiaomi/mimo-v2-flash:free',
    'nex-agi/deepseek-v3. 1-nex-n1:free',
]

OPENROUTER_API_URL = 'https://openrouter.ai/api/v1/chat/completions'


@dataclass
class StructureComponent:
    """Represents a component in the HTML structure."""
    type: str
    position: int
    original_html: str
    text_content: Optional[str] = None
    tag_name: Optional[str] = None
    attributes: Dict = field(default_factory=dict)


@dataclass
class StructureTemplate:
    """Template preserving the original HTML structure."""
    components: List[StructureComponent]
    has_table: bool = False
    image_positions: List[int] = field(default_factory=list)
    text_positions: List[int] = field(default_factory=list)


@dataclass
class AIOutput:
    """Output from the AI editor."""
    rewritten_content: Dict[int, str]
    faq_html: str
    table_html: Optional[str]
    meta_description: str
    raw_response: str


@dataclass
class MigrationResult:
    """Result of migrating a single article."""
    success: bool
    article_id: str
    post_id: Optional[int] = None
    error: Optional[str] = None


def create_debug_dir():
    """Create debug directory if it doesn't exist."""
    Path(CONFIG['DEBUG_DIR']).mkdir(parents=True, exist_ok=True)


def save_debug_file(filename: str, content: str):
    """Save content to debug file."""
    filepath = Path(CONFIG['DEBUG_DIR']) / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    logger.debug(f"Saved debug file: {filepath}")


def retry_request(func):
    """Decorator for retrying HTTP requests with exponential backoff."""
    def wrapper(*args, **kwargs):
        last_exception = None
        for attempt in range(CONFIG['MAX_RETRIES']):
            try:
                return func(*args, **kwargs)
            except requests.exceptions.RequestException as e:
                last_exception = e
                wait_time = CONFIG['RETRY_BACKOFF'] ** attempt
                logger.warning(f"Request failed (attempt {attempt + 1}/{CONFIG['MAX_RETRIES']}): {e}")
                if attempt < CONFIG['MAX_RETRIES'] - 1:
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
        raise last_exception
    return wrapper


class HTMLStructureParser:
    """Parses HTML and extracts structure template while preserving image positions."""
    
    TEXT_TAGS = {'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}
    LIST_TAGS = {'ul', 'ol'}
    
    def parse(self, html: str) -> StructureTemplate:
        """Parse HTML and extract structure template."""
        soup = BeautifulSoup(html, 'html.parser')
        components = []
        position = 0
        has_table = False
        image_positions = []
        text_positions = []
        
        for element in soup.children:
            if isinstance(element, NavigableString):
                text = str(element).strip()
                if text: 
                    components.append(StructureComponent(
                        type='text',
                        position=position,
                        original_html=str(element),
                        text_content=text
                    ))
                    text_positions.append(position)
                    position += 1
                continue
                
            if not isinstance(element, Tag):
                continue
            
            component = self._parse_element(element, position)
            components.append(component)
            
            if component.type == 'image':
                image_positions.append(position)
            elif component. type in ('heading', 'paragraph', 'list'):
                text_positions.append(position)
            elif component.type == 'table': 
                has_table = True
            
            position += 1
        
        for img in soup.find_all('img'):
            parent_pos = self._find_parent_position(img, components)
            if parent_pos is not None and parent_pos not in image_positions:
                image_positions.append(parent_pos)
        
        return StructureTemplate(
            components=components,
            has_table=has_table,
            image_positions=image_positions,
            text_positions=text_positions
        )
    
    def _parse_element(self, element: Tag, position: int) -> StructureComponent: 
        """Parse a single HTML element."""
        tag_name = element.name.lower()
        
        if tag_name == 'img' or element.find('img'):
            return StructureComponent(
                type='image',
                position=position,
                original_html=str(element),
                tag_name=tag_name,
                attributes=dict(element. attrs) if hasattr(element, 'attrs') else {}
            )
        
        if tag_name == 'table' or element.find('table'):
            return StructureComponent(
                type='table',
                position=position,
                original_html=str(element),
                tag_name=tag_name,
                text_content=element.get_text(strip=True)
            )
        
        if tag_name in {'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}:
            return StructureComponent(
                type='heading',
                position=position,
                original_html=str(element),
                tag_name=tag_name,
                text_content=element.get_text(strip=True),
                attributes=dict(element. attrs) if hasattr(element, 'attrs') else {}
            )
        
        if tag_name == 'p':
            return StructureComponent(
                type='paragraph',
                position=position,
                original_html=str(element),
                tag_name=tag_name,
                text_content=element.get_text(strip=True),
                attributes=dict(element. attrs) if hasattr(element, 'attrs') else {}
            )
        
        if tag_name in self.LIST_TAGS:
            return StructureComponent(
                type='list',
                position=position,
                original_html=str(element),
                tag_name=tag_name,
                text_content=element.get_text(strip=True),
                attributes=dict(element. attrs) if hasattr(element, 'attrs') else {}
            )
        
        return StructureComponent(
            type='other',
            position=position,
            original_html=str(element),
            tag_name=tag_name,
            text_content=element.get_text(strip=True) if element.get_text(strip=True) else None,
            attributes=dict(element.attrs) if hasattr(element, 'attrs') else {}
        )
    
    def _find_parent_position(self, element: Tag, components: List[StructureComponent]) -> Optional[int]:
        """Find the position of the parent component containing this element."""
        element_html = str(element)
        for comp in components:
            if element_html in comp.original_html:
                return comp.position
        return None
    
    def extract_text_for_ai(self, template: StructureTemplate) -> List[Dict]: 
        """Extract only text content for AI rewriting."""
        text_blocks = []
        for comp in template.components:
            if comp.type in ('heading', 'paragraph', 'list') and comp.text_content:
                text_blocks.append({
                    'position': comp.position,
                    'type': comp.type,
                    'tag':  comp.tag_name,
                    'content': comp.text_content
                })
        return text_blocks


class AIEditor:
    """Handles AI-based content rewriting."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def create_editor_prompt(self, title: str, text_blocks: List[Dict],
                            keyword:  str, needs_table: bool) -> str:
        """Create the prompt for the editor AI."""
        text_content = "\n\n".join([
            f"[{block['type']. upper()} - Position {block['position']}]\n{block['content']}"
            for block in text_blocks
        ])
        
        prompt = f"""شما یک ویرایشگر محتوای حرفه‌ای برای وب‌سایت یک کارگزاری بورس هستید. 

**عنوان مقاله:** {title}
**کلمه کلیدی اصلی:** {keyword}

**محتوای اصلی برای بازنویسی:**
{text_content}

**دستورالعمل‌ها:**
1. محتوا را به فارسی رسمی اما قابل فهم برای مخاطبان بورس بازنویسی کنید
2. عبارت "کارگزاری اقتصاد بیدار" را با "کارگزاری بیدار" جایگزین کنید
3. کلمه کلیدی اصلی باید در پاراگراف اول ظاهر شود
4. حداقل 1200 کلمه فارسی تولید کنید
5. از تگ‌های HTML مناسب استفاده کنید:  <p>, <h2>, <h3>, <ul>, <li>
6. 10 سوال متداول (FAQ) با پاسخ تولید کنید
{"7. یک جدول HTML مرتبط با موضوع تولید کنید" if needs_table else "7. جدول نیاز نیست"}
8. توضیحات متا (حداکثر 156 کاراکتر) تولید کنید
9. هیچ توصیه سرمایه‌گذاری یا وعده سود ندهید
10. فقط ادعاهای واقعی که در محتوای اصلی وجود دارد را بیان کنید

**فرمت خروجی (فقط JSON معتبر، بدون markdown یا توضیحات):**
{{
    "rewritten_blocks": [
        {{"position": 0, "type": "heading/paragraph/list", "html": "<h2>محتوای بازنویسی شده</h2>"}},
        ... 
    ],
    "faq_html":  "<div class='faq'>... </div>",
    "table_html": "<table>...</table>" یا null,
    "meta_description": "توضیحات متا..."
}}"""
        return prompt
    
    @retry_request
    def call_editor(self, prompt: str) -> str:
        """Call the editor AI model."""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        
        payload = {
            'model':  EDITOR_MODEL,
            'messages': [
                {'role': 'user', 'content': prompt}
            ],
            'temperature':  0.7,
            'max_tokens': 4000,
        }
        
        response = requests.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        
        result = response.json()
        return result['choices'][0]['message']['content']
    
    def parse_editor_response(self, response: str) -> Optional[AIOutput]:
        """Parse the JSON response from the editor."""
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)
            
            rewritten_content = {}
            for block in data. get('rewritten_blocks', []):
                rewritten_content[block['position']] = block['html']
            
            return AIOutput(
                rewritten_content=rewritten_content,
                faq_html=data. get('faq_html', ''),
                table_html=data.get('table_html'),
                meta_description=data. get('meta_description', ''),
                raw_response=response
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger. error(f"Failed to parse editor response: {e}")
            return None
    
    def rewrite_content(self, title: str, text_blocks: List[Dict],
                       keyword: str, needs_table: bool) -> Optional[AIOutput]:
        """Rewrite content using AI."""
        prompt = self.create_editor_prompt(title, text_blocks, keyword, needs_table)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_debug_file(f'editor_prompt_{timestamp}.txt', prompt)
        
        response = self.call_editor(prompt)
        
        save_debug_file(f'editor_response_{timestamp}.txt', response)
        
        return self.parse_editor_response(response)


class AIVoter:
    """Handles AI-based content grading."""
    
    def __init__(self, api_key:  str):
        self.api_key = api_key
    
    def create_voter_prompt(self, title: str, content: str, meta_description: str) -> str:
        """Create the prompt for voter AI."""
        prompt = f"""شما یک ارزیاب محتوای حرفه‌ای هستید.  محتوای زیر را بر اساس معیارهای مشخص شده ارزیابی کنید. 

**عنوان:** {title}
**توضیحات متا:** {meta_description}

**محتوا:**
{content[: 3000]}...

**معیارهای ارزیابی:**
1. کیفیت نگارش فارسی (0-20)
2. ساختار و سازماندهی (0-20)
3. رعایت اصول SEO (0-20)
4. عدم وجود توصیه سرمایه‌گذاری نادرست (0-20)
5. کامل بودن محتوا و FAQ (0-20)

**فرمت خروجی (فقط JSON معتبر):**
{{
    "scores": {{
        "writing_quality": 0-20,
        "structure": 0-20,
        "seo": 0-20,
        "compliance": 0-20,
        "completeness": 0-20
    }},
    "total_score": 0-100,
    "passed":  true/false,
    "suggestions": ["پیشنهاد 1", "پیشنهاد 2"]
}}"""
        return prompt
    
    @retry_request
    def call_voter(self, prompt: str, model: str) -> str:
        """Call a voter AI model."""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type':  'application/json',
        }
        
        payload = {
            'model': model,
            'messages': [
                {'role': 'user', 'content': prompt}
            ],
            'temperature': 0.3,
            'max_tokens':  1000,
        }
        
        response = requests.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json()
        return result['choices'][0]['message']['content']
    
    def parse_voter_response(self, response: str) -> Dict:
        """Parse voter response."""
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match. group())
            return json.loads(response)
        except (json.JSONDecodeError, TypeError):
            return {'total_score': 0, 'passed': False, 'suggestions':  ['Failed to parse response']}
    
    def grade_content(self, title: str, content: str, meta_description: str) -> Tuple[bool, List[str]]:
        """Grade content using multiple voter models."""
        prompt = self. create_voter_prompt(title, content, meta_description)
        
        votes = []
        all_suggestions = []
        
        for model in VOTER_MODELS:
            try:
                logger.info(f"Getting vote from {model}...")
                response = self.call_voter(prompt, model)
                result = self.parse_voter_response(response)
                
                score = result. get('total_score', 0)
                passed = score >= CONFIG['PASS_THRESHOLD']
                votes.append(passed)
                
                if result.get('suggestions'):
                    all_suggestions.extend(result['suggestions'])
                
                logger. info(f"  {model}:  Score={score}, Passed={passed}")
                
            except Exception as e:
                logger.warning(f"Voter {model} failed: {e}")
                votes.append(False)
        
        passed_count = sum(votes)
        majority_passed = passed_count > len(votes) / 2
        
        logger.info(f"Voting result: {passed_count}/{len(votes)} passed, Majority:  {majority_passed}")
        
        return majority_passed, all_suggestions


class HTMLReconstructor:
    """Reconstructs HTML from template and AI output."""
    
    def reconstruct(self, template: StructureTemplate, ai_output: AIOutput) -> str:
        """Reconstruct HTML preserving original structure."""
        html_parts = []
        
        for comp in template.components:
            if comp.position in ai_output.rewritten_content:
                html_parts. append(ai_output.rewritten_content[comp.position])
            else:
                html_parts.append(comp.original_html)
        
        if not template.has_table and ai_output.table_html:
            html_parts.append(ai_output.table_html)
        
        if ai_output.faq_html:
            html_parts.append(ai_output. faq_html)
        
        return '\n'.join(html_parts)


class ImageProcessor:
    """Handles image downloading and uploading."""
    
    def __init__(self, source_cms_url: str, wp_url: str, wp_auth_token: str):
        self.source_cms_url = source_cms_url. rstrip('/')
        self.wp_url = wp_url. rstrip('/')
        self.wp_auth_token = wp_auth_token
    
    def extract_image_urls(self, html: str, json_images: List[str] = None) -> List[str]:
        """Extract all image URLs from HTML and JSON metadata."""
        soup = BeautifulSoup(html, 'html.parser')
        urls = set()
        
        for img in soup.find_all('img'):
            src = img. get('src', '')
            if src:
                urls.add(src)
        
        if json_images: 
            urls.update(json_images)
        
        return list(urls)
    
    @retry_request
    def download_image(self, image_path: str) -> Optional[bytes]:
        """Download image from source CMS."""
        if image_path.startswith('/'):
            url = f"{self.source_cms_url}{image_path}"
        elif not image_path.startswith('http'):
            url = f"{self.source_cms_url}/{image_path}"
        else: 
            url = image_path
        
        url = unquote(url)
        
        logger.info(f"Downloading image:  {url}")
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        return response.content
    
    @retry_request
    def upload_to_wordpress(self, image_data: bytes, filename: str) -> Optional[Dict]:
        """Upload image to WordPress media library."""
        headers = {
            'Authorization': f'Basic {self.wp_auth_token}',
            'Content-Disposition': f'attachment; filename="{filename}"',
        }
        
        ext = filename.lower().split('.')[-1]
        content_types = {
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png',
            'gif': 'image/gif',
            'webp': 'image/webp',
        }
        content_type = content_types.get(ext, 'image/jpeg')
        headers['Content-Type'] = content_type
        
        response = requests.post(
            f"{self.wp_url}/wp-json/wp/v2/media",
            headers=headers,
            data=image_data,
            timeout=60
        )
        response.raise_for_status()
        
        return response.json()
    
    def process_images(self, html: str, json_images: List[str] = None) -> Tuple[str, Optional[int]]:
        """Process all images:  download, upload, and replace URLs."""
        image_urls = self.extract_image_urls(html, json_images)
        url_mapping = {}
        featured_image_id = None
        
        priority_url = json_images[0] if json_images else None
        
        for url in image_urls:
            try:
                parsed = urlparse(url)
                filename = unquote(parsed.path. split('/')[-1])
                if not filename:
                    filename = f"image_{hashlib.md5(url.encode()).hexdigest()[:8]}.jpg"
                
                image_data = self.download_image(url)
                if not image_data:
                    continue
                
                wp_media = self.upload_to_wordpress(image_data, filename)
                if wp_media:
                    new_url = wp_media.get('source_url', '')
                    url_mapping[url] = new_url
                    
                    if url == priority_url or (featured_image_id is None and new_url):
                        featured_image_id = wp_media.get('id')
                    
                    logger.info(f"Uploaded image: {filename} -> {new_url}")
                    
            except Exception as e: 
                logger.error(f"Failed to process image {url}: {e}")
        
        updated_html = html
        for old_url, new_url in url_mapping.items():
            updated_html = updated_html.replace(old_url, new_url)
        
        return updated_html, featured_image_id


class WordPressClient:
    """WordPress API client."""
    
    def __init__(self, url: str, auth_token: str):
        self.url = url. rstrip('/')
        self.auth_token = auth_token
    
    @retry_request
    def post_exists(self, title: str) -> bool:
        """Check if a post with the same title already exists."""
        if not title:
            logger.warning("Empty title provided for duplicate check")
            return False
        
        headers = {
            'Authorization': f'Basic {self.auth_token}',
        }
        
        response = requests.get(
            f"{self.url}/wp-json/wp/v2/posts",
            headers=headers,
            params={'search': title, 'per_page': 1},
            timeout=30
        )
        response.raise_for_status()
        
        posts = response.json()
        for post in posts:
            if post.get('title', {}).get('rendered', '').strip() == title.strip():
                return True
        return False
    
    @retry_request
    def create_post(self, title: str, content: str, meta_description: str,
                   featured_image_id: Optional[int] = None, status: str = 'draft') -> Dict:
        """Create a new WordPress post."""
        headers = {
            'Authorization': f'Basic {self.auth_token}',
            'Content-Type': 'application/json',
        }
        
        payload = {
            'title': title,
            'content': content,
            'status': status,
            'excerpt': meta_description,
        }
        
        if featured_image_id:
            payload['featured_media'] = featured_image_id
        
        response = requests.post(
            f"{self.url}/wp-json/wp/v2/posts",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        
        return response.json()


class MigrationEngine:
    """Main migration engine orchestrating the process."""
    
    def __init__(self):
        self.parser = HTMLStructureParser()
        self.editor = AIEditor(CONFIG['EDITOR_API_KEY'])
        self.voter = AIVoter(CONFIG['VOTER_API_KEY'])
        self.reconstructor = HTMLReconstructor()
        self.image_processor = ImageProcessor(
            CONFIG['SOURCE_CMS_URL'],
            CONFIG['WORDPRESS_URL'],
            CONFIG['WORDPRESS_AUTH_TOKEN']
        )
        self.wp_client = WordPressClient(
            CONFIG['WORDPRESS_URL'],
            CONFIG['WORDPRESS_AUTH_TOKEN']
        )
        
        create_debug_dir()
    
    def extract_keyword(self, title: str) -> str:
        """Extract primary keyword from title."""
        words = title.split()
        if len(words) >= 2:
            return ' '. join(words[:3])
        return title
    
    def migrate_article(self, article:  Dict) -> MigrationResult:
        """Migrate a single article."""
        article_id = article. get('contentItemId', 'unknown')
        title = article.get('displayText', '')
        
        if not title:
            logger.warning(f"Article {article_id} has no title (displayText field missing or empty)")
        
        logger.info(f"Processing article: {title} (ID: {article_id})")
        
        try: 
            if title and self.wp_client.post_exists(title):
                logger. info(f"Skipping duplicate:  {title}")
                return MigrationResult(
                    success=False,
                    article_id=article_id,
                    error="Duplicate post"
                )
            
            html_body = article.get('htmlBody', {})
            if isinstance(html_body, dict):
                html = html_body.get('html', '')
            else:
                html = str(html_body)
            
            if not html:
                logger.warning(f"Article {article_id} has no HTML content")
                return MigrationResult(
                    success=False,
                    article_id=article_id,
                    error="No HTML content"
                )
            
            image_data = article.get('image', {})
            json_images = image_data.get('urls', []) if isinstance(image_data, dict) else []
            
            logger.info("Parsing HTML structure...")
            template = self.parser.parse(html)
            text_blocks = self.parser.extract_text_for_ai(template)
            
            save_debug_file(
                f'structure_{article_id}.json',
                json.dumps({
                    'has_table': template.has_table,
                    'image_positions': template.image_positions,
                    'text_positions': template.text_positions,
                    'components': [
                        {'type': c.type, 'position': c.position, 'tag': c.tag_name}
                        for c in template.components
                    ]
                }, ensure_ascii=False, indent=2)
            )
            
            keyword = self.extract_keyword(title)
            needs_table = not template.has_table
            
            best_output = None
            
            for iteration in range(CONFIG['MAX_ITERATIONS']):
                logger.info(f"AI rewriting iteration {iteration + 1}/{CONFIG['MAX_ITERATIONS']}...")
                
                ai_output = self.editor.rewrite_content(title, text_blocks, keyword, needs_table)
                
                if not ai_output:
                    logger.error("Failed to get AI output")
                    continue
                
                reconstructed = self.reconstructor.reconstruct(template, ai_output)
                
                passed, suggestions = self.voter.grade_content(
                    title, reconstructed, ai_output.meta_description
                )
                
                if passed: 
                    logger.info("Content passed voting!")
                    best_output = ai_output
                    break
                else:
                    logger.info(f"Content failed voting.  Suggestions: {suggestions[: 3]}")
                    if best_output is None:
                        best_output = ai_output
                    if suggestions and iteration < CONFIG['MAX_ITERATIONS'] - 1:
                        text_blocks. append({
                            'position': -1,
                            'type': 'suggestions',
                            'tag':  'feedback',
                            'content': f"بازخورد ارزیابان:  {'; '.join(suggestions[: 5])}"
                        })
            
            if not best_output:
                return MigrationResult(
                    success=False,
                    article_id=article_id,
                    error="Failed to generate content"
                )
            
            logger.info("Reconstructing final HTML...")
            final_html = self.reconstructor.reconstruct(template, best_output)
            
            logger.info("Processing images...")
            final_html, featured_image_id = self.image_processor.process_images(
                final_html, json_images
            )
            
            logger.info("Creating WordPress post...")
            post = self.wp_client.create_post(
                title=title,
                content=final_html,
                meta_description=best_output.meta_description,
                featured_image_id=featured_image_id,
                status='draft'
            )
            
            post_id = post.get('id')
            logger.info(f"Created post ID: {post_id}")
            
            return MigrationResult(
                success=True,
                article_id=article_id,
                post_id=post_id
            )
            
        except Exception as e:
            logger.error(f"Failed to migrate article {article_id}:  {e}", exc_info=True)
            return MigrationResult(
                success=False,
                article_id=article_id,
                error=str(e)
            )
    
    def run(self, json_path: str):
        """Run the migration process."""
        logger.info(f"Starting migration from:  {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            articles = data
            logger.info("Detected JSON structure:  direct array of articles")
        elif isinstance(data, dict):
            if 'data' in data and 'article' in data['data']:
                articles = data['data']['article']
                logger.info("Detected JSON structure: data. article[]")
            elif 'data' in data and isinstance(data['data'], list):
                articles = data['data']
                logger.info("Detected JSON structure: data[]")
            elif 'articles' in data: 
                articles = data['articles']
                logger.info("Detected JSON structure: articles[]")
            elif 'article' in data:
                articles = data['article']
                logger.info("Detected JSON structure: article[]")
            else:
                articles = None
                for key, value in data.items():
                    if isinstance(value, list):
                        articles = value
                        logger.info(f"Detected JSON structure: {key}[]")
                        break
                    elif isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, list):
                                articles = sub_value
                                logger.info(f"Detected JSON structure: {key}.{sub_key}[]")
                                break
                        if articles:
                            break
                if articles is None:
                    logger.error("Could not find articles array in JSON structure")
                    logger.error(f"Top-level keys found: {list(data.keys())}")
                    return []
        else:
            logger.error(f"Unexpected JSON structure: {type(data)}")
            return []
        
        if not isinstance(articles, list):
            articles = [articles]
        
        logger.info(f"Found {len(articles)} articles to migrate")
        
        if articles:
            first_article = articles[0]
            logger.info(f"First article keys: {list(first_article. keys()) if isinstance(first_article, dict) else 'N/A'}")
            if isinstance(first_article, dict):
                logger.info(f"First article ID: {first_article.get('contentItemId', 'N/A')}")
                logger.info(f"First article title: {first_article.get('displayText', 'N/A')}")
        
        results = []
        success_count = 0
        failed_count = 0
        post_ids = []
        
        for i, article in enumerate(articles, 1):
            logger.info(f"\n{'='*50}")
            logger. info(f"Processing article {i}/{len(articles)}")
            logger.info('='*50)
            
            result = self.migrate_article(article)
            results.append(result)
            
            if result.success:
                success_count += 1
                if result.post_id:
                    post_ids.append(result.post_id)
            else:
                failed_count += 1
            
            time.sleep(1)
        
        logger.info(f"\n{'='*50}")
        logger.info("MIGRATION SUMMARY")
        logger.info('='*50)
        logger.info(f"Total articles: {len(articles)}")
        logger.info(f"Successful:  {success_count}")
        logger.info(f"Failed: {failed_count}")
        logger.info(f"Created post IDs: {post_ids}")
        
        return results


def main():
    """Main entry point."""
    required_vars = [
        'WORDPRESS_URL', 'WORDPRESS_AUTH_TOKEN',
        'EDITOR_API_KEY', 'VOTER_API_KEY'
    ]
    
    missing = [var for var in required_vars if not CONFIG.get(var)]
    if missing:
        logger.error(f"Missing required environment variables: {missing}")
        logger.error("Please check your . env file")
        return
    
    logger.info(f"WordPress URL: {CONFIG['WORDPRESS_URL']}")
    logger.info(f"Source CMS URL: {CONFIG['SOURCE_CMS_URL']}")
    logger.info(f"Input JSON Path: {CONFIG['INPUT_JSON_PATH']}")
    
    json_path = CONFIG['INPUT_JSON_PATH']
    if not os.path.exists(json_path):
        logger.error(f"Input JSON file not found: {json_path}")
        return
    
    engine = MigrationEngine()
    engine.run(json_path)


if __name__ == '__main__':
    main()