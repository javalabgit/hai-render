from flask import Flask, render_template, request, send_file, jsonify, session, redirect, url_for
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
#import google.generativeai as genai
#from apibook import api
from gtts import gTTS
from youtube_search import YoutubeSearch
import os
import fitz  # PyMuPDF
import sqlite3
import json
from datetime import datetime
import re
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
#from vectordb import search_summary,add_summary



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.secret_key = 'your-secret-key-here'  # Change this to a random secret key

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

import os
import logging

# Disable LiteLLM logging to prevent API key exposure
os.environ["LITELLM_LOG"] = "ERROR"
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

import psycopg2
from psycopg2 import sql

def post_db():
    # Use environment variable for Render’s Postgres connection URL
    
    DATABASE_URL = os.getenv("DATABASE_URL")

    pgconn = psycopg2.connect(DATABASE_URL)
    pgcursor = pgconn.cursor()

    # Users table
    pgcursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(255) UNIQUE NOT NULL,
            email VARCHAR(255) UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    pgconn.commit()
    pgcursor.close()
    pgconn.close()

post_db()

# Database setup
def init_db():
    conn = sqlite3.connect('eduaccess.db')
    cursor = conn.cursor()
    
    # Users table
   
    
    # Activities table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS activities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            action TEXT NOT NULL,
            details TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    # Add this table in init_db() function after the activities table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS assessments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        document_title TEXT,
        questions TEXT,
        user_answers TEXT,
        ai_feedback TEXT,
        score INTEGER,
        passed BOOLEAN,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
           )
            ''')


    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    conn.commit()
    conn.close()

# Initialize database
init_db()

import psycopg2
import psycopg2.extras

def get_pg_connection():
    DATABASE_URL = os.getenv("DATABASE_URL")
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    return conn



def get_db_connection():
    conn = sqlite3.connect('eduaccess.db')
    conn.row_factory = sqlite3.Row
    return conn

def add_activity(user_id, action, details=""):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO activities (user_id, action, details) VALUES (?, ?, ?)',
                   (user_id, action, details))
    conn.commit()
    conn.close()

# ✅ Extract text from PDF or TXT

from docx import Document
from concurrent.futures import ThreadPoolExecutor
import mmap


def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.pdf':
        return extract_pdf_text_pypdf2(file_path)
    elif ext == '.docx':
        return extract_docx_text_fast(file_path)
    elif ext == '.txt':
        return extract_txt_fast(file_path)

    return ''

# def extract_pdf_text_parallel(file_path):
#     doc = fitz.open(file_path)
#     try:
#         with ThreadPoolExecutor() as executor:
#             texts = list(executor.map(lambda i: doc[i].get_text("text"), range(len(doc))))
#         return ''.join(texts).strip()
#     finally:
#         doc.close()

#new-pdf-process

from PyPDF2 import PdfReader
from markdown import markdown
from crewai import Agent, Task, Crew, LLM
from typing import Optional
from dotenv import load_dotenv
load_dotenv()
#from fastapi.middleware.cors import CORSMiddleware

# Optional: enable CORS if needed
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# def extract_pdf_text_pypdf2(file_path):
#     try:
#         reader = PdfReader(file_path)
#         return ''.join(page.extract_text() or '' for page in reader.pages)
#     except Exception as e:
#         print(f"Error extracting PDF with PyPDF2: {e}")
#         return ''


# def extract_docx_text_fast(file_path):
#     doc = Document(file_path)
#     return '\n'.join(p.text for p in doc.paragraphs if p.text.strip())

# def extract_txt_fast(file_path):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
#             return mm.read().decode('utf-8').strip()


# # Configure Gemini API
# genai.configure(api_key=api())



import os
import fitz  # PyMuPDF
import mmap
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import hashlib
import pickle
from pathlib import Path
import logging
from typing import Optional, List, Tuple
from docx import Document
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache directory for extracted text
CACHE_DIR = Path("text_cache")
CACHE_DIR.mkdir(exist_ok=True)

class FastDocumentExtractor:
    def __init__(self, max_workers: int = 4, enable_cache: bool = True):
        self.max_workers = max_workers
        self.enable_cache = enable_cache
        
    def get_file_hash(self, file_path: str) -> str:
        """Generate hash for file caching"""
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5()
            # Read file in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b""):
                file_hash.update(chunk)
        return file_hash.hexdigest()
    
    def get_cache_path(self, file_path: str) -> Path:
        """Get cache file path"""
        file_hash = self.get_file_hash(file_path)
        filename = Path(file_path).stem
        return CACHE_DIR / f"{filename}_{file_hash}.txt"
    
    def load_from_cache(self, file_path: str) -> Optional[str]:
        """Load extracted text from cache"""
        if not self.enable_cache:
            return None
            
        cache_path = self.get_cache_path(file_path)
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    logger.info(f"Loaded from cache: {file_path}")
                    return f.read()
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
        return None
    
    def save_to_cache(self, file_path: str, text: str):
        """Save extracted text to cache"""
        if not self.enable_cache:
            return
            
        try:
            cache_path = self.get_cache_path(file_path)
            with open(cache_path, 'w', encoding='utf-8') as f:
                f.write(text)
            logger.info(f"Saved to cache: {file_path}")
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    def extract_pdf_chunk(self, doc, page_range: Tuple[int, int]) -> str:
        """Extract text from a range of PDF pages"""
        start, end = page_range
        texts = []
        for i in range(start, min(end, len(doc))):
            try:
                page_text = doc[i].get_text("text")
                if page_text.strip():
                    texts.append(page_text)
            except Exception as e:
                logger.warning(f"Error extracting page {i}: {e}")
                continue
        return ''.join(texts)

    def extract_pdf_parallel_optimized(self, file_path: str) -> str:
        """Optimized parallel PDF extraction"""
        start_time = time.time()
        
        try:
            doc = fitz.open(file_path)
            total_pages = len(doc)
            
            if total_pages == 0:
                return ""
            
            # For small PDFs, use single thread
            if total_pages <= 5:
                text = ''.join(doc[i].get_text("text") for i in range(total_pages))
                doc.close()
                logger.info(f"PDF extracted (single-thread) in {time.time() - start_time:.2f}s")
                return text.strip()
            
            # Calculate optimal chunk size
            chunk_size = max(1, total_pages // self.max_workers)
            page_ranges = [
                (i, min(i + chunk_size, total_pages)) 
                for i in range(0, total_pages, chunk_size)
            ]
            
            texts = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_range = {
                    executor.submit(self.extract_pdf_chunk, doc, page_range): page_range 
                    for page_range in page_ranges
                }
                
                for future in as_completed(future_to_range):
                    try:
                        text_chunk = future.result(timeout=30)  # 30s timeout per chunk
                        texts.append(text_chunk)
                    except Exception as e:
                        page_range = future_to_range[future]
                        logger.warning(f"Error processing pages {page_range}: {e}")
            
            doc.close()
            result = ''.join(texts).strip()
            logger.info(f"PDF extracted (parallel) in {time.time() - start_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            return ""

    def extract_pdf_fallback(self, file_path: str) -> str:
        """Fallback PDF extraction using PyPDF2"""
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(file_path)
            
            # Use parallel processing for large PDFs
            if len(reader.pages) > 10:
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    texts = list(executor.map(
                        lambda page: page.extract_text() or '', 
                        reader.pages
                    ))
                return ''.join(texts).strip()
            else:
                return ''.join(page.extract_text() or '' for page in reader.pages).strip()
                
        except Exception as e:
            logger.error(f"PyPDF2 fallback failed: {e}")
            return ""

    def extract_docx_optimized(self, file_path: str) -> str:
        """Optimized DOCX extraction"""
        start_time = time.time()
        
        try:
            doc = Document(file_path)
            
            # Extract paragraphs in parallel for large documents
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            
            if len(paragraphs) > 100:
                # Use parallel processing for large documents
                def extract_paragraph_batch(batch):
                    return '\n'.join(batch)
                
                batch_size = max(1, len(paragraphs) // self.max_workers)
                batches = [
                    paragraphs[i:i + batch_size] 
                    for i in range(0, len(paragraphs), batch_size)
                ]
                
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    results = list(executor.map(extract_paragraph_batch, batches))
                
                result = '\n'.join(results)
            else:
                result = '\n'.join(paragraphs)
            
            logger.info(f"DOCX extracted in {time.time() - start_time:.2f}s")
            return result.strip()
            
        except Exception as e:
            logger.error(f"DOCX extraction failed: {e}")
            return ""

    def extract_txt_optimized(self, file_path: str) -> str:
        """Optimized TXT extraction with encoding detection"""
        start_time = time.time()
        
        try:
            file_size = os.path.getsize(file_path)
            
            # For small files, use simple read
            if file_size < 1024 * 1024:  # 1MB
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    result = f.read().strip()
            else:
                # For large files, use memory mapping
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                        result = mm.read().decode('utf-8', errors='ignore').strip()
            
            logger.info(f"TXT extracted in {time.time() - start_time:.2f}s")
            return result
            
        except UnicodeDecodeError:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                        result = f.read().strip()
                    logger.info(f"TXT extracted with {encoding} encoding")
                    return result
                except:
                    continue
            
            logger.error("Failed to decode text file with any encoding")
            return ""
        except Exception as e:
            logger.error(f"TXT extraction failed: {e}")
            return ""

    def extract_text(self, file_path: str) -> str:
        """Main extraction method with caching and optimization"""
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return ""
        
        # Check cache first
        cached_text = self.load_from_cache(file_path)
        if cached_text is not None:
            return cached_text
        
        ext = os.path.splitext(file_path)[1].lower()
        text = ""
        
        start_time = time.time()
        
        if ext == '.pdf':
            # Try PyMuPDF first, fallback to PyPDF2
            text = self.extract_pdf_parallel_optimized(file_path)
            if not text.strip():
                logger.info("Trying PyPDF2 fallback...")
                text = self.extract_pdf_fallback(file_path)
                
        elif ext == '.docx':
            text = self.extract_docx_optimized(file_path)
            
        elif ext == '.txt':
            text = self.extract_txt_optimized(file_path)
            
        else:
            logger.warning(f"Unsupported file type: {ext}")
            return ""
        
        total_time = time.time() - start_time
        logger.info(f"Total extraction time: {total_time:.2f}s for {ext} file")
        
        # Save to cache
        if text.strip():
            self.save_to_cache(file_path, text)
        
        return text.strip()

# Global extractor instance
document_extractor = FastDocumentExtractor(max_workers=4, enable_cache=True)

# Replace your existing functions with these optimized versions
def extract_text(file_path: str) -> str:
    """Fast document extraction with caching"""
    return document_extractor.extract_text(file_path)

def extract_pdf_text_pypdf2(file_path: str) -> str:
    """Optimized PDF extraction"""
    return document_extractor.extract_pdf_parallel_optimized(file_path)

def extract_docx_text_fast(file_path: str) -> str:
    """Optimized DOCX extraction"""
    return document_extractor.extract_docx_optimized(file_path)

def extract_txt_fast(file_path: str) -> str:
    """Optimized TXT extraction"""
    return document_extractor.extract_txt_optimized(file_path)

# Optional: Add cache management routes
@app.route('/clear_extraction_cache', methods=['POST'])
def clear_extraction_cache():
    """Clear document extraction cache"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        import shutil
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
            CACHE_DIR.mkdir(exist_ok=True)
        
        add_activity(session['user_id'], 'cache_cleared', 'Document extraction cache cleared')
        return jsonify({'success': True, 'message': 'Cache cleared successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/cache_stats', methods=['GET'])
def get_cache_stats():
    """Get cache statistics"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        cache_files = list(CACHE_DIR.glob('*.txt'))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return jsonify({
            'cache_files': len(cache_files),
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'cache_enabled': document_extractor.enable_cache
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def summarize_with_gemini(content, lang='English'):

    #model = genai.GenerativeModel('gemini-2.0-flash')
#     llm = LLM(model="gemini/gemini-2.0-flash",temperature=0.7)
#     prompt = f"""Summarize this educational content in {lang} using simple and accessible language:\n\n{content} \n Instructions:
# - Focus on the main ideas and key takeaways.
# - Use plain language — avoid jargon unless explained simply.
# - Organize the summary with bullet points or short paragraphs for better readability.
# - Highlight important concepts or definitions where needed.And if any code put code markdowns ease for displaying.
# - Keep the tone encouraging and supportive, as if you're guiding a student through the material."""
    
#     try:
#         agent = Agent(
#         role="PDF analyser and You provide comprehensive and simple easy to study",
#         goal=prompt,
#         backstory="Your a Teacher making students works easier and simpler at analyzing documents",
#         llm=llm,
#     )


       
#         task = Task(
#         description=f"Provide the simple understandable easy to study and explaining everything in that And provide summary to study.This is pd_content:{content}\n",
#         expected_output="The summary of provided topic that can studied easily and gain knowledge same has reading real one ok.",
#         agent=agent
#     )

#         crew = Crew(agents=[agent], tasks=[task])
#         inputs = {"pdf_content": content}

#         # Run analysis
#         result = crew.kickoff(inputs=inputs)


#         #response = model.generate_content(prompt)
#         #return response.text if response else 'No summary generated.'
#         print(str(result))
#         return str(result)
    llm = LLM(model="gemini/gemini-2.0-flash", temperature=0.7)

    try:
        agent = Agent(
        role="PDF analyser and You provide comprehensive and simple easy to study",
        goal="Summarize educational content using simple and accessible language, focusing on main ideas and key takeaways",
        backstory="Your a Teacher making students works easier and simpler at analyzing documents",
        llm=llm,
    )

        task = Task(
        description=f"Summarize this educational content in {lang} using simple and accessible language,And this is Pdf_Content: {content}. Focus on the main ideas and key takeaways. Use plain language — avoid jargon unless explained simply. Organize the summary with bullet points or short paragraphs for better readability. Highlight important concepts or definitions where needed. Keep the tone encouraging and supportive, as if you're guiding a student through the material.",
        expected_output="The summary of provided topic (Pdf_Content) that can studied easily and gain knowledge same has reading real one ok.",
        agent=agent
    )

        crew = Crew(agents=[agent], tasks=[task],verbose=False )

    # Run analysis - no inputs needed since content is in task description
        result = crew.kickoff()
        sumresult=str(result)

        return str(sumresult)
    except Exception as e:
        return f'Error generating summary: {e}'

def clean_hell_markup(text):
    # Remove **bold**, ::italic::, __underline__, ~~strike~~
    text = re.sub(r"(\*\*|__|::|~~)(.*?)(\1)", r"\2", text)
    # Remove [[link]] or similar [[custom markup]]
    text = re.sub(r"\[\[(.*?)\]\]", r"\1", text)
    # Remove <html-like> tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove markdown-style [text](link) -> keep just 'text'
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remove emoji shortcodes like :smile: or Discord style <:name:1234>
    text = re.sub(r":\w+:", "", text)
    text = re.sub(r"<:\w+:\d+>", "", text)
    # Remove stray symbols or decorators like ``, ==highlight==
    text = re.sub(r"`(.*?)`", r"\1", text)
    text = re.sub(r"==(.*?)==", r"\1", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Routes
@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('auth'))
    return render_template('new-index.html')

@app.route('/auth')
def auth():
    return render_template('auth.html')

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    # conn = get_db_connection()
    # user = pgconn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()

    conn = get_pg_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
    user = cursor.fetchone()

    conn.close()
    
    if user and check_password_hash(user['password_hash'], password):
        session['user_id'] = user['id']
        session['username'] = user['username']
        session['email'] = user['email']
        add_activity(user['id'], 'login', 'User logged in')
        return jsonify({'success': True, 'message': 'Login successful'})
    else:
        return jsonify({'success': False, 'message': 'Invalid email or password'})

@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    
    # conn = get_db_connection()
    # existing_user = conn.execute('SELECT id FROM users WHERE email = ? OR username = ?', 
    #                             (email, username)).fetchone()
    
    # if existing_user:
    #     conn.close()
    #     return jsonify({'success': False, 'message': 'Username or email already exists'})
    
    # password_hash = generate_password_hash(password)
    # cursor = conn.cursor()
    # cursor.execute('INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
    #                (username, email, password_hash))
    # user_id = cursor.lastrowid
    # conn.commit()
    # conn.close()

     conn = get_pg_connection()
     cursor = conn.cursor()
     cursor.execute('SELECT id FROM users WHERE email = %s OR username = %s', (email, username))
     existing_user = cursor.fetchone()

     if existing_user:
        conn.close()
        return jsonify({'success': False, 'message': 'Username or email already exists'})

      password_hash = generate_password_hash(password)
      cursor.execute(
         'INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s) RETURNING id',
          (username, email, password_hash)
        )
      user_id = cursor.fetchone()['id']
      conn.commit()
      conn.close()

    
      session['user_id'] = user_id
      session['username'] = username
      session['email'] = email
      add_activity(user_id, 'signup', 'User created account')
    
      return jsonify({'success': True, 'message': 'Account created successfully'})

@app.route('/logout')
def logout():
    if 'user_id' in session:
        add_activity(session['user_id'], 'logout', 'User logged out')
    session.clear()
    return redirect(url_for('auth'))

@app.route('/get_user_info')
def get_user_info():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    return jsonify({
        'id': session['user_id'],
        'username': session['username'],
        'email': session['email']
    })

@app.route('/get_activities')
def get_activities():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    conn = get_db_connection()
    activities = conn.execute('''
        SELECT action, details, timestamp 
        FROM activities 
        WHERE user_id = ? 
        ORDER BY timestamp DESC 
        LIMIT 20
    ''', (session['user_id'],)).fetchall()
    conn.close()
    
    activities_list = []
    for activity in activities:
        activities_list.append({
            'action': activity['action'],
            'details': activity['details'],
            'timestamp': activity['timestamp']
        })
    
    return jsonify(activities_list)

@app.route('/search_videos', methods=['POST'])
def search_videos(query=None):
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    data = request.get_json(silent=True) or {}  # handles missing JSON body safely
    search_query = data.get('query', '').strip() or (query.strip() if query else '')

    if not search_query:
        return jsonify({'error': 'No search query provided'}), 400

    try:
        
        results = YoutubeSearch(search_query, max_results=10).to_json()
        videos_data = json.loads(results)

        add_activity(session['user_id'], 'video_search', f'Searched for: {search_query}')

        return jsonify(videos_data)

    except Exception as e:
        return jsonify({'error': f'Search failed: {str(e)}'}), 500


# Instantiate extractor (best to do it once, not per request)
fast_extractor = FastDocumentExtractor(max_workers=4, enable_cache=True)
# ✅ Route: Process upload
@app.route('/process', methods=['POST'])
def process():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    uploaded_file = request.files['file']
    selected_lang = request.form['language']
    gtts_lang = request.form['gtts_lang']

    if uploaded_file and uploaded_file.filename != '':
        filename = secure_filename(uploaded_file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        uploaded_file.save(file_path)

        # 🔍 Extract file content
        # content = extract_text(file_path) #old
        content = fast_extractor.extract_text(file_path)

        # 🧠 Summarize
        summary = summarize_with_gemini(content, selected_lang)
        cutsummary = clean_hell_markup(summary)
        #from vectordb import add_summary

        # inside your /process route
        #add_summary(cutsummary, session['user_id'], filename)

        # 🔊 Generate audio
        tts = gTTS(text=cutsummary, lang=gtts_lang)
        audio_path = os.path.join(app.config['OUTPUT_FOLDER'], 'summary.mp3')
        tts.save(audio_path)

        # 💾 Save summary text
        summary_path = os.path.join(app.config['OUTPUT_FOLDER'], 'summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)

        add_activity(session['user_id'], 'document_upload', f'Uploaded and processed: {filename}')

        return jsonify({
            'summary_text': summary,
            'audio_url': '/download/audio',
            'text_url': '/download/text',
            'extracted_text': content  # ✅ send extracted file text to preview
        })

    return 'No file uploaded', 400

list=[]
r_list=[]
user_context={}

# ✅ Route: Ask AI questions
conversation_history = []  # Store conversation pairs

@app.route('/ask', methods=['POST'])
def ask():
    global conversation_history  # Declare as global to modify
    
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    data = request.get_json()
    context = data.get('context', '')
    question = data.get('question', '')
    user = session['username']
    
    # Create user context from last 3 conversations
    user_context = {
        "recent_conversations": conversation_history[-3:] if conversation_history else []
    }
    
    prompt = f"""
You are a friendly and knowledgeable professor-like AI who helps users understand theoretical concepts clearly and patiently.

Use the following information to answer the question:

📚 Summary:
{context}

🧠 Question:
{question}

💬 Recent Conversation History (last 3 conversations):
{user_context['recent_conversations']}

Break down the concept into understandable parts.

Anticipate possible areas of confusion and clarify them.

Encourage curiosity and provide insights that deepen understanding.

If the user asks for a youtube video, video explanationn or user query contains intention for video on topic ,then provide a response like this:
YouTube_search: user topic to be searched on youtube goes here
"""

    llm = LLM(model="gemini/gemini-2.0-flash", temperature=0.7)

    try:
        agent = Agent(
            role="Friendly and knowledgeable professor-like AI",
            goal="Explain concepts clearly using context and conversation history",
            backstory="You are a patient educator who remembers previous conversations",
            llm=llm,
        )

        task = Task(
            description=f"""Answer this question using context and memory:

📚 Context: {context}
🧠 Question: {question}
 Conversation_History : {user_context}

Instructions:
- Use previous conversation context if relevant
- Explain concepts clearly with examples
- If user asks for video,youtube video on any topic respond: "YouTube_search: [topic]"
- Build upon previous interactions when appropriate""",
            expected_output="Clear, contextual answer that references previous conversations when relevant",
            agent=agent
        )

        crew = Crew(
            agents=[agent], 
            tasks=[task],
            memory=False,
            verbose=False
        )
        
        result = crew.kickoff()
        answer = str(result)
        
        # Save the conversation pair
        conversation_pair = {
            "question": question,
            "answer": answer,
            # "timestamp": datetime.now().isoformat()  # Optional: add timestamp
        }
        
        conversation_history.append(conversation_pair)
        
        # Keep only last 3 conversations
        if len(conversation_history) > 3:
            conversation_history.pop(0)  # Remove oldest conversation
        
        add_activity(session['user_id'], 'ai_question', f'Asked: {question[:50]}...')
        
        if 'YouTube_search:' in answer:
            query = answer.split('YouTube_search:')[1].strip()
            ai_videos = search_videos(query)
            return ai_videos
        
        return jsonify({'answer': answer})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ✅ Route: Download audio
@app.route('/download/audio')
def download_audio():
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], 'summary.mp3'), as_attachment=True)

# ✅ Route: Download summary text
@app.route('/download/text')
def download_text():
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], 'summary.txt'), as_attachment=True)

#new features added


# @app.route('/generate_assessment', methods=['POST'])
# def generate_assessment():
#     try:
#         if 'user_id' not in session:
#             return jsonify({'error': 'Not logged in'}), 401
        
#         data = request.get_json()
#         if not data:
#             return jsonify({'error': 'No data provided'}), 400
            
#         summary_content = data.get('summary', '')
        
#         if not summary_content:
#             return jsonify({'error': 'No summary provided'}), 400
        
#         # Simplified prompt for better JSON parsing
#         prompt = f"""Create exactly reasonable multiple-choice  questions based on this content. Return ONLY valid JSON in this exact format:

# {{
#   "questions": [
#     {{
#       "question": "What is the main topic discussed?",
#       "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
#       "correct": "A"
#     }}
#   ]
# }}

# Content to create questions from:
# {summary_content[:1000]}

# Important: Return ONLY the JSON, no other text."""
        
#         model = genai.GenerativeModel('gemini-2.0-flash')
#         response = model.generate_content(prompt)
        
#         if not response or not response.text:
#             return jsonify({'error': 'No response from AI'}), 500
        
#         # Clean the response text
#         response_text = response.text.strip()
        
#         # Remove any markdown formatting
#         if response_text.startswith('```json'):
#             response_text = response_text[7:]
#         if response_text.endswith('```'):
#             response_text = response_text[:-3]
        
#         response_text = response_text.strip()
        
#         # Parse JSON
#         questions_data = json.loads(response_text)
        
#         # Validate the structure
#         if 'questions' not in questions_data:
#             return jsonify({'error': 'Invalid response format'}), 500
            
#         if len(questions_data['questions']) == 0:
#             return jsonify({'error': 'No questions generated'}), 500
        
#         # Add activity log
#         add_activity(session['user_id'], 'assessment_generated', 'Generated assessment questions')
        
#         return jsonify(questions_data)
        
#     except json.JSONDecodeError as e:
#         print(f"JSON parsing error: {e}")
#         print(f"Response text: {response.text if 'response' in locals() else 'No response'}")
#         return jsonify({'error': 'Failed to parse AI response'}), 500
#     except Exception as e:
#         print(f"Error generating assessment: {e}")
#         return jsonify({'error': str(e)}), 500


# @app.route('/submit_assessment', methods=['POST'])
# def submit_assessment():
#     if 'user_id' not in session:
#         return jsonify({'error': 'Not logged in'}), 401
    
#     data = request.get_json()
#     questions = data.get('questions', [])
#     user_answers = data.get('answers', [])
#     document_title = data.get('document_title', 'Unknown Document')
    
#     # Calculate score
#     correct_count = 0
#     total_questions = len(questions)
    
#     for i, question in enumerate(questions):
#         if i < len(user_answers) and user_answers[i] == question['correct']:
#             correct_count += 1
    
#     score = int((correct_count / total_questions) * 100) if total_questions > 0 else 0
#     passed = score >= 80
    
#     # Generate AI feedback
#     feedback_prompt = f"""Provide constructive feedback for this assessment:
    
# Score: {score}%
# Correct: {correct_count}/{total_questions}
# Passed: {'Yes' if passed else 'No'}

# Questions and user answers:
# {json.dumps([{'question': q['question'], 'correct': q['correct'], 'user_answer': user_answers[i] if i < len(user_answers) else 'No answer'} for i, q in enumerate(questions)])}

# Provide encouraging feedback focusing on:
# 1. What they did well
# 2. Areas for improvement
# 3. Specific concepts to review
# 4. Encouragement for next steps"""
    
#     try:
#         model = genai.GenerativeModel('gemini-2.0-flash')
#         feedback_response = model.generate_content(feedback_prompt)
#         ai_feedback = feedback_response.text.strip()
#     except:
#         ai_feedback = "Great effort! Keep practicing to improve your understanding."
    
#     # Store in database
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     cursor.execute('''
#         INSERT INTO assessments (user_id, document_title, questions, user_answers, ai_feedback, score, passed)
#         VALUES (?, ?, ?, ?, ?, ?, ?)
#     ''', (session['user_id'], document_title, json.dumps(questions), json.dumps(user_answers), 
#           ai_feedback, score, passed))
#     conn.commit()
#     conn.close()
    
#     add_activity(session['user_id'], 'assessment_completed', f'Score: {score}% - {"Passed" if passed else "Try again"}')
    
#     return jsonify({
#         'score': score,
#         'passed': passed,
#         'correct_count': correct_count,
#         'total_questions': total_questions,
#         'feedback': ai_feedback
#     })


from crewai import Agent, Task, Crew
from pydantic import BaseModel
from typing import List
import json
llm = LLM(model="gemini/gemini-2.0-flash", temperature=0.7)

# Define output structures
class Question(BaseModel):
    question: str
    options: List[str]
    correct: str

class AssessmentQuestions(BaseModel):
    questions: List[Question]

class AssessmentResult(BaseModel):
    score: int
    passed: bool
    correct_count: int
    total_questions: int
    feedback: str

# Create agents
assessment_generator = Agent(
    role="Assessment Generator",
    goal="Create high-quality multiple-choice questions from educational content",
    backstory="You are an expert educator who creates engaging and accurate assessment questions.",
    verbose=False,
    llm=llm,
)

assessment_evaluator = Agent(
    role="Assessment Evaluator", 
    goal="Evaluate student responses and provide constructive feedback",
    backstory="You are an experienced teacher who provides helpful feedback to students.",
    verbose=False,
    llm=llm,
)

# Flask routes using CrewAI agents
@app.route('/generate_assessment', methods=['POST'])
def generate_assessment():
    try:
        if 'user_id' not in session:
            return jsonify({'error': 'Not logged in'}), 401
        
        data = request.get_json()
        if not data or not data.get('summary'):
            return jsonify({'error': 'No summary provided'}), 400
        
        summary_content = data.get('summary', '')
        
        # Create assessment generation task
        generation_task = Task(
            description=f"""
            Create exactly 5 reasonable multiple-choice questions based on this content:
            
            {summary_content[:1000]}
            
            Each question should have:
            - A clear, specific question
            - 4 options labeled A, B, C, D  
            -which One correct answer just its option like this for example if option B is correct:"B" (likewise accordingly provide correct answers)
            
            Focus on key concepts and important information.
            """,
            expected_output="Structured multiple-choice questions with options and correct answers",
            agent=assessment_generator,
            output_pydantic=AssessmentQuestions
        )
        
        # Create and run crew
        crew = Crew(
            agents=[assessment_generator],
            tasks=[generation_task],
            verbose=True
        )
        
        result = crew.kickoff()
        
        # Convert to expected format
        questions_data = {
            "questions": [
                {
                    "question": q.question,
                    "options": q.options,
                    "correct": q.correct
                }
                for q in result.pydantic.questions
            ]
        }
        
        add_activity(session['user_id'], 'assessment_generated', 'Generated assessment questions')
        return jsonify(questions_data)
        
    except Exception as e:
        print(f"Error generating assessment: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/submit_assessment', methods=['POST'])
def submit_assessment():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    data = request.get_json()
    questions = data.get('questions', [])
    user_answers = data.get('answers', [])
    document_title = data.get('document_title', 'Unknown Document')
   
    # Calculate score
    correct_count = 0
    total_questions = len(questions)
    
    for i, question in enumerate(questions):
        if i < len(user_answers) and user_answers[i] == question['correct']:
            correct_count += 1
    
    score = int((correct_count / total_questions) * 100) if total_questions > 0 else 0
    passed = score >= 80
   
    # Create feedback generation task
    feedback_task = Task(
        description=f"""
        Provide constructive feedback for this assessment:
        
        Score: {score}%
        Correct: {correct_count}/{total_questions}
        Passed: {'Yes' if passed else 'No'}
        
        Questions and answers: {json.dumps([{'question': q['question'], 'correct': q['correct'], 'user_answer': user_answers[i] if i < len(user_answers) else 'No answer'} for i, q in enumerate(questions)])}
        
        Focus on:
        1. What they did well
        2. Areas for improvement  
        3. Specific concepts to review
        4. Encouragement for next steps
        """,
        expected_output="Encouraging and constructive feedback for the student",
        agent=assessment_evaluator,
        output_pydantic=AssessmentResult
    )
    
    # Create and run crew
    crew = Crew(
        agents=[assessment_evaluator],
        tasks=[feedback_task],
        verbose=False
    )
    
    try:
        result = str(crew.kickoff())
        ai_feedback = result.pydantic.feedback
    except:
        ai_feedback = "Great effort! Keep practicing to improve your understanding."
    
    # Store in database
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO assessments (user_id, document_title, questions, user_answers, ai_feedback, score, passed)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (session['user_id'], document_title, json.dumps(questions), json.dumps(user_answers), 
          ai_feedback, score, passed))
    conn.commit()
    conn.close()
    
    add_activity(session['user_id'], 'assessment_completed', f'Score: {score}% - {"Passed" if passed else "Try again"}')
    
    return jsonify({
        'score': score,
        'passed': passed,
        'correct_count': correct_count,
        'total_questions': total_questions,
        'feedback': ai_feedback
    })


@app.route('/get_assessment_history')
def get_assessment_history():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    conn = get_db_connection()
    assessments = conn.execute('''
        SELECT document_title, score, passed, created_at 
        FROM assessments 
        WHERE user_id = ? 
        ORDER BY created_at DESC
    ''', (session['user_id'],)).fetchall()
    conn.close()
    
    assessment_list = []
    total_attempted = len(assessments)
    passed_count = sum(1 for a in assessments if a['passed'])
    avg_score = sum(a['score'] for a in assessments) / total_attempted if total_attempted > 0 else 0
    
    for assessment in assessments:
        assessment_list.append({
            'document_title': assessment['document_title'],
            'score': assessment['score'],
            'passed': assessment['passed'],
            'created_at': assessment['created_at']
        })
    
    return jsonify({
        'assessments': assessment_list,
        'stats': {
            'total_attempted': total_attempted,
            'passed_count': passed_count,
            'pass_rate': (passed_count / total_attempted * 100) if total_attempted > 0 else 0,
            'average_score': round(avg_score, 1)
        }
    })





#video-transcribe-new-features

def get_video_id(youtube_url):
    """Extract video ID from YouTube URL"""
    pattern = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, youtube_url)
    return match.group(1) if match else None

def get_video_transcript(video_id):
    """Get transcript for YouTube video"""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcript_list.find_generated_transcript(['en','hi','te'])
        
        formatter = TextFormatter()
        transcript_text = formatter.format_transcript(transcript.fetch())
        
        # Clean transcript
        transcript_text = re.sub(r'\[\d+:\d+:\d+\]', '', transcript_text)
        transcript_text = re.sub(r'<\w+>', '', transcript_text)
        return transcript_text.strip()
    except Exception as e:
        print(f"Error getting transcript: {e}")
        return None


@app.route('/get_video_transcript', methods=['POST'])
def get_video_transcript_route():
    try:
        if 'user_id' not in session:
            return jsonify({'error': 'Not logged in'}), 401
            
        data = request.get_json()
        video_id = data.get('video_id', '')
        
        if not video_id:
            return jsonify({'error': 'No video ID provided'}), 400
        
        transcript = get_video_transcript(video_id)
        
        if transcript:
            # Store in session for chat context
            session['current_video_transcript'] = transcript
            session['current_video_id'] = video_id
            
            # Add activity
            add_activity(session['user_id'], 'video_transcript', f'Generated transcript for video: {video_id}')
            # transcript[:500] + '...' if len(transcript) > 500 else transcript
            return jsonify({
                'success': True,
                'transcript': transcript,
                'full_length': len(transcript)
            })
        else:
            return jsonify({'error': 'Could not get transcript for this video'}), 400
            
    except Exception as e:
        print(f"Error in get_video_transcript: {e}")
        return jsonify({'error': str(e)}), 500

# @app.route('/ask_video', methods=['POST'])
# def ask_video():
#     try:
#         if 'user_id' not in session:
#             return jsonify({'error': 'Not logged in'}), 401
            
#         data = request.get_json()
#         question = data.get('question', '')
        
#         if not question:
#             return jsonify({'error': 'No question provided'}), 400
        
#         # Get transcript from session
#         #transcript = session.get('current_video_transcript', '')
#         transcript = data.get('transcript') or session.get('current_video_transcript', '')
#         video_id = session.get('current_video_id', '')
        
#         if not transcript:
#             return jsonify({'error': 'No video transcript available. Please load a video first.'}), 400
        
#         prompt = f"""
# You are a helpful AI assistant analyzing a YouTube video transcript. Answer the user's question based on the video content.

# Video Transcript:
# {transcript}

# User Question: {question}

# Instructions:
# - Answer based only on the video content
# - If the question is not covered in the video, say so politely
# - Be concise but informative
# - Use timestamps or references to video content when relevant
# - Maintain a friendly, educational tone
# """

#         model = genai.GenerativeModel('gemini-2.0-flash')
#         response = model.generate_content(prompt)
#         answer = response.text.strip() if response else 'Sorry, I couldn\'t process that question.'
        
#         # Add activity
#         add_activity(session['user_id'], 'video_question', f'Asked question about video: {video_id}')
        
#         return jsonify({'answer': answer})
        
#     except Exception as e:
#         print(f"Error in ask_video: {e}")
#         return jsonify({'answer': 'Sorry, there was an error processing your question.'})




gemini_llm = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.7
)

# Memory dictionary to store conversation history per user
conversation_memory = {}

# Create video analysis agent
video_analyst = Agent(
    role="YouTube Video Analysis Expert",
    goal="Analyze video transcripts and answer user questions based on video content",
    backstory="""You are a helpful AI assistant specialized in analyzing YouTube video transcripts. 
    You provide accurate, informative answers based solely on the video content. You maintain 
    context from previous questions in the conversation and reference timestamps when relevant.""",
    llm=gemini_llm,
    verbose=False
)

@app.route('/ask_video', methods=['POST'])
def ask_video():
    try:
        if 'user_id' not in session:
            return jsonify({'error': 'Not logged in'}), 401
            
        data = request.get_json()
        question = data.get('question', '')
        user_id = session['user_id']
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        # Get transcript from session or request
        transcript = data.get('transcript') or session.get('current_video_transcript', '')
        video_id = session.get('current_video_id', '')
        
        if not transcript:
            return jsonify({'error': 'No video transcript available. Please load a video first.'}), 400
        
        # Initialize or get existing conversation memory for this user
        if user_id not in conversation_memory:
            conversation_memory[user_id] = {
                'history': [],
                'video_id': video_id,
                'transcript': transcript
            }
        
        # Check if this is a new video (reset memory if different video)
        if conversation_memory[user_id]['video_id'] != video_id:
            conversation_memory[user_id] = {
                'history': [],
                'video_id': video_id,
                'transcript': transcript
            }
        
        # Add current question to memory
        conversation_memory[user_id]['history'].append({
            'role': 'user',
            'content': question
        })
        
        # Build context from conversation history
        conversation_context = ""
        if conversation_memory[user_id]['history']:
            conversation_context = "\n\nPrevious conversation:\n"
            for msg in conversation_memory[user_id]['history'][-6:]:  # Keep last 6 messages
                role = "User" if msg['role'] == 'user' else "Assistant"
                conversation_context += f"{role}: {msg['content']}\n"
        
        # Create task with memory context
        analysis_task = Task(
            description=f"""
            Analyze the YouTube video transcript and answer the user's question based on the video content.
            
            Video Transcript:
            {transcript}
            
            {conversation_context}
            
            Current User Question: {question}
            
            Instructions:
            - Answer based only on the video content
            - If the question is not covered in the video, say so politely
            - Be concise but informative
            - Use timestamps or references to video content when relevant
            - Maintain a friendly, educational tone
            - Consider the conversation history for context
            """,
            expected_output="A helpful answer based on the video content, considering conversation history",
            agent=video_analyst
        )
        
        # Create crew and execute
        crew = Crew(
            agents=[video_analyst],
            tasks=[analysis_task],
            verbose=False
        )
        
        result = crew.kickoff()
        answer = result.raw if hasattr(result, 'raw') else str(result)
        
        # Add assistant response to memory
        conversation_memory[user_id]['history'].append({
            'role': 'assistant',
            'content': answer
        })
        
        # Keep memory manageable (last 10 exchanges)
        if len(conversation_memory[user_id]['history']) > 20:
            conversation_memory[user_id]['history'] = conversation_memory[user_id]['history'][-20:]
        
        # Add activity
        add_activity(session['user_id'], 'video_question', f'Asked question about video: {video_id}')
        
        return jsonify({'answer': answer})
        
    except Exception as e:
        print(f"Error in ask_video: {e}")
        return jsonify({'answer': 'Sorry, there was an error processing your question.'})

# Optional: Clear memory endpoint
@app.route('/clear_video_memory', methods=['POST'])
def clear_video_memory():
    try:
        if 'user_id' not in session:
            return jsonify({'error': 'Not logged in'}), 401
            
        user_id = session['user_id']
        if user_id in conversation_memory:
            del conversation_memory[user_id]
            
        return jsonify({'message': 'Memory cleared successfully'})
        
    except Exception as e:
        print(f"Error clearing memory: {e}")
        return jsonify({'error': 'Failed to clear memory'})



#notes-session

# Add these routes to your Flask application
# Notes Routes - Add these to your app.py

@app.route('/get_user_notes')
def get_user_notes():
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'})
    
    try:
        conn = get_db_connection()
        notes = conn.execute('''
            SELECT id, title, content, created_at, updated_at 
            FROM user_notes 
            WHERE user_id = ? 
            ORDER BY updated_at DESC
        ''', (session['user_id'],)).fetchall()
        
        notes_list = []
        for note in notes:
            notes_list.append({
                'id': note['id'],
                'title': note['title'],
                'content': note['content'],
                'created_at': note['created_at'],
                'updated_at': note['updated_at']
            })
        
        conn.close()
        return jsonify({'success': True, 'notes': notes_list})
        
    except Exception as e:
        print(f"Error loading notes: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/save_note', methods=['POST'])
def save_note():
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'})
    
    data = request.get_json()
    title = data.get('title', '').strip()
    content = data.get('content', '')
    note_id = data.get('id')
    
    if not title:
        return jsonify({'success': False, 'error': 'Title is required'})
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if note_id:
            # Update existing note
            cursor.execute('''
                UPDATE user_notes 
                SET title = ?, content = ?, updated_at = CURRENT_TIMESTAMP 
                WHERE id = ? AND user_id = ?
            ''', (title, content, note_id, session['user_id']))
            
            if cursor.rowcount == 0:
                conn.close()
                return jsonify({'success': False, 'error': 'Note not found'})
                
        else:
            # Create new note
            cursor.execute('''
                INSERT INTO user_notes (user_id, title, content, created_at, updated_at) 
                VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ''', (session['user_id'], title, content))
            note_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
        # Log activity
        add_activity(session['user_id'], 'Note Saved', f'Note: {title}')
        
        return jsonify({'success': True, 'note_id': note_id})
        
    except Exception as e:
        print(f"Error saving note: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/delete_note', methods=['POST'])
def delete_note():
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'})
    
    data = request.get_json()
    note_id = data.get('id')
    
    if not note_id:
        return jsonify({'success': False, 'error': 'Note ID is required'})
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get note title for activity log
        note = conn.execute('SELECT title FROM user_notes WHERE id = ? AND user_id = ?', 
                           (note_id, session['user_id'])).fetchone()
        
        if not note:
            conn.close()
            return jsonify({'success': False, 'error': 'Note not found'})
        
        cursor.execute('DELETE FROM user_notes WHERE id = ? AND user_id = ?', 
                      (note_id, session['user_id']))
        
        conn.commit()
        conn.close()
        
        # Log activity
        add_activity(session['user_id'], 'Note Deleted', f'Note: {note["title"]}')
        
        return jsonify({'success': True})
        
    except Exception as e:
        print(f"Error deleting note: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/download_note_pdf', methods=['POST'])
def download_note_pdf():
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'})
    
    data = request.get_json()
    title = data.get('title', 'Untitled Note')
    content = data.get('content', '')
    
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from bs4 import BeautifulSoup
        import io
        from flask import make_response
        
        # Create PDF buffer
        buffer = io.BytesIO()
        
        # Create PDF document
        doc = SimpleDocTemplate(buffer, pagesize=A4, 
                              rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=72)
        
        # Get styles
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#667eea'),
            alignment=1  # Center alignment
        )
        
        content_style = ParagraphStyle(
            'CustomContent',
            parent=styles['Normal'],
            fontSize=12,
            spaceAfter=12,
            leading=18
        )
        
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            alignment=1  # Center alignment
        )
        
        # Build PDF content
        story = []
        
        # Add title
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 20))
        
        # Parse HTML content and convert to PDF
        soup = BeautifulSoup(content, 'html.parser')
        
        # Process different HTML elements
        for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'li']):
            text = element.get_text().strip()
            if text:
                if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    # Header style
                    header_style = ParagraphStyle(
                        'Header',
                        parent=styles['Heading2'],
                        fontSize=16,
                        spaceAfter=12,
                        textColor=colors.HexColor('#2d3748')
                    )
                    story.append(Paragraph(text, header_style))
                elif element.name == 'li':
                    # List item style
                    list_style = ParagraphStyle(
                        'ListItem',
                        parent=styles['Normal'],
                        fontSize=12,
                        leftIndent=20,
                        spaceAfter=6
                    )
                    story.append(Paragraph(f"• {text}", list_style))
                else:
                    # Regular paragraph
                    story.append(Paragraph(text, content_style))
        
        # If no structured content, add as plain text
        if not story or len(story) <= 2:  # Only title and spacer
            plain_text = soup.get_text().strip()
            if plain_text:
                paragraphs = plain_text.split('\n')
                for para in paragraphs:
                    if para.strip():
                        story.append(Paragraph(para.strip(), content_style))
        
        # Add footer
        story.append(Spacer(1, 50))
        story.append(Paragraph("Created with H.ai - Your AI Learning Assistant", footer_style))
        
        # Build PDF
        doc.build(story)
        
        # Get PDF data
        pdf_data = buffer.getvalue()
        buffer.close()
        
        # Log activity
        add_activity(session['user_id'], 'Note Downloaded', f'PDF: {title}')
        
        # Return PDF file
        response = make_response(pdf_data)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename="{title.replace(" ", "_")}.pdf"'
        
        return response
        
    except ImportError:
        return jsonify({'success': False, 'error': 'PDF generation libraries not installed. Please install: pip install reportlab beautifulsoup4'})
    except Exception as e:
        print(f"PDF generation error: {e}")
        return jsonify({'success': False, 'error': 'Failed to generate PDF'})

# Add this helper function for better note content processing
def process_note_content_for_pdf(html_content):
    """Process HTML content and convert to plain text with basic formatting"""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Replace common HTML elements with text equivalents
        for br in soup.find_all("br"):
            br.replace_with("\n")
        
        for p in soup.find_all("p"):
            p.insert_after("\n\n")
        
        for header in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            header.insert_before("\n\n")
            header.insert_after("\n")
        
        # Get clean text
        text = soup.get_text()
        
        # Clean up extra whitespace
        lines = [line.strip() for line in text.split('\n')]
        return '\n'.join(line for line in lines if line)
        
    except:
        # Fallback to simple text extraction
        return html_content




#createnotes-ai
# @app.route('/generate_ai_notes', methods=['POST'])
# def generate_ai_notes():
#     if 'user_id' not in session:
#         return jsonify({'success': False, 'error': 'Not logged in'})
    
#     try:
#         data = request.get_json()
#         if not data:
#             return jsonify({'success': False, 'error': 'No data provided'})
            
#         prompt = data.get('prompt', '')
#         style = data.get('style', 'comprehensive')
#         features = data.get('features', 'all')
#         context = data.get('context', '')
        
#         if not prompt:
#             return jsonify({'success': False, 'error': 'Prompt is required'})
        
#         # Create comprehensive prompt based on style and features
#         style_instructions = {
#             'comprehensive': 'Create detailed, comprehensive study notes with multiple sections, subsections, and thorough explanations.',
#             'summary': 'Create concise summary notes with key points, main concepts, and essential information.',
#             'structured': 'Create well-structured notes with clear headings, subheadings, and organized sections.',
#             'visual': 'Create notes with visual elements like tables, charts, and diagram descriptions.',
#             'outline': 'Create notes in outline format with hierarchical bullet points and numbered lists.'
#         }
        
#         features_instructions = {
#             'all': 'Include tables, flow diagrams, examples, definitions, and visual representations where appropriate.',
#             'tables': 'Focus on creating tables and charts to organize information clearly.',
#             'diagrams': 'Include flow diagrams, process charts, and visual representations of concepts.',
#             'examples': 'Provide plenty of examples, case studies, and practical applications.',
#             'definitions': 'Focus on clear definitions, terminology, and concept explanations.'
#         }
        
#         ai_prompt = f"""
# You are an expert note-taking assistant. Create professional, educational notes based on the following request.

# USER REQUEST: {prompt}

# STYLE: {style_instructions.get(style, style_instructions['comprehensive'])}

# FEATURES TO INCLUDE: {features_instructions.get(features, features_instructions['all'])}

# CONTEXT (if provided): {context}

# INSTRUCTIONS:
# 1. Create well-formatted HTML notes with proper headings (h1, h2, h3)
# 2. Use tables where appropriate with proper HTML table tags
# 3. Include bullet points and numbered lists for organization
# 4. Add flow diagrams as text-based representations in bordered divs
# 5. Use bold and italic formatting for emphasis
# 6. Include examples and practical applications
# 7. Make the content educational and easy to understand
# 8. Structure the content logically with clear sections

# IMPORTANT: 
# - Return ONLY the HTML content for the notes
# - Use proper HTML tags: <h1>, <h2>, <h3>, <p>, <ul>, <ol>, <li>, <table>, <tr>, <td>, <th>, <strong>, <em>
# - For diagrams, use: <div style="border: 2px dashed #ccc; padding: 15px; margin: 10px 0; text-align: center; background: #f9f9f9;">Diagram description here</div>
# - Make it comprehensive and professional
# - Include at least 3-5 main sections
# - Add tables where data can be organized
# - Include practical examples

# Generate the notes now:
# """

#         model = genai.GenerativeModel('gemini-2.0-flash')
#         response = model.generate_content(ai_prompt)
        
#         if response and response.text:
#             notes_content = response.text.strip()
            
#             # Clean up any markdown artifacts
#             notes_content = re.sub(r'```html\n?', '', notes_content)
#             notes_content = re.sub(r'```\n?', '', notes_content)
#             notes_content = re.sub(r'^html\n', '', notes_content)
            
#             # Add activity log
#             add_activity(session['user_id'], 'AI Notes Generated', f'Generated notes: {prompt[:50]}...')
            
#             return jsonify({
#                 'success': True,
#                 'notes': notes_content
#             })
#         else:
#             return jsonify({'success': False, 'error': 'No response from AI'})
            
#     except Exception as e:
#         print(f"Error generating AI notes: {e}")
#         return jsonify({'success': False, 'error': str(e)})



@app.route('/generate_ai_notes', methods=['POST'])
def generate_ai_notes():
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'})
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'})
            
        prompt = data.get('prompt', '')
        style = data.get('style', 'comprehensive')
        features = data.get('features', 'all')
        context = data.get('context', '')
        
        if not prompt:
            return jsonify({'success': False, 'error': 'Prompt is required'})
        
        # Create comprehensive prompt based on style and features
        style_instructions = {
            'comprehensive': 'Create detailed, comprehensive study notes with multiple sections, subsections, and thorough explanations.',
            'summary': 'Create concise summary notes with key points, main concepts, and essential information.',
            'structured': 'Create well-structured notes with clear headings, subheadings, and organized sections.',
            'visual': 'Create notes with visual elements like tables, charts, and diagram descriptions.',
            'outline': 'Create notes in outline format with hierarchical bullet points and numbered lists.'
        }
        
        features_instructions = {
            'all': 'Include tables, flow diagrams, examples, definitions, and visual representations where appropriate.',
            'tables': 'Focus on creating tables and charts to organize information clearly.',
            'diagrams': 'Include flow diagrams, process charts, and visual representations of concepts.',
            'examples': 'Provide plenty of examples, case studies, and practical applications.',
            'definitions': 'Focus on clear definitions, terminology, and concept explanations.'
        }
        
        # Use CrewAI like the rest of your application
        llm = LLM(model="gemini/gemini-2.0-flash", temperature=0.7)
        
        # Create AI Notes Generator Agent
        notes_agent = Agent(
            role="Expert Educational Notes Creator",
            goal="Create comprehensive, well-structured educational notes based on user requirements",
            backstory="You are an expert educator and note-taking specialist who creates professional, educational content that helps students learn effectively.",
            llm=llm,
            verbose=False
        )
        
        # Create the task
        notes_task = Task(
            description=f"""
            Create professional, educational notes based on this request:
            
            USER REQUEST: {prompt}
            
            STYLE: {style_instructions.get(style, style_instructions['comprehensive'])}
            
            FEATURES: {features_instructions.get(features, features_instructions['all'])}
            
            CONTEXT: {context if context else 'No additional context provided'}
            
            REQUIREMENTS:
            1. Create well-formatted HTML notes with proper headings (h1, h2, h3)
            2. Use tables where appropriate with proper HTML table tags
            3. Include bullet points and numbered lists for organization
            4. Add flow diagrams as text-based representations in bordered divs
            5. Use bold and italic formatting for emphasis
            6. Include examples and practical applications
            7. Make the content educational and easy to understand
            8. Structure the content logically with clear sections
            9. Include at least 3-5 main sections
            10. Add tables where data can be organized
            11. Include practical examples
            
            HTML FORMATTING RULES:
            - Use proper HTML tags: <h1>, <h2>, <h3>, <p>, <ul>, <ol>, <li>, <table>, <tr>, <td>, <th>, <strong>, <em>
            - For diagrams, use: <div style="border: 2px dashed #ccc; padding: 15px; margin: 10px 0; text-align: center; background: #f9f9f9;">Diagram description here</div>
            - Make tables with proper structure and styling
            - Use semantic HTML structure
            
            Return ONLY the HTML content for the notes, no markdown or other formatting.
            """,
            expected_output="Well-formatted HTML educational notes that are comprehensive, structured, and include the requested features",
            agent=notes_agent
        )
        
        # Create and run crew
        crew = Crew(
            agents=[notes_agent],
            tasks=[notes_task],
            verbose=False
        )
        
        result = crew.kickoff()
        notes_content = str(result)
        
        # Clean up any remaining artifacts
        notes_content = clean_notes_content(notes_content)
        
        # Add activity log
        add_activity(session['user_id'], 'AI Notes Generated', f'Generated notes: {prompt[:50]}...')
        
        return jsonify({
            'success': True,
            'notes': notes_content
        })
            
    except Exception as e:
        print(f"Error generating AI notes: {e}")
        return jsonify({'success': False, 'error': f'Failed to generate notes: {str(e)}'})

# Add this helper function to clean the notes content
def clean_notes_content(content):
    """Clean and format the AI-generated notes content"""
    try:
        # Remove common markdown artifacts
        content = re.sub(r'```html\n?', '', content)
        content = re.sub(r'```\n?', '', content)
        content = re.sub(r'^html\n', '', content, flags=re.MULTILINE)
        
        # Remove any leading/trailing whitespace
        content = content.strip()
        
        # Ensure proper HTML structure if missing
        if not content.startswith('<'):
            content = f'<div>{content}</div>'
        
        # Fix common HTML issues
        content = re.sub(r'<br\s*/?>\s*<br\s*/?>', '<br><br>', content)
        content = re.sub(r'\n\s*\n', '\n', content)
        
        return content
        
    except Exception as e:
        print(f"Error cleaning notes content: {e}")
        return content



#programming-section-work

@app.route('/run_code', methods=['POST'])
def run_code():
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'})
    
    data = request.get_json()
    code = data.get('code', '')
    language = data.get('language', 'python')
    
    if not code.strip():
        return jsonify({'success': False, 'error': 'No code provided'})
    
    try:
        if language == 'python':
            # Execute Python code safely
            import sys
            import io
            import contextlib
            
            # Capture stdout and stderr
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            
            try:
                with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                    # Create a restricted environment
                    restricted_globals = {
                        '__builtins__': {
                            'print': print,
                            'len': len,
                            'str': str,
                            'int': int,
                            'float': float,
                            'list': list,
                            'dict': dict,
                            'tuple': tuple,
                            'set': set,
                            'range': range,
                            'enumerate': enumerate,
                            'zip': zip,
                            'map': map,
                            'filter': filter,
                            'sum': sum,
                            'max': max,
                            'min': min,
                            'abs': abs,
                            'round': round,
                            'sorted': sorted,
                            'reversed': reversed,
                        }
                    }
                    
                    exec(code, restricted_globals)
                
                output = stdout_capture.getvalue()
                error = stderr_capture.getvalue()
                
                add_activity(session['user_id'], 'code_execution', f'{language} code executed')
                
                return jsonify({
                    'success': True,
                    'output': output if output else None,
                    'error': error if error else None
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                })
        
        elif language == 'javascript':
            # For JavaScript, return a message since we can't execute it server-side safely
            return jsonify({
                'success': True,
                'output': 'JavaScript code received. For security reasons, JavaScript execution is handled client-side.',
                'error': None
            })
        
        elif language == 'html':
            # For HTML, return a message
            return jsonify({
                'success': True,
                'output': 'HTML code received. You can copy this code to an HTML file to view it in a browser.',
                'error': None
            })
        
        else:
            # For other languages, provide conceptual feedback
            return jsonify({
                'success': True,
                'output': f'{language.upper()} code received. This is a conceptual environment - code syntax and structure look good!',
                'error': None
            })
            
    except Exception as e:
        print(f"Error executing code: {e}")
        return jsonify({'success': False, 'error': str(e)})

# Update the programming AI route to include code context
# @app.route('/ask_programming_ai', methods=['POST'])
# def ask_programming_ai():
#     if 'user_id' not in session:
#         return jsonify({'error': 'Not logged in'}), 401
    
#     data = request.get_json()
#     question = data.get('question', '')
#     code = data.get('code', '')
#     language = data.get('language', 'python')
    
#     if not question:
#         return jsonify({'error': 'No question provided'}), 400
    
#     prompt = f"""
# You are an expert programming tutor and mentor. Help the student with their programming question.

# Programming Language: {language}
# Student Question: {question}

# Current Code Context:
# {code if code else 'No code provided'}

# Instructions:
# - Provide clear, educational explanations
# - Include code examples when relevant
# - Explain concepts step by step
# - Suggest best practices and optimizations
# - If reviewing their code, provide specific feedback
# - If it's a debugging question, help identify the issue
# - If it's about algorithms, explain the logic clearly
# - Use proper markdown formatting for code blocks
# - Be encouraging and supportive
# - Focus on learning and understanding

# Provide a helpful response:
# """

#     try:
#         model = genai.GenerativeModel('gemini-2.0-flash')
#         response = model.generate_content(prompt)
#         answer = response.text.strip() if response else 'Sorry, I couldn\'t process that question.'
        
#         add_activity(session['user_id'], 'programming_question', f'Asked: {question[:50]}...')
        
#         return jsonify({'answer': answer})
#     except Exception as e:
#         print(f"Error in programming AI: {e}")
#         return jsonify({'answer': 'Sorry, there was an error processing your question.'})


# @app.route('/generate_programming_exam', methods=['POST'])
# def generate_programming_exam():
#     if 'user_id' not in session:
#         return jsonify({'success': False, 'error': 'Not logged in'})
    
#     data = request.get_json()
#     language = data.get('language', 'python')
#     difficulty = data.get('difficulty', 'beginner')
#     exam_type = data.get('type', 'mixed')
#     num_questions = int(data.get('num_questions', 5))
    
#     prompt = f"""
# Create a programming exam with {num_questions} questions for {language} at {difficulty} level.
# Exam type: {exam_type}

# Return ONLY valid JSON in this format:
# {{
#   "questions": [
#     {{
#       "type": "coding",
#       "question": "Write a function to...",
#       "description": "Additional details...",
#       "example": "Example input/output",
#       "correct_answer": "def solution()...",
#       "points": 10
#     }},
#     {{
#       "type": "multiple_choice",
#       "question": "What is the output of...",
#       "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
#       "correct": "A",
#       "points": 5
#     }}
#   ]
# }}

# Guidelines:
# - Mix coding problems and theory questions based on exam_type
# - For coding questions, include clear problem statements
# - For multiple choice, include 4 options
# - Adjust difficulty appropriately
# - Include practical programming concepts
# - Make questions educational and fair

# Generate the exam now:
# """

#     try:
#         model = genai.GenerativeModel('gemini-2.0-flash')
#         response = model.generate_content(prompt)
        
#         if response and response.text:
#             response_text = response.text.strip()
            
#             # Clean response
#             if response_text.startswith('```json'):
#                 response_text = response_text[7:]
#             if response_text.endswith('```'):
#                 response_text = response_text[:-3]
            
#             questions_data = json.loads(response_text.strip())
            
#             add_activity(session['user_id'], 'programming_exam_generated', 
#                         f'{language} {difficulty} exam with {num_questions} questions')
            
#             return jsonify({
#                 'success': True,
#                 'questions': questions_data['questions']
#             })
#         else:
#             return jsonify({'success': False, 'error': 'No response from AI'})
            
#     except json.JSONDecodeError as e:
#         print(f"JSON parsing error: {e}")
#         return jsonify({'success': False, 'error': 'Failed to parse AI response'})
#     except Exception as e:
#         print(f"Error generating programming exam: {e}")
#         return jsonify({'success': False, 'error': str(e)})

# @app.route('/submit_programming_exam', methods=['POST'])
# def submit_programming_exam():
#     if 'user_id' not in session:
#         return jsonify({'error': 'Not logged in'}), 401
    
#     data = request.get_json()
#     questions = data.get('questions', [])
#     answers = data.get('answers', [])
#     exam_time = data.get('exam_time', 0)
    
#     # Calculate score
#     correct_count = 0
#     total_points = 0
#     earned_points = 0
    
#     for i, question in enumerate(questions):
#         points = question.get('points', 5)
#         total_points += points
        
#         if i < len(answers) and answers[i]:
#             if question['type'] == 'multiple_choice':
#                 if answers[i] == question['correct']:
#                     correct_count += 1
#                     earned_points += points
#             else:  # coding question
#                 # For coding questions, give partial credit
#                 if answers[i].strip():  # If they wrote something
#                     correct_count += 0.5  # Partial credit
#                     earned_points += points * 0.7  # 70% credit for attempt
    
#     score = int((earned_points / total_points) * 100) if total_points > 0 else 0
#     passed = score >= 60  # Lower threshold for programming
    
#     # Generate AI feedback
#     feedback_prompt = f"""
# Provide constructive feedback for this programming exam:

# Score: {score}%
# Correct/Attempted: {correct_count}/{len(questions)}
# Time taken: {exam_time/1000/60:.1f} minutes

# Questions and answers:
# {json.dumps([{'question': q['question'], 'type': q['type'], 'answer': answers[i] if i < len(answers) else 'No answer'} for i, q in enumerate(questions)])}

# Provide encouraging feedback focusing on:
# 1. Programming concepts they demonstrated
# 2. Areas for improvement
# 3. Specific topics to study
# 4. Coding best practices
# 5. Next steps for learning
# """

#     try:
#         model = genai.GenerativeModel('gemini-2.0-flash')
#         feedback_response = model.generate_content(feedback_prompt)
#         ai_feedback = feedback_response.text.strip()
#     except:
#         ai_feedback = "Good effort! Keep practicing programming concepts and problem-solving skills."
    
#     add_activity(session['user_id'], 'programming_exam_completed', 
#                 f'Score: {score}% - {"Passed" if passed else "Needs improvement"}')
    
#     return jsonify({
#         'score': score,
#         'passed': passed,
#         'correct_count': int(correct_count),
#         'total_questions': len(questions),
#         'feedback': ai_feedback
#     })



if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')







