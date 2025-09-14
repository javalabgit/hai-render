from flask import Flask, render_template, request, send_file, jsonify, session, redirect, url_for
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
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
from docx import Document
from concurrent.futures import ThreadPoolExecutor
import mmap
from PyPDF2 import PdfReader
import uuid
import logging

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['CLASS_FILES_FOLDER'] = 'class_files'
app.secret_key = 'your-secret-key-here-change-this'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['CLASS_FILES_FOLDER'], exist_ok=True)

# Disable LiteLLM logging to prevent API key exposure
os.environ["LITELLM_LOG"] = "ERROR"
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

# Enhanced Database setup with all new features
def init_db():
    conn = sqlite3.connect('eduaccess.db', timeout=10)
    cursor = conn.cursor()
    
    # Enhanced Users table with role support
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'student',
            institution TEXT,
            subject TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
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
    
    # Assessments table
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

    # User Notes table
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

    # NEW TEACHER TABLES
    
    # Classes table (teacher classroom management)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS classes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            teacher_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            class_code TEXT UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (teacher_id) REFERENCES users (id) ON DELETE CASCADE
        )
    ''')
    
    # Class Members table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS class_members (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            class_id INTEGER NOT NULL,
            student_id INTEGER NOT NULL,
            status TEXT DEFAULT 'pending',
            joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (class_id) REFERENCES classes (id) ON DELETE CASCADE,
            FOREIGN KEY (student_id) REFERENCES users (id) ON DELETE CASCADE,
            UNIQUE(class_id, student_id)
        )
    ''')
    
    # Notifications table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            message TEXT NOT NULL,
            type TEXT DEFAULT 'info',
            class_id INTEGER,
            sender_id INTEGER,
            is_read BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
            FOREIGN KEY (class_id) REFERENCES classes (id) ON DELETE CASCADE,
            FOREIGN KEY (sender_id) REFERENCES users (id) ON DELETE CASCADE
        )
    ''')
    
    # Class Files table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS class_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            class_id INTEGER NOT NULL,
            teacher_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_size INTEGER,
            description TEXT,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (class_id) REFERENCES classes (id) ON DELETE CASCADE,
            FOREIGN KEY (teacher_id) REFERENCES users (id) ON DELETE CASCADE
        )
    ''')
    
    # Messages table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender_id INTEGER NOT NULL,
            recipient_id INTEGER NOT NULL,
            subject TEXT,
            message TEXT NOT NULL,
            is_read BOOLEAN DEFAULT FALSE,
            sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (sender_id) REFERENCES users (id) ON DELETE CASCADE,
            FOREIGN KEY (recipient_id) REFERENCES users (id) ON DELETE CASCADE
        )
    ''')
    
    # Announcements table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS announcements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            teacher_id INTEGER NOT NULL,
            class_id INTEGER,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            priority TEXT DEFAULT 'normal',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (teacher_id) REFERENCES users (id) ON DELETE CASCADE,
            FOREIGN KEY (class_id) REFERENCES classes (id) ON DELETE CASCADE
        )
    ''')

    conn.commit()
    conn.close()

# Initialize database
init_db()

def get_db_connection():
    conn = sqlite3.connect('eduaccess.db')
    conn.row_factory = sqlite3.Row
    return conn

def add_activity(user_id, action, details="", conn=None, cursor=None):
    close_conn = False
    if conn is None or cursor is None:
        conn = get_db_connection()
        cursor = conn.cursor()
        close_conn = True

    cursor.execute(
        'INSERT INTO activities (user_id, action, details) VALUES (?, ?, ?)',
        (user_id, action, details)
    )

    if close_conn:
        conn.commit()
        conn.close()




def create_notification(user_id, title, message, notification_type='info', class_id=None, sender_id=None, conn=None, cursor=None):
    close_conn = False
    if conn is None or cursor is None:
        conn = get_db_connection()
        cursor = conn.cursor()
        close_conn = True

    cursor.execute('''
        INSERT INTO notifications (user_id, title, message, type, class_id, sender_id)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (user_id, title, message, notification_type, class_id, sender_id))

    if close_conn:
        conn.commit()
        conn.close()




# Text extraction functions (optimized for speed)
def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.pdf':
        return extract_pdf_text_pypdf2(file_path)
    elif ext == '.docx':
        return extract_docx_text_fast(file_path)
    elif ext == '.txt':
        return extract_txt_fast(file_path)

    return ''

def extract_pdf_text_pypdf2(file_path):
    try:
        reader = PdfReader(file_path)
        # Use ThreadPoolExecutor for faster processing
        with ThreadPoolExecutor(max_workers=4) as executor:
            pages = [executor.submit(lambda p: p.extract_text() or '', page) for page in reader.pages]
            return ''.join(page.result() for page in pages)
    except Exception as e:
        print(f"Error extracting PDF with PyPDF2: {e}")
        return ''

def extract_docx_text_fast(file_path):
    doc = Document(file_path)
    return '\n'.join(p.text for p in doc.paragraphs if p.text.strip())

def extract_txt_fast(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                return mmapped_file.read().decode('utf-8').strip()
    except:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()

# AI Integration using CrewAI
from crewai import Agent, Task, Crew, LLM

def summarize_with_gemini(content, lang='English'):
    llm = LLM(model="gemini/gemini-2.0-flash", temperature=0.7)

    try:
        agent = Agent(
            role="PDF analyser and comprehensive study guide creator",
            goal="Summarize educational content using simple and accessible language, focusing on main ideas and key takeaways",
            backstory="You are a teacher making students' work easier and simpler by analyzing documents",
            llm=llm,
        )

        task = Task(
            description=f"Summarize this educational content in {lang} using simple and accessible language. Focus on the main ideas and key takeaways. Use plain language — avoid jargon unless explained simply. Organize the summary with bullet points or short paragraphs for better readability. Highlight important concepts or definitions where needed. Keep the tone encouraging and supportive, as if you're guiding a student through the material. Content: {content}",
            expected_output="The summary of provided topic that can be studied easily and gain knowledge same as reading the original document",
            agent=agent
        )

        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        result = crew.kickoff()
        return str(result)
    except Exception as e:
        return f'Error generating summary: {e}'


@app.route('/generate_audio', methods=['POST'])
def generate_audio():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    data = request.get_json()
    text = data.get('text')
    lang = data.get('lang', 'en')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        filename = f"summary_{uuid.uuid4().hex}.mp3"
        filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        tts = gTTS(text=text, lang=lang)
        tts.save(filepath)

        add_activity(session['user_id'], 'generate_audio', f'Generated audio file: {filename}')
        return jsonify({'success': True, 'file': url_for('download_file', filename=filename)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/outputs/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], filename), as_attachment=True)



def clean_hell_markup(text):
    # Remove various markup patterns
    text = re.sub(r"(\*\*|__|::|~~)(.*?)(\1)", r"\2", text)
    text = re.sub(r"\[\[(.*?)\]\]", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r":\w+:", "", text)
    text = re.sub(r"<:\w+:\d+>", "", text)
    text = re.sub(r"`(.*?)`", r"\1", text)
    text = re.sub(r"==(.*?)==", r"\1", text)
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

# Enhanced login with role support
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    role = data.get('role')
    
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
    conn.close()
    
    if user and check_password_hash(user['password_hash'], password):
        if user['role'] != role:
            return jsonify({'success': False, 'message': f'This account is registered as a {user["role"]}, not {role}'})
        
        session['user_id'] = user['id']
        session['username'] = user['username']
        session['email'] = user['email']
        session['role'] = user['role']
        add_activity(user['id'], 'login', f'User logged in as {role}')
        return jsonify({'success': True, 'message': 'Login successful'})
    else:
        return jsonify({'success': False, 'message': 'Invalid email or password'})

# Enhanced signup with role and additional fields
@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    role = data.get('role', 'student')
    institution = data.get('institution')
    subject = data.get('subject')
    
    conn = get_db_connection()
    existing_user = conn.execute('SELECT id FROM users WHERE email = ? OR username = ?', 
                                (email, username)).fetchone()
    
    if existing_user:
        conn.close()
        return jsonify({'success': False, 'message': 'Username or email already exists'})
    
    password_hash = generate_password_hash(password)
    cursor = conn.cursor()
    cursor.execute('''INSERT INTO users (username, email, password_hash, role, institution, subject) 
                     VALUES (?, ?, ?, ?, ?, ?)''',
                   (username, email, password_hash, role, institution, subject))
    user_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    session['user_id'] = user_id
    session['username'] = username
    session['email'] = email
    session['role'] = role
    add_activity(user_id, 'signup', f'User created {role} account')
    
    return jsonify({'success': True, 'message': 'Account created successfully'})

@app.route('/logout')
def logout():
    if 'user_id' in session:
        add_activity(session['user_id'], 'logout', 'User logged out')
    session.clear()
    return redirect(url_for('auth'))

# Enhanced user info with role
@app.route('/get_user_info')
def get_user_info():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    return jsonify({
        'id': session['user_id'],
        'username': session['username'],
        'email': session['email'],
        'role': session.get('role', 'student')
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

# Process documents with extracted text support (enhanced for speed)
@app.route('/process', methods=['POST'])
def process():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    uploaded_file = request.files['file']
    selected_lang = request.form['language']
    gtts_lang = request.form['gtts_lang']
    include_audio = request.form.get('include_audio', 'true') == 'true'
    extracted_text = request.form.get('extracted_text', '')  # Get client-extracted text

    if uploaded_file and uploaded_file.filename != '':
        filename = secure_filename(uploaded_file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        uploaded_file.save(file_path)

        # Use client-extracted text if available, otherwise extract server-side
        if extracted_text:
            content = extracted_text
        else:
            content = extract_text(file_path)

        # Summarize
        summary = summarize_with_gemini(content, selected_lang)
        cutsummary = clean_hell_markup(summary)

        response_data = {
            'summary_text': summary,
            'text_url': '/download/text',
            'extracted_text': content
        }

        # Generate audio only if requested
        if include_audio:
            tts = gTTS(text=cutsummary, lang=gtts_lang)
            audio_path = os.path.join(app.config['OUTPUT_FOLDER'], 'summary.mp3')
            tts.save(audio_path)
            response_data['audio_url'] = '/download/audio'

        # Save summary text
        summary_path = os.path.join(app.config['OUTPUT_FOLDER'], 'summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)

        add_activity(session['user_id'], 'document_upload', f'Uploaded and processed: {filename}')

        return jsonify(response_data)

    return 'No file uploaded', 400

# Conversation history for AI
conversation_history = []

@app.route('/ask', methods=['POST'])
def ask():
    global conversation_history
    
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    data = request.get_json()
    context = data.get('context', '')
    question = data.get('question', '')
    
    # Create user context from last 3 conversations
    user_context = {
        "recent_conversations": conversation_history[-3:] if conversation_history else []
    }
    
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
        }
        
        conversation_history.append(conversation_pair)
        
        # Keep only last 3 conversations
        if len(conversation_history) > 3:
            conversation_history.pop(0)
        
        add_activity(session['user_id'], 'ai_question', f'Asked: {question[:50]}...')
        
        if 'YouTube_search:' in answer:
            query = answer.split('YouTube_search:')[1].strip()
            ai_videos = search_videos(query)
            return ai_videos
        
        return jsonify({'answer': answer})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/search_videos', methods=['POST'])
def search_videos(query=None):
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    data = request.get_json(silent=True) or {}
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

@app.route('/download/audio')
def download_audio():
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], 'summary.mp3'), as_attachment=True)

@app.route('/download/text')
def download_text():
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], 'summary.txt'), as_attachment=True)

# Assessment functions using CrewAI
from pydantic import BaseModel
from typing import List

llm = LLM(model="gemini/gemini-2.0-flash", temperature=0.7)

class Question(BaseModel):
    question: str
    options: List[str]
    correct: str

class AssessmentQuestions(BaseModel):
    questions: List[Question]

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

@app.route('/generate_assessment', methods=['POST'])
def generate_assessment():
    try:
        if 'user_id' not in session:
            return jsonify({'error': 'Not logged in'}), 401
        
        data = request.get_json()
        if not data or not data.get('summary'):
            return jsonify({'error': 'No summary provided'}), 400
        
        summary_content = data.get('summary', '')
        
        generation_task = Task(
            description=f"""
            Create exactly 5 reasonable multiple-choice questions based on this content:
            
            {summary_content[:1000]}
            
            Each question should have:
            - A clear, specific question
            - 4 options labeled A, B, C, D  
            - One correct answer just its option like this for example if option B is correct:"B"
            
            Focus on key concepts and important information.
            """,
            expected_output="Structured multiple-choice questions with options and correct answers",
            agent=assessment_generator,
            output_pydantic=AssessmentQuestions
        )
        
        crew = Crew(
            agents=[assessment_generator],
            tasks=[generation_task],
            verbose=False
        )
        
        result = crew.kickoff()
        
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
   
    try:
        ai_feedback = "Good effort! Keep practicing to improve your understanding."
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

# Video transcript functions with enhanced video chat agent
def get_video_id(youtube_url):
    pattern = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, youtube_url)
    return match.group(1) if match else None

def get_video_transcript(video_id):
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
            session['current_video_transcript'] = transcript
            session['current_video_id'] = video_id
            add_activity(session['user_id'], 'video_transcript', f'Generated transcript for video: {video_id}')
            
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

# Enhanced Video chat with memory and crew agent
conversation_memory = {}

video_analyst = Agent(
    role="YouTube Video Analysis Expert and Educational Assistant",
    goal="Analyze video transcripts, answer questions, and provide educational guidance based on video content",
    backstory="""You are an expert video content analyzer and educational assistant who specializes in 
    understanding YouTube educational videos. You can extract key insights, explain complex concepts, 
    and answer specific questions about video content with precision and educational value.""",
    llm=llm,
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
        
        conversation_memory[user_id]['history'].append({
            'role': 'user',
            'content': question
        })
        
        # Build context from conversation history
        conversation_context = ""
        if conversation_memory[user_id]['history']:
            conversation_context = "\n\nPrevious conversation:\n"
            for msg in conversation_memory[user_id]['history'][-6:]:
                role = "User" if msg['role'] == 'user' else "Assistant"
                conversation_context += f"{role}: {msg['content']}\n"
        
        analysis_task = Task(
            description=f"""
            You are an expert educational video analyst with access to the full transcript of a YouTube video.
            Your task is to provide comprehensive, accurate, and educational answers based on the video content.
            
            Video Transcript (Full Content):
            {transcript}
            
            {conversation_context}
            
            Current User Question: {question}
            
            Instructions:
            - Analyze the video transcript thoroughly to answer the user's question
            - Provide specific details, examples, and explanations from the video content
            - If the question asks about specific topics, quote relevant parts from the transcript
            - If the question cannot be answered from the video, clearly state this and provide general educational context
            - Use timestamps or references to video sections when possible
            - Maintain an educational, friendly tone
            - Consider the conversation history for better context understanding
            - If asked about specific concepts, provide detailed explanations with examples from the video
            - Help the user understand complex topics by breaking them down into simpler concepts
            """,
            expected_output="A comprehensive, educational answer based on the video transcript that addresses the user's specific question with relevant details and examples",
            agent=video_analyst
        )
        
        crew = Crew(
            agents=[video_analyst],
            tasks=[analysis_task],
            verbose=False
        )
        
        result = crew.kickoff()
        answer = result.raw if hasattr(result, 'raw') else str(result)
        
        conversation_memory[user_id]['history'].append({
            'role': 'assistant',
            'content': answer
        })
        
        # Keep memory manageable (last 10 exchanges)
        if len(conversation_memory[user_id]['history']) > 20:
            conversation_memory[user_id]['history'] = conversation_memory[user_id]['history'][-20:]
        
        add_activity(session['user_id'], 'video_question', f'Asked question about video: {video_id}')
        
        return jsonify({'answer': answer})
        
    except Exception as e:
        print(f"Error in ask_video: {e}")
        return jsonify({'answer': 'Sorry, there was an error processing your question about the video.'})

# Notes management routes
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
            cursor.execute('''
                UPDATE user_notes 
                SET title = ?, content = ?, updated_at = CURRENT_TIMESTAMP 
                WHERE id = ? AND user_id = ?
            ''', (title, content, note_id, session['user_id']))
            
            if cursor.rowcount == 0:
                conn.close()
                return jsonify({'success': False, 'error': 'Note not found'})
                
        else:
            cursor.execute('''
                INSERT INTO user_notes (user_id, title, content, created_at, updated_at) 
                VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ''', (session['user_id'], title, content))
            note_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
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
        
        note = conn.execute('SELECT title FROM user_notes WHERE id = ? AND user_id = ?', 
                           (note_id, session['user_id'])).fetchone()
        
        if not note:
            conn.close()
            return jsonify({'success': False, 'error': 'Note not found'})
        
        cursor.execute('DELETE FROM user_notes WHERE id = ? AND user_id = ?', 
                      (note_id, session['user_id']))
        
        conn.commit()
        conn.close()
        
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
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from bs4 import BeautifulSoup
        import io
        from flask import make_response

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, 
                              rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=72)
        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#667eea'),
            alignment=1
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
            alignment=1
        )

        story = []
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 20))

        soup = BeautifulSoup(content, 'html.parser')

        for element in soup.find_all(recursive=False):
            tag = element.name

            if tag in ['h1','h2','h3','h4','h5','h6']:
                header_style = ParagraphStyle(
                    'Header',
                    parent=styles['Heading2'],
                    fontSize=16,
                    spaceAfter=12,
                    textColor=colors.HexColor('#2d3748')
                )
                story.append(Paragraph(element.get_text(), header_style))

            elif tag == 'p':
                story.append(Paragraph(element.decode_contents(), content_style))

            elif tag in ['ul','ol']:
                for li in element.find_all('li'):
                    list_style = ParagraphStyle(
                        'ListItem',
                        parent=styles['Normal'],
                        fontSize=12,
                        leftIndent=20,
                        spaceAfter=6
                    )
                    story.append(Paragraph(f"• {li.decode_contents()}", list_style))

        # Add footer
        story.append(Spacer(1, 50))
        story.append(Paragraph("Created with H.ai - Your AI Learning Assistant", footer_style))

        doc.build(story)

        pdf_data = buffer.getvalue()
        buffer.close()

        add_activity(session['user_id'], 'Note Downloaded', f'PDF: {title}')

        response = make_response(pdf_data)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename="{title.replace(" ", "_")}.pdf"'

        return response

    except Exception as e:
        print(f"PDF generation error: {e}")
        return jsonify({'success': False, 'error': 'Failed to generate PDF'})

# AI Notes generation
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
        
        llm = LLM(model="gemini/gemini-2.0-flash", temperature=0.7)
        
        notes_agent = Agent(
            role="Expert Educational Notes Creator",
            goal="Create comprehensive, well-structured educational notes based on user requirements",
            backstory="You are an expert educator and note-taking specialist who creates professional, educational content that helps students learn effectively.",
            llm=llm,
            verbose=False
        )
        
        notes_task = Task(
            description=f"""
            Create professional, educational notes based on this request:
            
            USER REQUEST: {prompt}
            
            CONTEXT: {context if context else 'No additional context provided'}
            
            REQUIREMENTS:
            1. Provide the explanation in beautiful styling and new approaches
            2. Best Way is Providing Animation of The topic With css Stylings , @Media animation With correct without any errors
            3. Make sure to provide relevant or suitable creativity figures with css and @media best colors
            4. All content should be modern ,and best presentation
            
            HTML FORMATTING RULES:
            - Use best animations and diagrams using media and css styling
            - Use proper HTML tags: <h1>, <h2>, <h3>, <p>, <ul>, <ol>, <li>, <table>, <tr>, <td>, <th>, <strong>, <em>
            - For diagrams, use: <div style="border: 2px dashed #ccc; padding: 15px; margin: 10px 0; text-align: center; background: #f9f9f9;">Diagram description here</div>
            - Make tables with proper structure and styling
            - Use semantic HTML structure
            
            Return ONLY the HTML content for the notes, no markdown or other formatting.
            """,
            expected_output="Well-formatted HTML educational notes that are comprehensive, structured, and include the requested features",
            agent=notes_agent
        )
        
        crew = Crew(
            agents=[notes_agent],
            tasks=[notes_task],
            verbose=False
        )
        
        result = crew.kickoff()
        notes_content = str(result)
        
        # Clean up any remaining artifacts
        notes_content = clean_notes_content(notes_content)
        
        add_activity(session['user_id'], 'AI Notes Generated', f'Generated notes: {prompt[:50]}...')
        
        return jsonify({
            'success': True,
            'notes': notes_content
        })
            
    except Exception as e:
        print(f"Error generating AI notes: {e}")
        return jsonify({'success': False, 'error': f'Failed to generate notes: {str(e)}'})

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

# TEACHER DASHBOARD AND CLASS MANAGEMENT ROUTES

@app.route('/get_all_students')
def get_all_students():
    if 'user_id' not in session or session.get('role') != 'teacher':
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        conn = get_db_connection()
        students = conn.execute('''
            SELECT id, username, email, institution, created_at 
            FROM users 
            WHERE role = 'student'
            ORDER BY username
        ''').fetchall()
        
        students_list = []
        for student in students:
            students_list.append({
                'id': student['id'],
                'username': student['username'],
                'email': student['email'],
                'institution': student['institution'],
                'created_at': student['created_at']
            })
        
        conn.close()
        return jsonify({'success': True, 'students': students_list})
        
    except Exception as e:
        print(f"Error loading students: {e}")
        return jsonify({'success': False, 'error': str(e)})



@app.route('/create_class', methods=['POST'])
def create_class():
    if 'user_id' not in session or session.get('role') != 'teacher':
        return jsonify({'error': 'Access denied'}), 403
    
    conn = None
    try:
        data = request.get_json()
        name = data.get('name')
        description = data.get('description', '')
        selected_students = data.get('students', [])
        
        if not name:
            return jsonify({'success': False, 'error': 'Class name is required'})
        
        # Generate unique class code
        class_code = str(uuid.uuid4())[:8].upper()
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create class
        cursor.execute('''
            INSERT INTO classes (teacher_id, name, description, class_code)
            VALUES (?, ?, ?, ?)
        ''', (session['user_id'], name, description, class_code))
        
        class_id = cursor.lastrowid
        
        # Add selected students to class using the SAME connection
        for student_id in selected_students:
            cursor.execute('''
                INSERT INTO class_members (class_id, student_id, status)
                VALUES (?, ?, 'pending')
            ''', (class_id, student_id))
            
            # Create notification using the SAME connection
            cursor.execute('''
                INSERT INTO notifications (user_id, title, message, type, class_id, sender_id)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (student_id, 'Class Invitation', 
                  f'You have been invited to join the class "{name}" by {session["username"]}',
                  'class_invite', class_id, session['user_id']))
        
        # Add activity using the SAME connection
        cursor.execute('''
            INSERT INTO activities (user_id, action, details) 
            VALUES (?, ?, ?)
        ''', (session['user_id'], 'class_created', f'Created class: {name}'))
        
        # Commit all changes at once
        conn.commit()
        
        return jsonify({
            'success': True, 
            'class_id': class_id,
            'class_code': class_code
        })
        
    except Exception as e:
        print(f"Error creating class: {e}")
        if conn:
            conn.rollback()  # Rollback on error
        return jsonify({'success': False, 'error': str(e)})
    
    finally:
        if conn:
            conn.close()  # Always close connection




@app.route('/get_teacher_classes')
def get_teacher_classes():
    if 'user_id' not in session or session.get('role') != 'teacher':
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        conn = get_db_connection()
        classes = conn.execute('''
            SELECT c.*, COUNT(cm.student_id) as student_count
            FROM classes c
            LEFT JOIN class_members cm ON c.id = cm.class_id AND cm.status = 'accepted'
            WHERE c.teacher_id = ?
            GROUP BY c.id
            ORDER BY c.created_at DESC
        ''', (session['user_id'],)).fetchall()
        
        classes_list = []
        for class_item in classes:
            classes_list.append({
                'id': class_item['id'],
                'name': class_item['name'],
                'description': class_item['description'],
                'class_code': class_item['class_code'],
                'student_count': class_item['student_count'],
                'created_at': class_item['created_at']
            })
        
        conn.close()
        return jsonify({'success': True, 'classes': classes_list})
        
    except Exception as e:
        print(f"Error loading teacher classes: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_class_details/<int:class_id>')
def get_class_details(class_id):
    if 'user_id' not in session or session.get('role') != 'teacher':
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        conn = get_db_connection()
        
        # Verify teacher owns this class
        class_info = conn.execute('''
            SELECT * FROM classes WHERE id = ? AND teacher_id = ?
        ''', (class_id, session['user_id'])).fetchone()
        
        if not class_info:
            return jsonify({'error': 'Class not found'}), 404
        
        # Get class members
        members = conn.execute('''
            SELECT u.id, u.username, u.email, cm.status, cm.joined_at
            FROM class_members cm
            JOIN users u ON cm.student_id = u.id
            WHERE cm.class_id = ?
            ORDER BY cm.joined_at DESC
        ''', (class_id,)).fetchall()
        
        # Get class files
        files = conn.execute('''
            SELECT * FROM class_files
            WHERE class_id = ?
            ORDER BY uploaded_at DESC
        ''', (class_id,)).fetchall()
        
        conn.close()
        
        return jsonify({
            'success': True,
            'class': dict(class_info),
            'members': [dict(member) for member in members],
            'files': [dict(file) for file in files]
        })
        
    except Exception as e:
        print(f"Error loading class details: {e}")
        return jsonify({'success': False, 'error': str(e)})

# @app.route('/upload_class_file', methods=['POST'])
# def upload_class_file():
#     if 'user_id' not in session or session.get('role') != 'teacher':
#         return jsonify({'error': 'Access denied'}), 403
    
#     try:
#         class_id = request.form.get('class_id')
#         description = request.form.get('description', '')
#         file = request.files['file']
        
#         if not file:
#             return jsonify({'success': False, 'error': 'No file provided'})
        
#         # Verify teacher owns this class
#         conn = get_db_connection()
#         class_check = conn.execute('''
#             SELECT id FROM classes WHERE id = ? AND teacher_id = ?
#         ''', (class_id, session['user_id'])).fetchone()
        
#         if not class_check:
#             return jsonify({'success': False, 'error': 'Access denied'})
        
#         # Save file
#         filename = secure_filename(file.filename)
#         unique_filename = f"{uuid.uuid4()}_{filename}"
#         file_path = os.path.join(app.config['CLASS_FILES_FOLDER'], unique_filename)
#         file.save(file_path)
        
#         # Store file info in database
#         cursor = conn.cursor()
#         cursor.execute('''
#             INSERT INTO class_files 
#             (class_id, teacher_id, filename, original_filename, file_path, file_size, description)
#             VALUES (?, ?, ?, ?, ?, ?, ?)
#         ''', (class_id, session['user_id'], unique_filename, filename, 
#               file_path, os.path.getsize(file_path), description))
        
#         file_id = cursor.lastrowid
        
#         # Notify all class members
#         members = conn.execute('''
#             SELECT student_id FROM class_members 
#             WHERE class_id = ? AND status = 'accepted'
#         ''', (class_id,)).fetchall()
        
#         class_name = conn.execute('SELECT name FROM classes WHERE id = ?', (class_id,)).fetchone()['name']
        
#         for member in members:
#             create_notification(
#                 member['student_id'],
#                 'New File Uploaded',
#                 f'A new file "{filename}" has been uploaded to class "{class_name}"',
#                 'file_upload',
#                 class_id,
#                 session['user_id']
#             )
        
#         conn.commit()
#         conn.close()
        
#         add_activity(session['user_id'], 'file_uploaded', f'Uploaded file to class: {filename}')
        
#         return jsonify({'success': True, 'file_id': file_id})
        
#     except Exception as e:
#         print(f"Error uploading class file: {e}")
#         return jsonify({'success': False, 'error': str(e)})


@app.route('/upload_class_file', methods=['POST'])
def upload_class_file():
    if 'user_id' not in session or session.get('role') != 'teacher':
        return jsonify({'error': 'Access denied'}), 403

    conn = None
    try:
        class_id = request.form.get('class_id')
        description = request.form.get('description', '')
        file = request.files['file']

        if not file:
            return jsonify({'success': False, 'error': 'No file provided'})

        conn = get_db_connection()
        cursor = conn.cursor()

        # Verify teacher owns this class
        class_check = conn.execute('''
            SELECT id FROM classes WHERE id = ? AND teacher_id = ?
        ''', (class_id, session['user_id'])).fetchone()

        if not class_check:
            conn.close()
            return jsonify({'success': False, 'error': 'Access denied'})

        # Save file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['CLASS_FILES_FOLDER'], unique_filename)
        file.save(file_path)

        # Insert into DB
        cursor.execute('''
            INSERT INTO class_files 
            (class_id, teacher_id, filename, original_filename, file_path, file_size, description)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (class_id, session['user_id'], unique_filename, filename,
              file_path, os.path.getsize(file_path), description))

        file_id = cursor.lastrowid

        # Notify all class members (✅ use SAME conn)
        members = conn.execute('''
            SELECT student_id FROM class_members 
            WHERE class_id = ? AND status = 'accepted'
        ''', (class_id,)).fetchall()

        class_name = conn.execute('SELECT name FROM classes WHERE id = ?', (class_id,)).fetchone()['name']

        for member in members:
            create_notification(
                member['student_id'],
                'New File Uploaded',
                f'A new file "{filename}" has been uploaded to class "{class_name}"',
                'file_upload',
                class_id,
                session['user_id'],
                conn=conn,      # ✅ same transaction
                cursor=cursor
            )

        conn.commit()
        conn.close()

        add_activity(session['user_id'], 'file_uploaded', f'Uploaded file to class: {filename}')

        return jsonify({'success': True, 'file_id': file_id})

    except Exception as e:
        print(f"Error uploading class file: {e}")
        if conn:
            conn.rollback()
            conn.close()
        return jsonify({'success': False, 'error': str(e)})



@app.route('/download_class_file/<int:file_id>')
def download_class_file(file_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        conn = get_db_connection()
        
        # Get file info
        file_info = conn.execute('''
            SELECT cf.*, c.name as class_name
            FROM class_files cf
            JOIN classes c ON cf.class_id = c.id
            WHERE cf.id = ?
        ''', (file_id,)).fetchone()
        
        if not file_info:
            return jsonify({'error': 'File not found'}), 404
        
        # Check if user has access (teacher or class member)
        if session.get('role') == 'teacher' and file_info['teacher_id'] == session['user_id']:
            # Teacher owns the file
            pass
        else:
            # Check if student is member of class
            member_check = conn.execute('''
                SELECT id FROM class_members 
                WHERE class_id = ? AND student_id = ? AND status = 'accepted'
            ''', (file_info['class_id'], session['user_id'])).fetchone()
            
            if not member_check:
                return jsonify({'error': 'Access denied'}), 403
        
        conn.close()
        
        # Send file
        return send_file(
            file_info['file_path'], 
            as_attachment=True, 
            download_name=file_info['original_filename']
        )
        
    except Exception as e:
        print(f"Error downloading class file: {e}")
        return jsonify({'error': 'File not found'}), 404

# STUDENT NOTIFICATION AND CLASS MANAGEMENT ROUTES

@app.route('/get_notifications')
def get_notifications():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        conn = get_db_connection()
        notifications = conn.execute('''
            SELECT n.*, u.username as sender_name, c.name as class_name
            FROM notifications n
            LEFT JOIN users u ON n.sender_id = u.id
            LEFT JOIN classes c ON n.class_id = c.id
            WHERE n.user_id = ?
            ORDER BY n.created_at DESC
            LIMIT 50
        ''', (session['user_id'],)).fetchall()
        
        notifications_list = []
        for notification in notifications:
            notifications_list.append({
                'id': notification['id'],
                'title': notification['title'],
                'message': notification['message'],
                'type': notification['type'],
                'class_id': notification['class_id'],
                'class_name': notification['class_name'],
                'sender_name': notification['sender_name'],
                'is_read': notification['is_read'],
                'created_at': notification['created_at']
            })
        
        conn.close()
        return jsonify({'success': True, 'notifications': notifications_list})
        
    except Exception as e:
        print(f"Error loading notifications: {e}")
        return jsonify({'success': False, 'error': str(e)})

# @app.route('/respond_to_class_invite', methods=['POST'])
# def respond_to_class_invite():
#     if 'user_id' not in session:
#         return jsonify({'error': 'Not logged in'}), 401
    
#     try:
#         data = request.get_json()
#         class_id = data.get('class_id')
#         response = data.get('response')  # 'accept' or 'decline'
#         notification_id = data.get('notification_id')
        
#         conn = get_db_connection()
#         cursor = conn.cursor()
        
#         # Update class membership status
#         if response == 'accept':
#             cursor.execute('''
#                 UPDATE class_members 
#                 SET status = 'accepted', joined_at = CURRENT_TIMESTAMP
#                 WHERE class_id = ? AND student_id = ?
#             ''', (class_id, session['user_id']))
            
#             # Get class info for notification
#             class_info = conn.execute('SELECT name, teacher_id FROM classes WHERE id = ?', (class_id,)).fetchone()
            
#             # Notify teacher
#             create_notification(
#                 class_info['teacher_id'],
#                 'Student Joined Class',
#                 f'{session["username"]} has joined your class "{class_info["name"]}"',
#                 'class_join',
#                 class_id,
#                 session['user_id']
#             )
            
#             message = 'Successfully joined the class!'
            
#         else:  # decline
#             cursor.execute('''
#                 DELETE FROM class_members 
#                 WHERE class_id = ? AND student_id = ?
#             ''', (class_id, session['user_id']))
            
#             message = 'Class invitation declined.'
        
#         # Mark notification as read
#         cursor.execute('''
#             UPDATE notifications 
#             SET is_read = TRUE 
#             WHERE id = ? AND user_id = ?
#         ''', (notification_id, session['user_id']))
        
#         conn.commit()
#         conn.close()
        
#         add_activity(session['user_id'], 'class_invite_response', f'{response.title()} class invitation')
        
#         return jsonify({'success': True, 'message': message})
        
#     except Exception as e:
#         print(f"Error responding to class invite: {e}")
#         return jsonify({'success': False, 'error': str(e)})


@app.route('/respond_to_class_invite', methods=['POST'])
def respond_to_class_invite():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    conn = None
    try:
        data = request.get_json()
        class_id = data.get('class_id')
        response = data.get('response')  # 'accept' or 'decline'
        notification_id = data.get('notification_id')

        conn = get_db_connection()
        cursor = conn.cursor()

        if response == 'accept':
            cursor.execute('''
                UPDATE class_members 
                SET status = 'accepted', joined_at = CURRENT_TIMESTAMP
                WHERE class_id = ? AND student_id = ?
            ''', (class_id, session['user_id']))

            # Get class info
            class_info = conn.execute('SELECT name, teacher_id FROM classes WHERE id = ?', (class_id,)).fetchone()

            # ✅ use SAME connection here
            create_notification(
                class_info['teacher_id'],
                'Student Joined Class',
                f'{session["username"]} has joined your class "{class_info["name"]}"',
                'class_join',
                class_id,
                session['user_id'],
                conn=conn,
                cursor=cursor
            )

            message = 'Successfully joined the class!'

        else:  # decline
            cursor.execute('''
                DELETE FROM class_members 
                WHERE class_id = ? AND student_id = ?
            ''', (class_id, session['user_id']))
            message = 'Class invitation declined.'

        # Mark notification as read
        cursor.execute('''
            UPDATE notifications 
            SET is_read = TRUE 
            WHERE id = ? AND user_id = ?
        ''', (notification_id, session['user_id']))

        conn.commit()
        conn.close()

        add_activity(session['user_id'], 'class_invite_response', f'{response.title()} class invitation')

        return jsonify({'success': True, 'message': message})

    except Exception as e:
        print(f"Error responding to class invite: {e}")
        if conn:
            conn.rollback()
            conn.close()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/get_student_classes')
def get_student_classes():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        conn = get_db_connection()
        
        # Get classes student is member of
        enrolled_classes = conn.execute('''
            SELECT c.*, u.username as teacher_name, cm.status, cm.joined_at
            FROM classes c
            JOIN users u ON c.teacher_id = u.id
            JOIN class_members cm ON c.id = cm.class_id
            WHERE cm.student_id = ? AND cm.status = 'accepted'
            ORDER BY cm.joined_at DESC
        ''', (session['user_id'],)).fetchall()
        
        enrolled_list = []
        for class_item in enrolled_classes:
            # Get file count for this class
            file_count = conn.execute('''
                SELECT COUNT(*) as count FROM class_files WHERE class_id = ?
            ''', (class_item['id'],)).fetchone()['count']
            
            enrolled_list.append({
                'id': class_item['id'],
                'name': class_item['name'],
                'description': class_item['description'],
                'teacher_name': class_item['teacher_name'],
                'joined_at': class_item['joined_at'],
                'file_count': file_count
            })
        
        conn.close()
        return jsonify({'success': True, 'classes': enrolled_list})
        
    except Exception as e:
        print(f"Error loading student classes: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_available_classes')
def get_available_classes():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        conn = get_db_connection()
        
        # Get public classes student is not a member of
        available_classes = conn.execute('''
            SELECT c.*, u.username as teacher_name
            FROM classes c
            JOIN users u ON c.teacher_id = u.id
            WHERE c.id NOT IN (
                SELECT class_id FROM class_members WHERE student_id = ?
            )
            ORDER BY c.created_at DESC
            LIMIT 20
        ''', (session['user_id'],)).fetchall()
        
        available_list = []
        for class_item in available_classes:
            # Get member count
            member_count = conn.execute('''
                SELECT COUNT(*) as count FROM class_members 
                WHERE class_id = ? AND status = 'accepted'
            ''', (class_item['id'],)).fetchone()['count']
            
            available_list.append({
                'id': class_item['id'],
                'name': class_item['name'],
                'description': class_item['description'],
                'teacher_name': class_item['teacher_name'],
                'member_count': member_count,
                'created_at': class_item['created_at']
            })
        
        conn.close()
        return jsonify({'success': True, 'classes': available_list})
        
    except Exception as e:
        print(f"Error loading available classes: {e}")
        return jsonify({'success': False, 'error': str(e)})

# @app.route('/request_class_join', methods=['POST'])
# def request_class_join():
#     if 'user_id' not in session:
#         return jsonify({'error': 'Not logged in'}), 401
    
#     try:
#         data = request.get_json()
#         class_id = data.get('class_id')
        
#         conn = get_db_connection()
#         cursor = conn.cursor()
        
#         # Get class info
#         class_info = conn.execute('''
#             SELECT name, teacher_id FROM classes WHERE id = ?
#         ''', (class_id,)).fetchone()
        
#         if not class_info:
#             return jsonify({'success': False, 'error': 'Class not found'})
        
#         # Check if already requested/member
#         existing = conn.execute('''
#             SELECT id FROM class_members WHERE class_id = ? AND student_id = ?
#         ''', (class_id, session['user_id'])).fetchone()
        
#         if existing:
#             return jsonify({'success': False, 'error': 'Already requested or member of this class'})
        
#         # Create join request
#         cursor.execute('''
#             INSERT INTO class_members (class_id, student_id, status)
#             VALUES (?, ?, 'pending')
#         ''', (class_id, session['user_id']))
        
#         # Notify teacher
#         create_notification(
#             class_info['teacher_id'],
#             'Class Join Request',
#             f'{session["username"]} has requested to join your class "{class_info["name"]}"',
#             'join_request',
#             class_id,
#             session['user_id']
#         )
        
#         conn.commit()
#         conn.close()
        
#         add_activity(session['user_id'], 'class_join_request', f'Requested to join class: {class_info["name"]}')
        
#         return jsonify({'success': True, 'message': 'Join request sent to teacher'})
        
#     except Exception as e:
#         print(f"Error requesting class join: {e}")
#         return jsonify({'success': False, 'error': str(e)})


@app.route('/request_class_join', methods=['POST'])
def request_class_join():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    try:
        data = request.get_json()
        class_id = data.get('class_id')

        conn = get_db_connection()
        cursor = conn.cursor()

        # Get class info
        class_info = conn.execute('''
            SELECT name, teacher_id FROM classes WHERE id = ?
        ''', (class_id,)).fetchone()

        if not class_info:
            return jsonify({'success': False, 'error': 'Class not found'})

        # Check if already requested/member
        existing = conn.execute('''
            SELECT id FROM class_members WHERE class_id = ? AND student_id = ?
        ''', (class_id, session['user_id'])).fetchone()

        if existing:
            return jsonify({'success': False, 'error': 'Already requested or member of this class'})

        # Create join request
        cursor.execute('''
            INSERT INTO class_members (class_id, student_id, status)
            VALUES (?, ?, 'pending')
        ''', (class_id, session['user_id']))

        # Notify teacher using SAME connection
        create_notification(
            class_info['teacher_id'],
            'Class Join Request',
            f'{session["username"]} has requested to join your class "{class_info["name"]}"',
            'join_request',
            class_id,
            session['user_id'],
            conn=conn,
            cursor=cursor
        )

        conn.commit()
        conn.close()

        add_activity(session['user_id'], 'class_join_request', f'Requested to join class: {class_info["name"]}')

        return jsonify({'success': True, 'message': 'Join request sent to teacher'})

    except Exception as e:
        print(f"Error requesting class join: {e}")
        if conn:
            conn.rollback()
            conn.close()
        return jsonify({'success': False, 'error': str(e)})



@app.route('/get_class_files/<int:class_id>')
def get_class_files(class_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        conn = get_db_connection()
        
        # Check if user has access to this class
        if session.get('role') == 'teacher':
            # Check if teacher owns this class
            access_check = conn.execute('''
                SELECT id FROM classes WHERE id = ? AND teacher_id = ?
            ''', (class_id, session['user_id'])).fetchone()
        else:
            # Check if student is member of this class
            access_check = conn.execute('''
                SELECT id FROM class_members 
                WHERE class_id = ? AND student_id = ? AND status = 'accepted'
            ''', (class_id, session['user_id'])).fetchone()
        
        if not access_check:
            return jsonify({'error': 'Access denied'}), 403
        
        # Get files
        files = conn.execute('''
            SELECT cf.*, c.name as class_name
            FROM class_files cf
            JOIN classes c ON cf.class_id = c.id
            WHERE cf.class_id = ?
            ORDER BY cf.uploaded_at DESC
        ''', (class_id,)).fetchall()
        
        files_list = []
        for file in files:
            files_list.append({
                'id': file['id'],
                'filename': file['original_filename'],
                'description': file['description'],
                'file_size': file['file_size'],
                'uploaded_at': file['uploaded_at'],
                'class_name': file['class_name']
            })
        
        conn.close()
        return jsonify({'success': True, 'files': files_list})
        
    except Exception as e:
        print(f"Error loading class files: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/mark_notification_read', methods=['POST'])
def mark_notification_read():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        data = request.get_json()
        notification_id = data.get('notification_id')
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE notifications 
            SET is_read = TRUE 
            WHERE id = ? AND user_id = ?
        ''', (notification_id, session['user_id']))
        
        conn.commit()
        conn.close()
        
        return jsonify({'success': True})
        
    except Exception as e:
        print(f"Error marking notification as read: {e}")
        return jsonify({'success': False, 'error': str(e)})

# MESSAGING SYSTEM

@app.route('/send_message', methods=['POST'])
def send_message():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        data = request.get_json()
        recipient_id = data.get('recipient_id')
        subject = data.get('subject', '')
        message = data.get('message')
        
        if not recipient_id or not message:
            return jsonify({'success': False, 'error': 'Recipient and message are required'})
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Send message
        cursor.execute('''
            INSERT INTO messages (sender_id, recipient_id, subject, message)
            VALUES (?, ?, ?, ?)
        ''', (session['user_id'], recipient_id, subject, message))
        
        # Create notification for recipient
        create_notification(
            recipient_id,
            'New Message',
            f'You have a new message from {session["username"]}',
            'message',
            None,
            session['user_id']
        )
        
        conn.commit()
        conn.close()
        
        add_activity(session['user_id'], 'message_sent', f'Sent message to user {recipient_id}')
        
        return jsonify({'success': True, 'message': 'Message sent successfully'})
        
    except Exception as e:
        print(f"Error sending message: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_messages')
def get_messages():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        conn = get_db_connection()
        messages = conn.execute('''
            SELECT m.*, 
                   sender.username as sender_name,
                   recipient.username as recipient_name
            FROM messages m
            JOIN users sender ON m.sender_id = sender.id
            JOIN users recipient ON m.recipient_id = recipient.id
            WHERE m.sender_id = ? OR m.recipient_id = ?
            ORDER BY m.sent_at DESC
            LIMIT 50
        ''', (session['user_id'], session['user_id'])).fetchall()
        
        messages_list = []
        for message in messages:
            messages_list.append({
                'id': message['id'],
                'sender_id': message['sender_id'],
                'recipient_id': message['recipient_id'],
                'sender_name': message['sender_name'],
                'recipient_name': message['recipient_name'],
                'subject': message['subject'],
                'message': message['message'],
                'is_read': message['is_read'],
                'sent_at': message['sent_at']
            })
        
        conn.close()
        return jsonify({'success': True, 'messages': messages_list})
        
    except Exception as e:
        print(f"Error loading messages: {e}")
        return jsonify({'success': False, 'error': str(e)})
#new video-chat function

@app.route('/video_chat_ai', methods=['POST'])
def video_chat_ai():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    data = request.get_json()
    question = data.get('question', '')

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    try:
        # Use your LLM (same as summarize_with_gemini)
        llm = LLM(model="gemini/gemini-2.0-flash", temperature=0.7)
        agent = Agent(
            role="Video AI Assistant",
            goal="Answer user questions about the video content or transcript",
            backstory="You are an AI that helps learners by discussing video lessons",
            llm=llm,
        )
        task = Task(description=f"Answer clearly: {question}", expected_output="Helpful answer", agent=agent)
        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        answer = crew.kickoff()

        add_activity(session['user_id'], 'video_chat', question)
        return jsonify({'success': True, 'answer': str(answer)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500



