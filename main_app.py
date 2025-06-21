# cv_personality_prediction_complete.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import logging
from typing import Dict, List, Any, Optional, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NLP Libraries
try:
    import nltk
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available. Some features will be limited.")

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not available. Linguistic features will be limited.")

try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False
    logger.warning("textstat not available. Readability metrics will be skipped.")

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available. Using only Random Forest.")

# Document Processing
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    logger.warning("PyPDF2 not available. PDF processing limited.")

try:
    import docx2txt
    DOCX2TXT_AVAILABLE = True
except ImportError:
    DOCX2TXT_AVAILABLE = False
    logger.warning("docx2txt not available. DOCX processing disabled.")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber not available. Advanced PDF processing disabled.")

# Web Framework
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename

class DocumentProcessor:
    """Handles document processing for various file formats"""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.docx', '.txt']
        logger.info("DocumentProcessor initialized")
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF using available libraries"""
        text = ""
        
        if PDFPLUMBER_AVAILABLE:
            try:
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        text += page.extract_text() or ""
                if text.strip():
                    return text
            except Exception as e:
                logger.warning(f"pdfplumber failed: {e}")
        
        if PYPDF2_AVAILABLE:
            try:
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    text = "\n".join([page.extract_text() or "" for page in reader.pages])
                return text
            except Exception as e:
                logger.warning(f"PyPDF2 failed: {e}")
        
        return "PDF processing requires PyPDF2 or pdfplumber"

    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX files"""
        if DOCX2TXT_AVAILABLE:
            try:
                return docx2txt.process(file_path)
            except Exception as e:
                logger.error(f"DOCX extraction failed: {e}")
        return "DOCX processing requires docx2txt"

    def process_document(self, file_path: str) -> str:
        """Process any supported document format"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_ext == '.docx':
            return self.extract_text_from_docx(file_path)
        elif file_ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

class FeatureExtractor:
    """Extracts features from text for personality prediction"""
    
    def __init__(self):
        self.nlp = None
        self.stop_words = set()
        self._initialize_nlp()
        self.tfidf = TfidfVectorizer(max_features=100, stop_words='english')
    
    def _initialize_nlp(self):
        """Initialize NLP components"""
        if NLTK_AVAILABLE:
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                self.stop_words = set(stopwords.words('english'))
            except:
                self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at'}
        
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")

    def extract_basic_features(self, text: str) -> Dict[str, float]:
        """Extract basic text statistics"""
        if not text:
            return {}
            
        words = re.findall(r'\w+', text.lower())
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        features = {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'avg_sentence_length': len(words)/max(len(sentences), 1),
            'unique_word_ratio': len(set(words))/max(len(words), 1),
            'stopword_ratio': sum(1 for w in words if w in self.stop_words)/max(len(words), 1)
        }
        
        if TEXTSTAT_AVAILABLE:
            try:
                features.update({
                    'flesch_reading_ease': textstat.flesch_reading_ease(text),
                    'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
                    'smog_index': textstat.smog_index(text)
                })
            except:
                pass
                
        return features
    
    def extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """Extract linguistic features using spaCy"""
        features = {}
        
        if not self.nlp or not text.strip():
            return features
            
        try:
            doc = self.nlp(text)
            pos_counts = {}
            for token in doc:
                pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
            
            total = len(doc)
            for pos in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON']:
                features[f'{pos.lower()}_ratio'] = pos_counts.get(pos, 0) / total
            
            # Named entities
            ner_counts = {}
            for ent in doc.ents:
                ner_counts[ent.label_] = ner_counts.get(ent.label_, 0) + 1
            
            for label in ['PERSON', 'ORG', 'GPE', 'DATE']:
                features[f'{label.lower()}_count'] = ner_counts.get(label, 0)
                
        except Exception as e:
            logger.error(f"Linguistic feature extraction failed: {e}")
            
        return features
    
    def extract_personality_indicators(self, text: str) -> Dict[str, float]:
        """Extract indicators of personality traits"""
        indicators = {
            'extroversion': ['team', 'collaborate', 'lead', 'present', 'social'],
            'conscientiousness': ['organized', 'detail', 'plan', 'schedule', 'efficient'],
            'openness': ['creative', 'innovative', 'research', 'explore', 'learn'],
            'agreeableness': ['help', 'support', 'assist', 'cooperate', 'friendly'],
            'neuroticism': ['stress', 'pressure', 'difficult', 'challenge', 'worry']
        }
        
        text_lower = text.lower()
        features = {}
        
        for trait, words in indicators.items():
            count = sum(text_lower.count(word) for word in words)
            features[f'{trait}_score'] = count
            features[f'{trait}_ratio'] = count / max(len(text_lower.split()), 1)
            
        return features
    
    def extract_all_features(self, text: str) -> Dict[str, float]:
        """Combine all feature extraction methods"""
        features = {}
        features.update(self.extract_basic_features(text))
        features.update(self.extract_linguistic_features(text))
        features.update(self.extract_personality_indicators(text))
        return features

class PersonalityModel:
    """Machine learning model for personality prediction"""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess the synthetic CV data"""
        # Clean experience years
        df['experience_years'] = df.apply(
            lambda x: min(x['experience_years'], x['age'] - 18), 
            axis=1
        )
        
        # Process skills into binary features
        all_skills = set()
        for skills in df['skills'].str.split(','):
            all_skills.update([s.strip().lower() for s in skills])
        
        for skill in all_skills:
            df[f'skill_{skill}'] = df['skills'].str.lower().str.contains(skill).astype(int)
        
        # One-hot encode education
        df = pd.get_dummies(df, columns=['education'], prefix='edu')
        
        # Prepare features and target
        X = df.drop(columns=['personality', 'name', 'skills'])
        y = df['personality']
        
        return X, y
    
    def train(self, df: pd.DataFrame) -> float:
        """Train the model on the synthetic data"""
        try:
            X, y = self.preprocess_data(df)
            
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, test_size=0.2, random_state=42
            )
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            self.is_trained = True
            
            logger.info(f"Model trained with accuracy: {accuracy:.2f}")
            return accuracy
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return 0.0
    
    def predict_from_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict personality from extracted features"""
        if not self.is_trained:
            return {'error': 'Model not trained'}
        
        try:
            # Convert features to dataframe
            features_df = pd.DataFrame([features])
            
            # Scale features
            features_scaled = self.scaler.transform(features_df)
            
            # Predict
            pred = self.model.predict(features_scaled)[0]
            proba = self.model.predict_proba(features_scaled)[0]
            
            # Decode prediction
            personality = self.label_encoder.inverse_transform([pred])[0]
            
            return {
                'personality': personality,
                'confidence': max(proba),
                'probabilities': {
                    cls: float(prob) 
                    for cls, prob in zip(self.label_encoder.classes_, proba)
                }
            }
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {'error': str(e)}

class PersonalitySystem:
    """Main system integrating all components"""
    
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.feature_extractor = FeatureExtractor()
        self.model = PersonalityModel()
        logger.info("PersonalitySystem initialized")
    
    def train_from_csv(self, csv_path: str) -> float:
        """Train the system from a CSV file"""
        try:
            df = pd.read_csv(csv_path)
            # Combine all relevant text fields into one string per row
            df['text'] = df.apply(lambda row: f"{row['education']} {row['skills']} {row['experience_years']}", axis=1)
            # Extract features from text
            features = [self.feature_extractor.extract_all_features(text) for text in df['text']]
            X = pd.DataFrame(features).fillna(0)
            y = df['personality']
            
            # Encode labels
            y_encoded = self.model.label_encoder.fit_transform(y)
            
            # Scale features
            X_scaled = self.model.scaler.fit_transform(X)
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, test_size=0.2, random_state=42
            )
            
            # Train model
            self.model.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            self.model.is_trained = True
            
            logger.info(f"Model trained with accuracy: {accuracy:.2f}")
            return accuracy
            
        except Exception as e:
            logger.error(f"Training from CSV failed: {e}")
            return 0.0
    
    def process_cv(self, file_path: str) -> Dict[str, Any]:
        """Process a CV file and predict personality"""
        try:
            # Step 1: Extract text
            text = self.doc_processor.process_document(file_path)
            if not text.strip():
                return {'error': 'No text extracted from document'}
            
            # Step 2: Extract features
            features = self.feature_extractor.extract_all_features(text)
            
            # Step 3: Predict personality
            prediction = self.model.predict_from_features(features)
            
            return {
                'text': text[:1000] + '...' if len(text) > 1000 else text,
                'features': features,
                'prediction': prediction
            }
        except Exception as e:
            logger.error(f"CV processing failed: {e}")
            return {'error': str(e)}

# Flask Application Setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize system
system = PersonalitySystem()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result = system.process_cv(filepath)
        # os.remove(filepath)  # File is now kept in uploads folder for record-keeping
        
        return jsonify(result)

@app.route('/train', methods=['POST'])
def train():
    if 'file' not in request.files:
        return jsonify({'error': 'No training file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        accuracy = system.train_from_csv(filepath)
        os.remove(filepath)
        
        return jsonify({
            'status': 'success',
            'accuracy': accuracy
        })
    
    return jsonify({'error': 'Invalid file format. Please upload a CSV file'})

if __name__ == '__main__':
    # Train with synthetic data if available
    if os.path.exists('synthetic_cv_data.csv'):
        system.train_from_csv('synthetic_cv_data.csv')
    
    app.run(debug=True)
