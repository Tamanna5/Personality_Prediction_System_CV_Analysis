# CV Personality Prediction System

A sophisticated web application that leverages advanced Natural Language Processing (NLP) and Machine Learning techniques to predict personality traits from CV/resume documents. The system features a modern, responsive UI and comprehensive document analysis capabilities.

## 🚀 Features

### **Advanced Document Processing**
- **Multi-format Support**: Handles PDF, DOCX, and TXT files with robust text extraction
- **Intelligent Text Processing**: Uses multiple libraries (PyPDF2, pdfplumber, docx2txt) for optimal text extraction
- **File Retention**: All uploaded files are securely stored in the `uploads/` folder for record-keeping
- **Large File Support**: Accepts files up to 16MB

### **Comprehensive NLP Analysis**
- **Feature Extraction**: Analyzes 50+ linguistic and semantic features including:
  - Text complexity metrics (readability scores, sentence structure)
  - Vocabulary analysis (word diversity, technical terms)
  - Writing style indicators (formality, tone, structure)
  - Semantic features (keyword analysis, topic modeling)
- **Advanced NLP Libraries**: Integrates NLTK, spaCy, and textstat for comprehensive analysis
- **Multi-language Support**: Built-in support for various languages and writing styles

### **Machine Learning Capabilities**
- **Ensemble Models**: Uses Random Forest and XGBoost for robust predictions
- **Feature Engineering**: Automatic extraction and scaling of 50+ features
- **Model Training**: Live model retraining capability with custom datasets
- **Performance Metrics**: Real-time accuracy tracking and model evaluation

### **Modern Web Interface**
- **Responsive Design**: Beautiful gradient-based UI with card layout
- **Real-time Processing**: Instant results display without page reload
- **Interactive Elements**: Hover effects, loading animations, and smooth transitions
- **User-friendly**: Intuitive drag-and-drop file upload interface

### **Developer Features**
- **API Endpoints**: RESTful API for integration with other systems
- **Training Interface**: `/train` endpoint for model retraining with custom data
- **Comprehensive Logging**: Detailed logging for debugging and monitoring
- **Error Handling**: Robust error handling with user-friendly messages

## 🛠️ Technology Stack

### **Backend**
- **Flask**: Lightweight web framework
- **scikit-learn**: Machine learning algorithms
- **XGBoost**: Gradient boosting for enhanced predictions
- **NLTK & spaCy**: Natural Language Processing
- **Pandas & NumPy**: Data manipulation and numerical computing

### **Document Processing**
- **PyPDF2 & pdfplumber**: PDF text extraction
- **python-docx & docx2txt**: DOCX file processing
- **textstat**: Readability and text complexity analysis

### **Frontend**
- **HTML5 & CSS3**: Modern, responsive design
- **JavaScript**: Dynamic interactions and AJAX requests
- **Gradient Design**: Beautiful visual aesthetics

## 📋 Prerequisites

- Python 3.7 or higher
- pip package manager
- Sufficient disk space for uploaded files

## 🚀 Installation & Setup

### 1. **Clone and Navigate**
```bash
cd Personality_Prediction_System_CV_Analysis
```

### 2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 3. **Download NLTK Data** (First time setup)
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
```

### 4. **Download spaCy Model** (Optional but recommended)
```bash
python -m spacy download en_core_web_sm
```

### 5. **Run the Application**
```bash
python main_app.py
```

### 6. **Access the Application**
Open your browser and navigate to: [http://localhost:5000](http://localhost:5000)

## 📖 Usage Guide

### **For End Users**

1. **Upload a CV/Resume**
   - Click the upload area or drag-and-drop your file
   - Supported formats: PDF, DOCX, TXT (max 16MB)
   - The system will automatically process your document

2. **View Results**
   - Personality prediction with confidence score
   - Detailed feature breakdown
   - Extracted text preview
   - Analysis insights

3. **File Management**
   - All uploaded files are stored in the `uploads/` folder
   - Files are retained for future reference
   - No automatic deletion

### **For Developers**

#### **API Endpoints**

**Upload and Analyze CV:**
```bash
POST /upload
Content-Type: multipart/form-data
Body: file (PDF/DOCX/TXT)
```

**Train Model with Custom Data:**
```bash
POST /train
Content-Type: multipart/form-data
Body: file (CSV format)
```

#### **Model Training**
- Upload a CSV file with columns: `text`, `personality_type`
- The system will automatically retrain the model
- Training accuracy is returned as response

## 🔧 Configuration

### **File Upload Settings**
- Maximum file size: 16MB
- Supported formats: PDF, DOCX, TXT
- Upload directory: `uploads/` (auto-created)

### **Model Configuration**
- Default training data: `synthetic_cv_data.csv`
- Feature extraction: 50+ linguistic features
- Algorithm: Random Forest + XGBoost ensemble

### **UI Customization**
- Modify `templates/index.html` for UI changes
- CSS styles are embedded for easy customization
- Responsive design for mobile compatibility

## 📊 Features Analyzed

The system extracts and analyzes the following features from CV documents:

### **Text Complexity**
- Readability scores (Flesch, Gunning Fog, SMOG)
- Sentence length and structure
- Word complexity and vocabulary diversity
- Paragraph organization

### **Writing Style**
- Formality level and tone
- Technical vs. general vocabulary
- Sentence structure patterns
- Punctuation and formatting

### **Content Analysis**
- Keyword frequency and importance
- Skill mentions and technical terms
- Experience indicators
- Education and certification references

### **Semantic Features**
- Topic modeling and categorization
- Sentiment analysis
- Professional terminology
- Industry-specific language

## 🎯 Personality Types Predicted

The system can predict various personality types based on the CV content analysis, including but not limited to:

- **Analytical**: Detail-oriented, data-driven, systematic
- **Creative**: Innovative, artistic, unconventional
- **Social**: People-oriented, communicative, collaborative
- **Practical**: Results-focused, efficient, hands-on
- **Leadership**: Strategic, decision-making, visionary

## 🔍 Troubleshooting

### **Common Issues**

1. **NLTK Data Missing**
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

2. **spaCy Model Not Found**
   ```bash
   python -m spacy download en_core_web_sm
   ```

3. **PDF Processing Issues**
   - Ensure PDF is not password-protected
   - Check if PDF contains extractable text
   - Try alternative PDF processing libraries

4. **Model Training Failures**
   - Verify CSV format: `text,personality_type` columns
   - Ensure sufficient training data (minimum 50 samples)
   - Check for data quality issues

### **Performance Optimization**

- **Large Files**: Consider splitting large documents
- **Batch Processing**: Use API endpoints for multiple files
- **Model Updates**: Retrain periodically with new data

## 🤝 Contributing

We welcome contributions! Please feel free to:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

### **Development Setup**
```bash
git clone <repository-url>
cd Personality_Prediction_System_CV_Analysis
pip install -r requirements.txt
python main_app.py
```

## 📝 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

**Tamanna Kalariya**

## 🙏 Acknowledgments

- **NLTK Team**: For comprehensive NLP tools
- **spaCy**: For advanced language processing
- **scikit-learn**: For machine learning algorithms
- **Flask**: For the web framework

## 📞 Support

For questions, issues, or feature requests:
- Create an issue on GitHub
- Contact the author directly
- Check the troubleshooting section above

---

**Note**: This system is designed for educational and research purposes. Personality predictions should not be used as the sole basis for hiring decisions. 