# CV Personality Prediction System

A modern web application that predicts personality traits from CV/resume documents using NLP and machine learning. The app features a unique, visually appealing UI and stores all uploaded files for record-keeping.

## Features

- **Modern, Unique UI:**
  - Gradient header, card layout, custom upload area, and stylish result display.
  - Responsive and user-friendly design.
- **File Upload & Retention:**
  - Upload CVs in PDF, DOCX, or TXT format (max 16MB).
  - Uploaded files are saved in the `uploads/` folder and are not deleted after processing.
- **AI-Powered Personality Prediction:**
  - Extracts NLP features from CV text and predicts personality type.
  - Shows prediction confidence and feature breakdown.
- **Live Results:**
  - Results are displayed instantly below the upload form without reloading the page.
- **Training Endpoint:**
  - Upload a new CSV to `/train` to retrain the model with your own data.

## Setup Instructions

1. **Clone the repository and install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure you have the following files/folders:**
   - `main_app.py`
   - `synthetic_cv_data.csv` (for initial training)
   - `templates/index.html` (UI template)
   - `uploads/` (will be created automatically if missing)

3. **Run the application:**

   ```bash
   python main_app.py
   ```

4. **Open your browser and go to:**
   - [http://localhost:5000](http://localhost:5000)

## Usage

- **Upload a CV:**
  - Use the web interface to upload a PDF, DOCX, or TXT file.
  - The app will analyze the document and display the predicted personality, confidence, and feature details.
  - The uploaded file will be saved in the `uploads/` folder for your records.

- **Retrain the Model:**
  - Use the `/train` endpoint (via a tool like Postman or curl) to upload a new CSV and retrain the model.

## Customization

- The UI can be customized in `templates/index.html`.
- Uploaded files are stored in `uploads/` and are not deleted automatically.
- The model can be retrained with your own data using the `/train` endpoint.

## Credits

- **Author:** Tamanna Kalariya
- **UI & Backend:** Modern Flask, HTML, CSS, and Python

---

For questions or improvements, feel free to reach out or submit a pull request! 