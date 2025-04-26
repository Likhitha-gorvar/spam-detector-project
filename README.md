# Spam Detector Project

This project uses machine learning to classify SMS messages as spam or ham (not spam).

## Overview

The **Spam Text Classifier** is developed to identify and classify messages as either spam or ham (legitimate).

- **File:** `spam.csv`
- **Columns:**
  - `v1`: label (spam/ham)
  - `v2`: message text

### Features:
- Preprocessing with NLTK
- TF-IDF vectorization
- Naive Bayes classification
- Evaluation using precision, recall, F1-score

---

## 🛠️ How to Run
1. **Set up a virtual environment (optional but recommended)**:
   - For `virtualenv`:
     ```bash
     python -m venv venv
     source venv/bin/activate  # On macOS/Linux
     .\venv\Scripts\activate   # On Windows
     ```
   - For `conda`:
     ```bash
     conda create --name myenv python=3.x
     conda activate myenv
     ```
2. **Install dependencies**:
   - Install the required libraries using `pip`:
     ```bash
     pip install -r requirements.txt
     ```

3. **Run the Streamlit app**:
   - After installing dependencies, run the Streamlit app:
     ```bash
     streamlit run spam_detector_app.py
     ```

   - You should be able to access the app via:
     - **Local URL:** [http://localhost:8502](http://localhost:8502)
     - **Network URL:** [http://192.168.1.108:8502](http://192.168.1.108:8502) (for access on other devices in the same network)

4. **Running Tests**:
   - If you have tests set up, run them using:
     ```bash
     pytest  # Or any other test command you use
     ```

---
