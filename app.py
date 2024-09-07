import streamlit as st
import pickle
import re
import nltk
from PIL import Image
import plotly.graph_objs as go
from fpdf import FPDF
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import textstat
import sqlite3

nltk.download('punkt')
nltk.download('stopwords')

# Load models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidfd = pickle.load(open('tfidf.pkl', 'rb'))

# Database functions
def create_user_table():
    conn = sqlite3.connect('user.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY, email TEXT UNIQUE, password TEXT)''')
    conn.commit()
    conn.close()

def create_feedback_table():
    conn = sqlite3.connect('feedback.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS feedback
                 (id INTEGER PRIMARY KEY, user_id INTEGER, email TEXT, feedback TEXT, FOREIGN KEY (user_id) REFERENCES users(id))''')
    conn.commit()
    conn.close()

def update_feedback_table():
    conn = sqlite3.connect('feedback.db')
    c = conn.cursor()
    
    # Check if table exists and has the right schema
    c.execute("PRAGMA table_info(feedback)")
    columns = [column[1] for column in c.fetchall()]

    if 'user_id' not in columns:
        # Add user_id column if it doesn't exist
        c.execute("ALTER TABLE feedback ADD COLUMN user_id INTEGER")
    
    if 'email' not in columns:
        # Add email column if it doesn't exist
        c.execute("ALTER TABLE feedback ADD COLUMN email TEXT")

    conn.commit()
    conn.close()

def save_user(email, password):
    conn = sqlite3.connect('user.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (email, password) VALUES (?, ?)", (email, hash_password(password)))
        conn.commit()
    except sqlite3.IntegrityError:
        st.error("User with this email already exists.")
    conn.close()

def authenticate_user(email, password):
    conn = sqlite3.connect('user.db')
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE email = ? AND password = ?", (email, hash_password(password)))
    user_id = c.fetchone()
    conn.close()
    return user_id[0] if user_id else None

def save_feedback_to_db(user_id, email, feedback):
    conn = sqlite3.connect('feedback.db')
    c = conn.cursor()
    c.execute("INSERT INTO feedback (user_id, email, feedback) VALUES (?, ?, ?)", (user_id, email, feedback))
    conn.commit()
    conn.close()


# Text cleaning function
def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text

# Analyzers
def analyze_readability(text):
    return textstat.flesch_reading_ease(text)

def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)
    return score['compound']

def plot_sentiment(sentiment):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['Positive', 'Neutral', 'Negative'],
        y=[max(sentiment, 0), 1 - abs(sentiment), max(-sentiment, 0)],
        marker_color=['green', 'gray', 'red']
    ))
    fig.update_layout(title_text='Sentiment Analysis', xaxis_title='Sentiment', yaxis_title='Score')
    st.plotly_chart(fig)

def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    wordcloud.to_file('wordcloud.png')

def create_pdf_report(category_name, sentiment, word_count, sentence_count, readability_score):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.cell(200, 10, txt="Resume Analysis Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Predicted Category: {category_name}", ln=True)
    pdf.cell(200, 10, txt=f"Sentiment Score: {sentiment:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Word Count: {word_count}", ln=True)
    pdf.cell(200, 10, txt=f"Sentence Count: {sentence_count}", ln=True)
    pdf.cell(200, 10, txt=f"Readability Score: {readability_score:.2f}", ln=True)
    pdf.output("resume_report.pdf")


def main():
    # Create the feedback table if it doesn't exist
    create_feedback_table()

    st.markdown("""
        <div style="background-color:#4CAF50;padding:10px;border-radius:10px;">
        <h1 style="text-align:center;color:white;">Resume Category App</h1>
        </div>
    """, unsafe_allow_html=True)

    st.sidebar.title("Menu")
    choice = st.sidebar.radio("Select an option", ["Home", "Login", "Register", "Feedback"])

    # Check login status for access control
    if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
        logged_in = False
    else:
        logged_in = True

    # Common title for all sections
    st.title("Resume Category App")

    if choice == "Home":
        if not logged_in:
            st.markdown("""
    <p style='color: #FF0000; font-weight: bold; font-size: 16px;'>
    Please log in first to categorize your resume.
    </p>
""", unsafe_allow_html=True)
        else:
            st.markdown("<p style='text-align: center; font-size: 18px;'>Upload your resume, and we'll categorize it based on job roles. Quick, efficient, and smart!</p>", unsafe_allow_html=True)

            image = Image.open('resume_image.png')  # Ensure you have an image file in the same directory
            st.image(image, use_column_width=True)

            st.markdown("<h3 style='text-align: center;'>Upload Your Resume</h3>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader('', type=['txt', 'pdf'])

            if uploaded_file is not None:
                try:
                    resume_bytes = uploaded_file.read()
                    resume_text = resume_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    resume_text = resume_bytes.decode('latin-1')

                st.progress(50)

                cleaned_resume = clean_resume(resume_text)
                input_features = tfidfd.transform([cleaned_resume])

                prediction_id = clf.predict(input_features)[0]

                category_mapping = {
                    15: "Java Developer",
                    23: "Testing",
                    8: "DevOps Engineer",
                    20: "Python Developer",
                    24: "Web Designing",
                    12: "HR",
                    13: "Hadoop",
                    3: "Blockchain",
                    10: "ETL Developer",
                    18: "Operations Manager",
                    6: "Data Science",
                    22: "Sales",
                    16: "Mechanical Engineer",
                    1: "Arts",
                    7: "Database",
                    11: "Electrical Engineering",
                    14: "Health and fitness",
                    19: "PMO",
                    4: "Business Analyst",
                    9: "DotNet Developer",
                    2: "Automation Testing",
                    17: "Network Security Engineer",
                    21: "SAP Developer",
                    5: "Civil Engineer",
                    0: "Advocate",
                }

                category_name = category_mapping.get(prediction_id, "Unknown")

                word_count = len(cleaned_resume.split())
                sentence_count = len(re.split(r'[.!?]', cleaned_resume))
                readability_score = analyze_readability(cleaned_resume)
                sentiment = analyze_sentiment(cleaned_resume)

                st.markdown("<br>", unsafe_allow_html=True)

                st.markdown(f"""
                    <div style="border-radius: 10px; padding: 20px; background-color: #E3F2FD; color: #1A237E;">
                        <h3 style='text-align: center;'>Predicted Category:</h3>
                        <p style='text-align: center; font-size: 22px;'>{category_name}</p>
                    </div>
                """, unsafe_allow_html=True)

                st.write(f"Word Count: {word_count}")
                st.write(f"Sentence Count: {sentence_count}")
                st.write(f"Readability Score: {readability_score:.2f}")

                st.write(f"Cover Letter Sentiment: {sentiment}")

                plot_sentiment(sentiment)

                generate_wordcloud(cleaned_resume)
                st.image('wordcloud.png', caption='Word Cloud of Resume Text')

                create_pdf_report(category_name, sentiment, word_count, sentence_count, readability_score)
                with open("resume_report.pdf", "rb") as pdf_file:
                    st.download_button("Download Analysis Report", pdf_file, "resume_report.pdf")

    elif choice == "Login":
        st.title("Login")
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type='password')

        if st.button("Login"):
            # Add authentication logic here (e.g., check credentials)
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.session_state['email'] = email
            st.success("You have logged in successfully!")

    elif choice == "Register":
        st.title("Register")
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type='password')

        if st.button("Register"):
            # Add registration logic here (e.g., save to a database)
            st.success("Registration successful!")

    elif choice == "Feedback":
        if not logged_in:
            st.markdown("""
    <p style='color: #FF0000; font-weight: bold; font-size: 16px;'>
    Please log in first to give the feedback.
    </p>
""", unsafe_allow_html=True)
        else:
            st.title("Feedback")
            feedback = st.text_area("Please leave your feedback:")
            if st.button("Submit Feedback"):
                if feedback:
                    save_feedback_to_db(st.session_state['username'], st.session_state['email'], feedback)
                    # save_feedback_to_db(st.session_state['username'], feedback)
                    st.write("Thank you for your feedback!")

    st.markdown("""
    <br><hr>
    <p style='text-align: center; font-size: 18px; font-weight: bold;'>
    Designed by Using  Machine Learning & Natural Language Processing | Developed by Matrika Dhamala
    </p>
""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()


