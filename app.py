from flask import Flask, render_template, request, jsonify
from deepface import DeepFace
import base64
import cv2
import numpy as np
import threading

app = Flask(__name__)
app.jinja_env.globals.update(enumerate=enumerate)

# Questions and answer map
questions = [
    "How often have you felt down, depressed, or hopeless?",
    "How often have you experienced a lack of interest or pleasure in doing things?",
    "How frequently have you had trouble sleeping (too much or too little)?",
    "How often do you feel tired or have low energy?",
    "How often have you felt bad about yourself, or that you are a failure?",
    "How often have you had difficulty concentrating on tasks, such as reading or watching TV?",
    "How often have you felt anxious, worried, or on edge?",
    "How often do you feel that life isn’t worth living?",
    "How frequently do you feel irritable or easily frustrated?",
    "How often do you feel disconnected from people around you?"
]

answer_map = {
    "Never": 0,
    "Sometimes": 1,
    "Often": 2,
    "Almost always": 3
}

dominant_emotion = ""

# Function to analyze emotion from base64 image data
def analyze_emotion_from_base64(image_data):
    global dominant_emotion
    img_data = base64.b64decode(image_data.split(',')[1])
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    try:
        result = DeepFace.analyze(img, actions=["emotion"], enforce_detection=False)
        dominant_emotion = result[0]["dominant_emotion"]
    except Exception as e:
        print(f"Error analyzing emotion: {e}")

# Route to handle emotion detection from the client-side
@app.route("/detect_emotion", methods=["POST"])
def detect_emotion():
    data = request.get_json()
    image_data = data['image']  # Base64 encoded image from the client
    analyze_emotion_from_base64(image_data)
    return jsonify({"dominant_emotion": dominant_emotion})

# Route to render the questionnaire form
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        scores = [int(request.form.get(f"q{i}")) for i in range(len(questions))]
        total_score = sum(scores)

        # Determine depression level
        if 0 <= total_score <= 5:
            result = "No depression"
        elif 6 <= total_score <= 10:
            result = "Moderate depression"
        elif 11 <= total_score <= 20:
            result = "More than moderate depression"
        elif 21 <= total_score <= 30:
            result = "High depression"
        else:
            result = "Invalid score"

        # Emotion-based judgment
        # Emotion-based judgment
        if dominant_emotion in ["happy", "angry", "surprise", "fear", "sad"]:
            seriousness = "You are not serious in the examination."
        elif dominant_emotion == "neutral":
            seriousness = "You gave the exam seriously."
        else:
            seriousness = "Emotion analysis was inconclusive. Please retake the exam."


        return render_template("result.html", total_score=total_score, result=result, seriousness=seriousness)
    
    return render_template("index.html", questions=questions, answer_map=answer_map)

# Start the app
if __name__ == "__main__":
    app.run(debug=True)
