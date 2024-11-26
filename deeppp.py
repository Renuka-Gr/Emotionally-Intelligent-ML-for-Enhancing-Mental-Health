import cv2
from deepface import DeepFace
import time

# Start the webcam
cap = cv2.VideoCapture(0)

# Start the timer
start_time = time.time()

# To track emotions and count them
emotion_count = {"angry": 0, "disgust": 0, "fear": 0, "happy": 0, "sad": 0, "surprise": 0, "neutral": 0}

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Analyze the frame for emotion
    try:
        # DeepFace analyze with 'emotion' action
        result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)

        # Get the dominant emotion
        dominant_emotion = result[0]["dominant_emotion"]

        # Update the count for the detected emotion
        if dominant_emotion in emotion_count:
            emotion_count[dominant_emotion] += 1

        # Display the current emotion on the frame
        cv2.putText(frame, f"Emotion: {dominant_emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    except Exception as e:
        print(f"Error: {e}")

    # Display the resulting frame
    cv2.imshow("Emotion Detection", frame)

    # Check if 2 minutes (120 seconds) have passed
    elapsed_time = time.time() - start_time
    if elapsed_time >= 120:
        # Once 2 minutes have passed, break the loop
        break

    # Continue capturing frames for 2 minutes
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After 2 minutes, analyze the dominant emotion
dominant_emotion = max(emotion_count, key=emotion_count.get)
cv2.destroyAllWindows()

# Display the result
print(f"After 2 minutes, the dominant emotion was: {dominant_emotion}")
