#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
from keras.models import model_from_json
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
import pygame
import time
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Initialize Pygame for music playback
pygame.init()

# Load your music files
happy_music = "happy.mp3"
animal_music = "an.mp3"

# Load model
json_file = open("facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("facialemotionmodel.h5")

# Initialize Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Define feature extraction
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape((1, 48, 48, 1))
    return feature / 255.0

# Initialize Sentiment Intensity Analyzer
sia = SentimentIntensityAnalyzer()

# Initialize webcam
webcam = cv2.VideoCapture(0)

# Emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Initialize music and emotion counts
current_music = None
emotion_counts = {'happy': 0, 'sad': 0, 'neutral': 0}

# Initialize lists to store true labels and predicted scores
true_labels = []
predicted_scores = []

# Hysteresis parameters
neutral_hysteresis_duration = 5  # in seconds
neutral_music_playing = False
neutral_hysteresis_timer = time.time()
music_duration = 30  # in seconds
music_start_time = time.time()  # Initialize music_start_time

while True:
    i, im = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(im, 1.3, 5)
    
    try:
        # Reset emotion counts for each frame
        emotion_counts = {'happy': 0, 'sad': 0, 'neutral': 0}
        face_detected = False  # Assume no face is detected initially
        
        for (p, q, r, s) in faces:
            face_detected = True
            image = gray[q:q + s, p:p + r]
            cv2.rectangle(im, (p, q), (p + r, q + s), (255, 0, 0), 2)
            image = cv2.resize(image, (48, 48))
            img = extract_features(image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            
            # Get sentiment scores for demonstration purposes
            text_associated_with_image = "I'm feeling happy today!"
            sentiment_scores = sia.polarity_scores(text_associated_with_image)
            sentiment_label = 'Neutral'
            if sentiment_scores['compound'] >= 0.05:
                sentiment_label = 'Positive'
            elif sentiment_scores['compound'] <= -0.05:
                sentiment_label = 'Negative'
            
            # Update emotion counts
            if prediction_label in emotion_counts:
                emotion_counts[prediction_label] += 1
                cv2.putText(im, f"{prediction_label}", (p - 10, q - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
        
        total_emotion_count = sum(emotion_counts.values())
        average_neutral_score = emotion_counts['neutral'] / total_emotion_count if total_emotion_count > 0 else 0
        
        # Decide music based on emotion detection
        if emotion_counts['sad'] > 0 and emotion_counts['neutral'] == 0:
            if current_music != 'sad':
                pygame.mixer.music.load(happy_music)
                pygame.mixer.music.play()
                current_music = 'sad'
                music_start_time = time.time()
        
        elif emotion_counts['neutral'] > 0:
            if not neutral_music_playing and (time.time() - neutral_hysteresis_timer > neutral_hysteresis_duration):
                if average_neutral_score >= 0.5:
                    pygame.mixer.music.load(animal_music)
                    pygame.mixer.music.play()
                    current_music = 'neutral'
                    neutral_music_playing = True
                    music_start_time = time.time()
        
        # Stop the music if no face is detected or time exceeds music duration
        if time.time() - music_start_time > music_duration or not face_detected:
            pygame.mixer.music.stop()
            neutral_music_playing = False
            current_music = None
        
        # Update hysteresis timer for neutral emotion
        if emotion_counts['neutral'] > 0:
            neutral_hysteresis_timer = time.time()
        
        true_label = 1 if sentiment_label == 'Positive' else 0
        true_labels.append(true_label)
        predicted_scores.append(sentiment_scores['compound'])
        
        cv2.imshow("Output", im)
        cv2.waitKey(27)
    
    except cv2.error:
        pass

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(true_labels, predicted_scores)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc="lower right")
plt.show()

