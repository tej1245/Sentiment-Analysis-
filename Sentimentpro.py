import cv2
from deepface import DeepFace  # Using DeepFace for emotion recognition
import random
import numpy as np
import pygame  # For playing songs

# Load Haar Cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Initialize pygame mixer for playing music
pygame.mixer.init()

# PSO parameters
class Particle:
    def _init_(self, dim):
        self.position = np.random.uniform(-5, 5, dim)  # Random starting position
        self.velocity = np.random.uniform(-1, 1, dim)  # Random velocity
        self.best_position = self.position
        self.best_value = float('inf')

# Example PSO implementation for optimizing emotion detection process (hypothetically)
def pso_optimizer(num_particles, max_iter):
    dim = 2  # Let's say we optimize two parameters (just an example)
    particles = [Particle(dim) for _ in range(num_particles)]
    global_best_position = None
    global_best_value = float('inf')

    for _ in range(max_iter):
        for particle in particles:
            # Simple fitness function (you can replace this with your own logic)
            fitness_value = np.sum(particle.position**2)  # Sum of squares as example

            if fitness_value < particle.best_value:
                particle.best_value = fitness_value
                particle.best_position = particle.position

            if fitness_value < global_best_value:
                global_best_value = fitness_value
                global_best_position = particle.position

            # Update velocity and position
            w = 0.5  # Inertia weight
            c1 = 1.5  # Cognitive coefficient
            c2 = 1.5  # Social coefficient
            r1, r2 = np.random.rand(2)

            particle.velocity = w * particle.velocity + c1 * r1 * (particle.best_position - particle.position) + c2 * r2 * (global_best_position - particle.position)
            particle.position = particle.position + particle.velocity

    return global_best_position  # Optimized parameters

# Initialize webcam
webcam = cv2.VideoCapture(0)

# Run PSO optimization (example)
best_params = pso_optimizer(num_particles=10, max_iter=10)
print(f"Optimized Parameters: {best_params}")

# Initialize variables to manage song play state
current_emotion = None
song_playing = False

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    try:
        # Assume no face is detected initially
        face_detected = False

        for (x, y, w, h) in faces:
            face_detected = True  # Set to True if any face is detected
            face = frame[y:y+h, x:x+w]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Use DeepFace to detect emotions
            analysis = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)

            if analysis:
                # Ensure the emotions key exists
                if 'emotion' in analysis[0]:
                    emotions = analysis[0]['emotion']
                    dominant_emotion = analysis[0]['dominant_emotion']

                    # Only check for 'happy', 'sad', or 'neutral'
                    if dominant_emotion in ['happy', 'sad', 'neutral']:
                        # Display only the dominant emotion on the frame
                        cv2.putText(frame, f"Emotion: {dominant_emotion}", 
                                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                        # If the emotion changes, stop the previous song
                        if dominant_emotion != current_emotion:
                            if current_emotion in ['neutral', 'sad']:
                                pygame.mixer.music.stop()

                            if dominant_emotion == 'sad':
                                pygame.mixer.music.load('sentisad.mp3')  # Play the sad song
                                pygame.mixer.music.play(-1)  # Loop the song
                                song_playing = True
                            elif dominant_emotion == 'neutral':
                                pygame.mixer.music.load('sentihappy.mp3')  # Play the neutral song
                                pygame.mixer.music.play(-1)  # Loop the song
                                song_playing = True
                            elif dominant_emotion == 'happy':
                                song_playing = False  # No song needed for happy

                            current_emotion = dominant_emotion

        # Display the frame
        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except cv2.error:
        pass

# Release the webcam and close any OpenCV windows
webcam.release()
cv2.destroyAllWindows()

# Stop any playing music before exiting
pygame.mixer.music.stop()
