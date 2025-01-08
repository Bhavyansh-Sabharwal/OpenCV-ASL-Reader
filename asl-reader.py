import cv2
import mediapipe as mp
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

ASL_LETTERS = {
    'A': [0.1, 0.85, 0.85, 0.9, 0.9],  # All fingers in a fist, thumb alongside the fist
    'B': [0.2, 0.03, 0.03, 0.03, 0.01],  # All fingers extended straight up, thumb across the palm
    'C': [0.1, 0.15, 0.2, 0.3, 0.2],  # All fingers curved to form a 'C' shape
    'D': [0.1, 0.01, 0.85, 0.85, 0.8],  # Index finger extended, other fingers in a fist, thumb touching middle finger
    'E': [0.2, 0.85, 0.9, 0.95, 0.9],  # All fingers bent to touch the thumb, forming an 'E' shape
    'F': [0.02, 0.65, 0, 0, 0.02],  # Thumb and index finger touching to form a circle, other fingers extended
    'G': [0.25, 0.05, 0.55, 0.65, 0.7],  # Thumb and index finger extended parallel, other fingers in a fist
    'H': [0.25, 0.05, 0.05, 0.65, 0.7],  # Index and middle fingers extended, other fingers in a fist
    'I': [0.15, 0.9, 0.9, 0.9, 0.05],  # Pinky finger extended, other fingers in a fist
    'J': [0.2, 0.5, 0.6, 0.7, 0.05],  # Pinky finger extended, other fingers in a fist, with a 'J' motion
    'K': [0.1, 0.05, 0.05, 0.95, 0.9],  # Thumb between extended index and middle fingers, other fingers in a fist
    'L': [0.05, 0.05, 0.7, 0.7, 0.7],  # Thumb and index finger form an 'L' shape, other fingers in a fist
    'M': [0.2, 0.8, 0.85, 0.9, 0.6],  # Thumb under three fingers, pinky finger over thumb
    'N': [0.17, 0.75, 0.8, 0.7, 0.6],  # Thumb under two fingers, ring finger over thumb
    'O': [0.1, 0.5, 0.5, 0.55, 0.5],  # All fingers curved to touch thumb, forming an 'O' shape
    'P': [0.05, 0.05, 0.15, 0.3, 0.3],  # Thumb between extended index and middle fingers, hand pointing down
    'Q': [0.1, 0, 0.5, 0.5, 0.4],  # Thumb and index finger extended downward, other fingers in a fist
    'R': [0.15, 0.02, 0.02, 0.85, 0.75],  # Index and middle fingers crossed, other fingers in a fist
    'S': [0.2, 0.85, 0.9, 0.9, 0.9],  # All fingers in a fist, thumb over fingers
    'T': [0.1, 0.8, 0.7, 0.7, 0.7],  # Thumb between index and middle fingers, other fingers in a fist
    'U': [0.15, 0.02, 0, 0.9, 0.8],  # Index and middle fingers extended together, other fingers in a fist
    'V': [0.15, 0, 0, 0.8, 0.7],  # Index and middle fingers extended apart, forming a 'V', other fingers in a fist
    'W': [0.3, 0, 0, 0, 0.9],  # Index, middle, and ring fingers extended apart, forming a 'W', pinky in a fist
    'X': [0.15, 0.1, 0.8, 0.8, 0.8],  # Index finger bent, other fingers in a fist
    'Y': [0.1, 0.8, 0.8, 0.9, 0.1],  # Thumb and pinky extended, other fingers in a fist
    'Z': [0.1, 0.1, 0.4, 0.4, 0.4],  # Index finger extended, drawing a 'Z' in the air, other fingers in a fist
}

def calculate_finger_angles(hand_landmarks):
    """Calculate angles between finger joints"""
    angles = []
    
    for finger in range(5):
        base = finger * 4 + 1
        points = []
        for i in range(4):
            point = hand_landmarks.landmark[base + i]
            points.append([point.x, point.y, point.z])
        
        # Calculate angle for each finger
        v1 = np.array(points[1]) - np.array(points[0])
        v2 = np.array(points[2]) - np.array(points[1])
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angles.append(angle)
    
    return angles

def predict_letter(angles):
    """Predict ASL letter based on finger angles"""
    max_similarity = -1
    predicted_letter = None
    
    # Normalize the input angles
    normalized_angles = [angle / np.pi for angle in angles]
    angles_normalized = np.array(normalized_angles).reshape(1, -1)
    
    for letter, pattern in ASL_LETTERS.items():
        pattern_normalized = np.array(pattern).reshape(1, -1)
        similarity = cosine_similarity(angles_normalized, pattern_normalized)[0][0]
        
        if similarity > max_similarity:
            max_similarity = similarity
            predicted_letter = letter
    
    return predicted_letter if max_similarity > 0.8 else None

def main():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                angles = calculate_finger_angles(hand_landmarks)
                predicted_letter = predict_letter(angles)
                
                for i, angle in enumerate(angles):
                    angle_deg = round(np.degrees(angle), 2)
                    cv2.putText(frame, f'Finger {i}: {angle_deg}', 
                              (50, 50 + i*30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              0.7, (0, 255, 0), 2)
                
                for i, angle in enumerate(angles):
                    normalized = round(angle / np.pi, 2)
                    cv2.putText(frame, f'Norm {i}: {normalized}', 
                              (250, 50 + i*30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              0.7, (255, 0, 0), 2)
                
                if predicted_letter:
                    cv2.putText(frame, f'Letter: {predicted_letter}', 
                              (450, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              1.5, (0, 0, 255), 2)
        
        cv2.imshow('ASL Reader', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
