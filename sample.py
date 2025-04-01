import cv2
import mediapipe as mp
import pygame
import numpy as np
import sys
import signal

class GestureChordPlayer:
    def __init__(self):
        # Initialize MediaPipe Hand tracking
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Initialize Pygame for sound
        pygame.mixer.init()
        
        # Load chord sounds (you'll need to replace these with actual chord sound files)
        self.chords = {
            1: pygame.mixer.Sound('C_chord.wav'),
            2: pygame.mixer.Sound('Am_chord.wav'),
            3: pygame.mixer.Sound('F_chord.wav'),
            4: pygame.mixer.Sound('G_chord.wav')
        }
        
        # To prevent repeated chord playing
        self.last_played_chord = None
        self.chord_cooldown = 1000  # milliseconds
        self.last_play_time = 0
        
        # Exit flag
        self.should_exit = False

    def count_extended_fingers(self, hand_landmarks):
        """
        Count number of extended fingers
        Uses y-coordinate of fingertip and middle joint to determine if finger is extended
        """
        finger_tips = [
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP
        ]
        
        extended_fingers = 0
        for tip in finger_tips:
            tip_y = hand_landmarks.landmark[tip].y
            middle_y = hand_landmarks.landmark[tip-2].y  # corresponding middle joint
            
            # If tip is higher than middle joint, finger is extended
            if tip_y < middle_y:
                extended_fingers += 1
        
        return extended_fingers

    def play_chord(self, num_fingers):
        """
        Play chord based on number of extended fingers
        """
        current_time = pygame.time.get_ticks()
        
        # Check if the chord is in our dictionary and cooldown has passed
        if (num_fingers in self.chords and 
            (self.last_played_chord != num_fingers or 
             current_time - self.last_play_time > self.chord_cooldown)):
            
            # Stop any currently playing sound
            pygame.mixer.stop()
            
            # Play the chord
            self.chords[num_fingers].play()
            
            # Update tracking variables
            self.last_played_chord = num_fingers
            self.last_play_time = current_time

    def handle_exit(self, signum=None, frame=None):
        """
        Handle exit signals
        """
        print("\nExiting the application...")
        self.should_exit = True

    def run(self):
        """
        Main application loop
        """
        # Set up signal handlers for graceful exit
        signal.signal(signal.SIGINT, self.handle_exit)
        signal.signal(signal.SIGTERM, self.handle_exit)

        # Open webcam
        cap = cv2.VideoCapture(0)
        
        print("Application running. Press Ctrl+C to exit.")
        
        while cap.isOpened() and not self.should_exit:
            # Read frame from webcam
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break
            
            # Flip the image horizontally for a later selfie-view display
            image = cv2.flip(image, 1)
            
            # Convert the BGR image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image and find hands
            results = self.hands.process(image_rgb)
            
            # Draw the hand annotations on the image
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    self.mp_drawing.draw_landmarks(
                        image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Count extended fingers
                    num_fingers = self.count_extended_fingers(hand_landmarks)
                    
                    # Play corresponding chord
                    if num_fingers > 0:
                        self.play_chord(num_fingers)
            
            # Display the image
            cv2.imshow('Hand Gesture Chord Player', image)
            
            # Check for exit key (ESC key)
            key = cv2.waitKey(5) & 0xFF
            if key == 27:  # ESC key
                break
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

def main():
    print("Hand Gesture Chord Player")
    print("Instructions:")
    print("- Extend fingers to play chords")
    print("- 1 finger: C Chord")
    print("- 2 fingers: Am Chord")
    print("- 3 fingers: F Chord")
    print("- 4 fingers: G Chord")
    print("Press ESC or Ctrl+C to quit")
    
    try:
        player = GestureChordPlayer()
        player.run()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == "__main__":
    main()