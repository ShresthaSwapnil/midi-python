import cv2
import mediapipe as mp
import pygame
import time
import os

# --- Configuration ---
AUDIO_FOLDER = "audio" # Folder containing your chord WAV files
WEBCAM_ID = 0          # Usually 0 for the default webcam
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7

# Map finger counts to audio filenames (customize this!)
# Ensure you have corresponding files in the AUDIO_FOLDER
finger_chord_map = {
    1: "C_chord.wav",    # Example: 1 finger = C Major
    2: "Am_chord.wav",  # Example: 2 fingers = A Minor
    3: "F_chord.wav",   # Example: 3 fingers = G Major
    4: "G_chord.wav",   # Example: 4 fingers = F Major
    5: "C_chord.wav",  # Example: 5 fingers = E Minor
    # Add more mappings as needed
}

# --- Initialization ---
print("Initializing...")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    max_num_hands=1 # Process only one hand for simplicity
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize Pygame Mixer for audio playback
try:
    pygame.mixer.init()
    print("Pygame Mixer initialized.")
except pygame.error as e:
    print(f"Error initializing Pygame Mixer: {e}")
    print("Audio playback will be disabled.")
    pygame_mixer_ok = False
else:
    pygame_mixer_ok = True

# Load audio files
chord_sounds = {}
if pygame_mixer_ok:
    print("Loading audio files...")
    for fingers, filename in finger_chord_map.items():
        filepath = os.path.join(AUDIO_FOLDER, filename)
        if os.path.exists(filepath):
            try:
                chord_sounds[fingers] = pygame.mixer.Sound(filepath)
                print(f" - Loaded: {filename}")
            except pygame.error as e:
                print(f"Error loading sound file {filepath}: {e}")
        else:
            print(f"Warning: Audio file not found: {filepath}")
    print(f"Loaded {len(chord_sounds)} audio files.")


# Initialize Webcam
cap = cv2.VideoCapture(WEBCAM_ID)
if not cap.isOpened():
    print(f"Error: Could not open webcam {WEBCAM_ID}.")
    exit()
print(f"Webcam {WEBCAM_ID} opened successfully.")


# --- State Variables ---
current_fingers_up = 0
last_played_fingers = -1 # Use -1 to indicate nothing played initially
debounce_time = 0.3 # Seconds to wait before playing another chord
last_sound_play_time = 0

# --- Main Loop ---
print("Starting detection loop... Press 'q' to quit.")
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a later selfie-view display
    # Convert the BGR image to RGB before processing
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Convert back to BGR for OpenCV display

    fingers_up_count = 0
    hand_label = ""

    if results.multi_hand_landmarks:
        # Use only the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        handedness = results.multi_handedness[0] # Get handedness info
        hand_label = handedness.classification[0].label # "Left" or "Right"

        # --- Finger Counting Logic ---
        # Based on landmark positions (y-coordinate: lower value means higher up)
        # Tip IDs: 4 (Thumb), 8 (Index), 12 (Middle), 16 (Ring), 20 (Pinky)
        # PIP IDs (approx): 6 (Index), 10 (Middle), 14 (Ring), 18 (Pinky)
        # MCP IDs (approx): 5 (Index), 9 (Middle), 13 (Ring), 17 (Pinky)
        # Thumb IP: 3, Thumb MCP: 2
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        thumb_tip = 4
        thumb_ip = 3
        thumb_mcp = 2 # Use MCP for vertical thumb check

        lm = hand_landmarks.landmark # List of landmark objects

        # Count fingers 2-5 (Index to Pinky)
        for tip_id, pip_id in zip(finger_tips, finger_pips):
            if lm[tip_id].y < lm[pip_id].y: # Check if tip is higher than PIP joint
                fingers_up_count += 1

        # Count Thumb (more complex - check if tip is 'out' or 'up')
        # Simple vertical check: Is thumb tip higher than IP or MCP?
        # This might need adjustment depending on hand orientation.
        # A more robust check often compares x-coordinates relative to hand direction.
        if lm[thumb_tip].y < lm[thumb_ip].y and lm[thumb_tip].y < lm[thumb_mcp].y :
             fingers_up_count += 1
        # Alternate thumb logic (more robust for side-ways thumb):
        # if hand_label == "Right": # Thumb tip x should be less than IP x
        #      if lm[thumb_tip].x < lm[thumb_ip].x: fingers_up_count += 1
        # else: # Left hand: Thumb tip x should be greater than IP x
        #      if lm[thumb_tip].x > lm[thumb_ip].x: fingers_up_count += 1


        # Draw landmarks and connections
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    # --- Chord Playing Logic ---
    current_time = time.time()

    # Only trigger a new chord if the finger count *changes*
    # and enough time has passed since the last chord (debounce)
    if fingers_up_count != last_played_fingers and pygame_mixer_ok:
         if (current_time - last_sound_play_time > debounce_time):
            # Stop any currently playing sound first
            pygame.mixer.stop()

            # Check if the current finger count maps to a chord
            if fingers_up_count in chord_sounds:
                print(f"Playing chord for {fingers_up_count} fingers ({finger_chord_map[fingers_up_count]})")
                chord_sounds[fingers_up_count].play()
                last_played_fingers = fingers_up_count
                last_sound_play_time = current_time
            else:
                # If the new finger count doesn't map, reset last_played
                last_played_fingers = 0 # Or -1 if you prefer

    # If no hand is detected, stop sound and reset
    elif not results.multi_hand_landmarks and last_played_fingers != -1:
         if pygame_mixer_ok:
             pygame.mixer.stop()
         last_played_fingers = -1

    # Display finger count on screen
    cv2.putText(image, f'Fingers: {fingers_up_count}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    if hand_label:
         cv2.putText(image, f'Hand: {hand_label}', (10, 120), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)


    # Display the image
    cv2.imshow('Hand Gesture Chord Player - Press Q to Quit', image)

    # Exit condition
    if cv2.waitKey(5) & 0xFF == ord('q'):
        print("Quitting...")
        break

# --- Cleanup ---
print("Cleaning up...")
if pygame_mixer_ok:
    pygame.mixer.quit()
hands.close()
cap.release()
cv2.destroyAllWindows()
print("Done.")