# src/embed_model.py
import base64
from pathlib import Path

# This script reads the .task model file and converts it into a Base64 text string.

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "src" / "models" / "face_landmarker.task"
OUTPUT_PATH = PROJECT_ROOT / "model_base64.txt"

def encode_model():
    """Reads the model file and saves it as a Base64 string in a text file."""
    try:
        with open(MODEL_PATH, "rb") as model_file:
            # Read the binary file and encode it
            encoded_string = base64.b64encode(model_file.read()).decode('utf-8')
        
        with open(OUTPUT_PATH, "w") as text_file:
            # Write the long string to an output file
            text_file.write(encoded_string)
            
        print(f"✅ Model successfully encoded to Base64.")
        print(f"   The string has been saved to: {OUTPUT_PATH}")
        print(f"\n👉 Next Step: Open '{OUTPUT_PATH.name}', copy the entire string, and paste it into your HTML file where indicated.")

    except FileNotFoundError:
        print(f"❌ Error: Could not find the model file at {MODEL_PATH}")
        print("   Please ensure 'face_landmarker.task' is inside the 'src/models' folder.")

if __name__ == "__main__":
    encode_model()
