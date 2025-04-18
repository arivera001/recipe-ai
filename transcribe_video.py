import yt_dlp
from moviepy import AudioFileClip
import openai
import cv2
import pytesseract
import os
import time

# API key!
client = openai.OpenAI(api_key="")

# Set tesseract OCR path (windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Download and extract audio using yt_dlp
def download_audio(youtube_url):
    try:
        print(f"📎 Attempting to download from URL: {youtube_url}")

        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
            'outtmpl': 'temp_video.%(ext)s',
            'quiet': True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])

        # After download, look for video file
        video_path = "temp_video.mp4"
        audio_path = "temp_audio.wav"

        # If video isn't in mp4 format, convert using ffmpeg
        if not os.path.exists(video_path):
            webm_path = "temp_video.webm"
            if os.path.exists(webm_path):
                print("⚙️ Converting .webm to .mp4...")
                os.system(f"ffmpeg -y -i {webm_path} -c:v libx264 -c:a aac {video_path}")
                os.remove(webm_path)  # Remove original if converted
            else:
                print("❌ Could not find downloaded video.")
                return None, None

        print("🎧 Extracting audio...")
        clip = AudioFileClip(video_path)
        clip.write_audiofile(audio_path)
        clip.close()

        return audio_path, video_path
    except Exception as error:
        print("❌ Error downloading or processing video:", error)
        return None, None

# Transcribe audio using Whisper
def transcribe_audio(file_path):
    try:
        with open(file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
            return transcript.text

    except Exception as error:
        print("❌ Error transcribing audio:", error)
        return None

# Extract on-screen text with OCR
def extract_text_from_video(video_path, frame_interval=30, max_frames=500):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Could not open video.")
        return ""

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🎞️ Total frames in video: {total_frames}")

    on_screen_text = []
    frame_idx = 0
    processed = 0
    start_time = time.time()

    while processed < max_frames:
        success, frame = cap.read()
        if not success:
            break

        if frame_idx % frame_interval == 0:
            try:
                print(f"🔍 Processing frame {frame_idx}...")
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                text = pytesseract.image_to_string(gray)
                if text.strip():
                    on_screen_text.append(text.strip())
                processed += 1
            except Exception as e:
                print(f"⚠️ OCR failed on frame {frame_idx}: {e}")

        frame_idx += 1

    cap.release()
    duration = time.time() - start_time
    print(f"✅ OCR completed in {duration:.2f} seconds, processed {processed} frames.")
    return "\n".join(on_screen_text)

# Analyze recipe with GPT
def analyze_recipe_text(full_text):
    try:
        print("🧠 Analyzing recipe data with GPT...")
        prompt = f"""
You are an intelligent recipe extractor.
Given the following text from both a video transcript and on-screen text, extract:
1. A list of ingredients (with quantities if available).
2. A list of step-by-step cooking instructions
3. Estimated total calories
4. Total macros (protein, carbs, fat). If macros are provided in the text, use them. If not, provide a rough estimate based on common values.

Text:
\"\"\"{full_text}\"\"\"

Format the output like this:
---
Ingredients:
- Item 1
- Item 2

Instructions:
1. Step one
2. Step two

Nutrition:
Calories: __ cal
Protein: __ g
Carbs: __ g
Fat: __ g
---
        """

        response = client.chat.completions.create(model="gpt-4", messages=[{
            "role": "system",
            "content": "You are a helpful assistant that extracts structured recipe information from messy video transcripts and screen text."
        }, {
            "role": "user",
            "content": prompt
        }])

        return response.choices[0].message.content
    except Exception as e:
        print("❌ Error analyzing recipe:", e)
        return None

# Main flow
if __name__ == "__main__":
    youtube_link = input("🔗 Paste the YouTube video link: ").strip()

    print("\n📥 Step 1: Downloading and extracting audio...")
    audio_path, video_path = download_audio(youtube_link)

    if not audio_path or not video_path:
        print("❌ Stopping due to audio download failure.")
        exit()

    print("\n🧠 Step 2: Transcribing audio...")
    transcription = transcribe_audio(audio_path)

    print("\n📸 Step 3: Extracting text...")
    on_screen_text = extract_text_from_video(video_path)

    combined_text = (transcription or "") + "\n\n" + str(on_screen_text)

    print("\n🍳 Step 4: Analyzing combined recipe data...")
    recipe_output = analyze_recipe_text(combined_text)

    if recipe_output:
        with open("recipe_info.txt", "w", encoding="utf-8") as f:
            f.write(recipe_output)
        print("\n✅ Recipe info saved to 'recipe_info.txt'")
        print("\n📋 Result:\n")
        print(recipe_output)
    else:
        print("❌ Recipe analysis failed.")

    # Clean up temp files
    if os.path.exists(audio_path):
        os.remove(audio_path)
    if os.path.exists(video_path):
        os.remove(video_path)