import os
from pathlib import Path
from dotenv import load_dotenv


env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)

FFMPEG_PATH = os.getenv("FFMPEG_PATH", "")
if FFMPEG_PATH and os.path.exists(FFMPEG_PATH):
    os.environ["PATH"] = FFMPEG_PATH + os.pathsep + os.environ.get("PATH", "")

from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
from pydub import AudioSegment
import io
import time

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

client = ElevenLabs(api_key=ELEVENLABS_API_KEY) if ELEVENLABS_API_KEY else None
AGENT_VOICE = os.getenv("ELEVENLABS_AGENT_VOICE_ID")  # Adam - professional male
CLIENT_VOICE = os.getenv("ELEVENLABS_CLIENT_VOICE_ID")  # Bella - warm female

# The conversation
conversation = [
    {"speaker": "agent", "emotion": "professional", "text": "Hello, thank you for calling customer support. My name is Rajesh. How can I help you today?"},
    {"speaker": "client", "emotion": "frustrated", "text": "Hi Rajesh. I'm really upset. My bill this month is way higher than last month. I was charged 2500 instead of 1800. What's going on?"},
    {"speaker": "agent", "emotion": "empathetic", "text": "I completely understand your frustration. That's definitely concerning. Let me help you figure this out. Can I get your phone number please?"},
    {"speaker": "client", "emotion": "calmer", "text": "Sure, it's 9876543210."},
    {"speaker": "agent", "emotion": "professional", "text": "Thank you. Let me check your account. I can see the bill increase. There are several reasons this could happen. Additional data usage, new subscriptions, service charges, or a plan upgrade. Let me check your details."},
    {"speaker": "client", "emotion": "curious", "text": "I don't think I added anything. I haven't changed my plan. So it must be the data usage then?"},
    {"speaker": "agent", "emotion": "informative", "text": "Good question. Background apps like cloud storage, updates, and social media can use data without you realizing. Looking at your account, you used 35 gigabytes this month compared to your usual 20. That's an extra 15 gigabytes. Since your plan includes 25 gigabytes, you were charged for the extra 10 gigabytes at 75 rupees per gigabyte. That's 1125 rupees in overage charges."},
    {"speaker": "client", "emotion": "surprised", "text": "Oh wow, 35 gigabytes? I didn't use that much intentionally. So that's the extra charge then?"},
    {"speaker": "agent", "emotion": "helpful", "text": "Yes, that's part of it. But here's the good news. We can prevent this from happening again. I can upgrade you to our Unlimited Plan at 899 rupees per month. You get unlimited data with no overage charges. Or the Premium Data Plan at 699 rupees with 150 gigabytes and rollover. What sounds best to you?"},
    {"speaker": "client", "emotion": "interested", "text": "The unlimited plan sounds good. But when will the change take effect?"},
    {"speaker": "agent", "emotion": "clear", "text": "Great choice. The upgrade takes effect immediately for your next billing cycle. So this month's charges stay as is, but starting next month you'll be on unlimited and have no surprise charges. I'm also going to waive 30 percent of your overage charges since this was unexpected. Your new bill will be about 1750 instead of 2500. How does that sound?"},
    {"speaker": "client", "emotion": "grateful", "text": "Really? You're waiving some charges? That's amazing! Thank you so much. I really appreciate that."},
    {"speaker": "agent", "emotion": "warm", "text": "You're very welcome! Your upgrade is confirmed and will be active within 5 minutes. You'll get an SMS confirmation. Is there anything else I can help you with?"},
    {"speaker": "client", "emotion": "happy", "text": "No, I think that's it. Thanks again for being so helpful and understanding. This was much better than I expected."},
    {"speaker": "agent", "emotion": "professional", "text": "Thank you for being a valued customer. Have a wonderful day, and don't hesitate to reach out if you need anything in the future!"},
]


def generate_call():
    """Generate the full telecom call audio."""
    audio_segments = []
    
    for i, turn in enumerate(conversation):
        speaker = turn["speaker"]
        text = turn["text"]
        emotion = turn["emotion"]
        
        print(f"[{i+1}/{len(conversation)}] Generating {speaker} ({emotion})...")
        
        voice_id = AGENT_VOICE if speaker == "agent" else CLIENT_VOICE
        
        if emotion in ["frustrated", "surprised"]:
            stability, style = 0.65, 0.7
        elif emotion in ["grateful", "happy"]:
            stability, style = 0.7, 0.8
        else:
            stability, style = 0.75, 0.5
        
        audio_generator = client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id="eleven_multilingual_v2",
            voice_settings=VoiceSettings(
                stability=stability,
                similarity_boost=0.75,
                style=style,
                use_speaker_boost=True,
            ),
        )
        
        audio_bytes = b"".join(audio_generator)
        
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
        audio_segments.append(audio)
        
        pause = AudioSegment.silent(duration=500)
        audio_segments.append(pause)
        
        time.sleep(0.3)
    
    print("\nCombining audio segments...")
    final_audio = audio_segments[0]
    for segment in audio_segments[1:]:
        final_audio += segment
    
    output_file = "telecom_call.mp3"
    final_audio.export(output_file, format="mp3", bitrate="192k")
    
    print(f"\n✓ SUCCESS! Audio saved as: {output_file}")
    print(f"  Duration: {len(final_audio) / 1000:.1f} seconds")


if __name__ == "__main__":
    generate_call()