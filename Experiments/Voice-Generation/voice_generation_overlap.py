"""
Voice Generation with Natural Overlap
======================================
Enhanced version that generates realistic conversations with:
- Natural interruptions and overlaps
- Variable pause lengths
- Emotion-based timing adjustments
"""

import os
import argparse
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
import random

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
client = ElevenLabs(api_key=ELEVENLABS_API_KEY) if ELEVENLABS_API_KEY else None
AGENT_VOICE = os.getenv("ELEVENLABS_AGENT_VOICE_ID")  # Adam - professional male
CLIENT_VOICE = os.getenv("ELEVENLABS_CLIENT_VOICE_ID")  # Bella - warm female


def get_available_voices():
    """Fetch available voices and select appropriate ones for agent and client."""
    global AGENT_VOICE, CLIENT_VOICE
    
    # Check for custom voices in .env first
    custom_agent = os.getenv("ELEVENLABS_AGENT_VOICE_ID")
    custom_client = os.getenv("ELEVENLABS_CLIENT_VOICE_ID")
    
    if custom_agent and custom_client:
        AGENT_VOICE = custom_agent
        CLIENT_VOICE = custom_client
        print(f"✅ Using custom voices from .env")
        return True
    
    try:
        print("🔍 Fetching available voices from ElevenLabs...")
        response = client.voices.get_all()
        voices = response.voices
        
        if not voices:
            print("❌ No voices available")
            return False
        
        print(f"📋 Found {len(voices)} available voices")
        
        # Look for male and female voices
        male_voices = [v for v in voices if 'male' in v.labels.get('gender', '').lower()]
        female_voices = [v for v in voices if 'female' in v.labels.get('gender', '').lower()]
        
        # Fallback: just pick first two different voices
        if not male_voices:
            male_voices = [v for v in voices if v not in female_voices]
        if not female_voices:
            female_voices = [v for v in voices if v not in male_voices]
        
        if male_voices and female_voices:
            AGENT_VOICE = male_voices[0].voice_id
            CLIENT_VOICE = female_voices[0].voice_id
            print(f"✅ Selected voices:")
            print(f"   Agent: {male_voices[0].name} (ID: {AGENT_VOICE})")
            print(f"   Client: {female_voices[0].name} (ID: {CLIENT_VOICE})")
            return True
        elif len(voices) >= 2:
            AGENT_VOICE = voices[0].voice_id
            CLIENT_VOICE = voices[1].voice_id
            print(f"✅ Selected voices:")
            print(f"   Agent: {voices[0].name} (ID: {AGENT_VOICE})")
            print(f"   Client: {voices[1].name} (ID: {CLIENT_VOICE})")
            return True
        else:
            print("❌ Not enough voices available")
            return False
            
    except Exception as e:
        print(f"❌ Error fetching voices: {e}")
        print("\n💡 TIP: You can manually specify voice IDs in .env:")
        print("   ELEVENLABS_AGENT_VOICE_ID=your_voice_id")
        print("   ELEVENLABS_CLIENT_VOICE_ID=your_voice_id")
        return False

# Multiple conversation scenarios with varying difficulty
SCENARIOS = {
    # NO OVERLAP SCENARIOS (Easy to Hard)
    "easy_no_overlap": {
        "name": "Easy - Short Clear Conversation",
        "difficulty": "easy",
        "has_overlap": False,
        "conversation": [
            {"speaker": "agent", "emotion": "professional", "text": "Hello, customer support. How can I help you?"},
            {"speaker": "client", "emotion": "neutral", "text": "Hi, I need help with my account."},
            {"speaker": "agent", "emotion": "helpful", "text": "Sure, I can help. What's the issue?"},
            {"speaker": "client", "emotion": "curious", "text": "I can't access my online account."},
            {"speaker": "agent", "emotion": "professional", "text": "Let me check. Can you provide your account number?"},
            {"speaker": "client", "emotion": "neutral", "text": "It's 9876543210."},
            {"speaker": "agent", "emotion": "helpful", "text": "Thank you. I've reset your password. Check your email."},
            {"speaker": "client", "emotion": "grateful", "text": "Great, thank you so much!"},
        ]
    },
    
    "medium_no_overlap": {
        "name": "Medium - Standard Support Call",
        "difficulty": "medium",
        "has_overlap": False,
        "conversation": [
            {"speaker": "agent", "emotion": "professional", "text": "Good afternoon, technical support. My name is Sarah. How may I assist you today?"},
            {"speaker": "client", "emotion": "frustrated", "text": "Hi Sarah. My internet has been really slow for the past three days. I've tried restarting the router but nothing helps."},
            {"speaker": "agent", "emotion": "empathetic", "text": "I understand how frustrating that must be. Let me run some diagnostics on your connection. Can I have your account number please?"},
            {"speaker": "client", "emotion": "calmer", "text": "Sure, it's 5544332211."},
            {"speaker": "agent", "emotion": "professional", "text": "Thank you. I see your connection speed is below normal. It looks like there might be an issue with your router firmware. Have you tried updating it recently?"},
            {"speaker": "client", "emotion": "curious", "text": "No, I didn't know I needed to. How do I do that?"},
            {"speaker": "agent", "emotion": "informative", "text": "I'll walk you through it. First, open your web browser and type 192.168.1.1. Then enter your admin credentials. Do you see the router dashboard?"},
            {"speaker": "client", "emotion": "neutral", "text": "Yes, I'm in now."},
            {"speaker": "agent", "emotion": "helpful", "text": "Perfect. Click on System Settings, then Firmware Update. Click Check for Updates and install if available. This should fix the speed issue."},
            {"speaker": "client", "emotion": "grateful", "text": "Okay, it's updating now. Thank you for the help!"},
            {"speaker": "agent", "emotion": "professional", "text": "You're welcome. The update takes about five minutes. Your internet should be back to normal speed after that. Is there anything else?"},
            {"speaker": "client", "emotion": "happy", "text": "No, that's all. Thanks again!"},
        ]
    },
    
    "hard_no_overlap": {
        "name": "Hard - Complex Multi-Issue Call",
        "difficulty": "hard",
        "has_overlap": False,
        "conversation": [
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
    },
    
    # OVERLAP SCENARIOS (Easy to Hard)
    "easy_overlap": {
        "name": "Easy Overlap - Single Interruption",
        "difficulty": "easy",
        "has_overlap": True,
        "conversation": [
            {"speaker": "agent", "emotion": "professional", "text": "Hello, customer support. How can I help?"},
            {"speaker": "client", "emotion": "excited", "text": "Hi! I just wanted to say", "overlap": 0.2},
            {"speaker": "agent", "emotion": "friendly", "text": "Oh, please go ahead!"},
            {"speaker": "client", "emotion": "grateful", "text": "Your service has been excellent. Thank you!"},
            {"speaker": "agent", "emotion": "warm", "text": "That's wonderful to hear! We appreciate your feedback."},
            {"speaker": "client", "emotion": "happy", "text": "Keep up the great work!"},
        ]
    },
    
    "medium_overlap": {
        "name": "Medium Overlap - Multiple Interruptions",
        "difficulty": "medium",
        "has_overlap": True,
        "conversation": [
            {"speaker": "agent", "emotion": "professional", "text": "Tech support, how can I help you today?"},
            {"speaker": "client", "emotion": "frustrated", "text": "My phone keeps crashing and I've had enough", "overlap": 0.3},
            {"speaker": "agent", "emotion": "empathetic", "text": "I understand your frustration. Let me help you."},
            {"speaker": "client", "emotion": "impatient", "text": "It's been happening for days! I can't", "overlap": 0.25},
            {"speaker": "agent", "emotion": "calm", "text": "I hear you. Can you tell me what happens exactly?"},
            {"speaker": "client", "emotion": "calmer", "text": "Apps freeze and then it restarts. Really annoying."},
            {"speaker": "agent", "emotion": "helpful", "text": "That sounds like a memory issue. Have you tried clearing the cache?"},
            {"speaker": "client", "emotion": "curious", "text": "No, how do I", "overlap": 0.15},
            {"speaker": "agent", "emotion": "informative", "text": "Go to Settings, then Storage, and tap Clear Cache. This should help."},
            {"speaker": "client", "emotion": "grateful", "text": "Okay, doing it now. Thanks for the help!"},
        ]
    },
    
    "hard_overlap": {
        "name": "Hard Overlap - Frequent Interruptions",
        "difficulty": "hard",
        "has_overlap": True,
        "conversation": [
            {"speaker": "agent", "emotion": "professional", "text": "Billing department, my name is Maria. How can I"},
            {"speaker": "client", "emotion": "angry", "text": "I've been charged twice! This is unacceptable", "overlap": 0.4},
            {"speaker": "agent", "emotion": "empathetic", "text": "I'm so sorry about that. Let me look into"},
            {"speaker": "client", "emotion": "frustrated", "text": "I want a refund immediately! I can't believe", "overlap": 0.5},
            {"speaker": "agent", "emotion": "calm", "text": "I completely understand. I'm checking your account right now."},
            {"speaker": "client", "emotion": "impatient", "text": "How long will this take? I need this fixed", "overlap": 0.3},
            {"speaker": "agent", "emotion": "professional", "text": "I see the duplicate charge. I'm processing the refund now. It will"},
            {"speaker": "client", "emotion": "anxious", "text": "When will I get my money back?", "overlap": 0.35},
            {"speaker": "agent", "emotion": "clear", "text": "The refund will appear in three to five business days. I'm also adding a credit to your account for the inconvenience."},
            {"speaker": "client", "emotion": "surprised", "text": "Really? A credit too? That's", "overlap": 0.2},
            {"speaker": "agent", "emotion": "warm", "text": "Yes, fifty dollars as an apology. Is there anything else I can help with?"},
            {"speaker": "client", "emotion": "grateful", "text": "No, thank you. I appreciate you fixing this so quickly."},
        ]
    },
}


def get_voice_settings(emotion: str) -> dict:
    """Get voice settings based on emotion."""
    settings = {
        "frustrated": {"stability": 0.65, "style": 0.7},
        "surprised": {"stability": 0.65, "style": 0.7},
        "grateful": {"stability": 0.7, "style": 0.8},
        "happy": {"stability": 0.7, "style": 0.8},
        "excited": {"stability": 0.6, "style": 0.85},
        "panicked": {"stability": 0.5, "style": 0.9},
        "default": {"stability": 0.75, "style": 0.5},
    }
    return settings.get(emotion, settings["default"])


def get_pause_duration(current_emotion: str, next_emotion: str) -> int:
    """Get pause duration based on emotional context (in milliseconds)."""
    # Shorter pauses for emotional/excited conversations
    if current_emotion in ["frustrated", "surprised", "excited"]:
        return random.randint(300, 500)
    # Longer pauses for calm/professional exchanges
    elif current_emotion in ["professional", "informative"]:
        return random.randint(500, 800)
    # Default medium pause
    return random.randint(400, 600)


def generate_audio_segment(text: str, voice_id: str, emotion: str) -> AudioSegment:
    """Generate a single audio segment."""
    settings = get_voice_settings(emotion)
    
    audio_generator = client.text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        model_id="eleven_multilingual_v2",
        voice_settings=VoiceSettings(
            stability=settings["stability"],
            similarity_boost=0.75,
            style=settings["style"],
            use_speaker_boost=True,
        ),
    )
    
    audio_bytes = b"".join(audio_generator)
    return AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")


def generate_call_with_overlap(scenario_key: str, output_file: str):
    """Generate call audio with natural overlaps and interruptions."""
    scenario = SCENARIOS[scenario_key]
    conversation = scenario["conversation"]
    has_overlap = scenario["has_overlap"]
    
    print("\n" + "="*60)
    print(f"GENERATING: {scenario['name']}")
    print(f"Difficulty: {scenario['difficulty'].upper()} | Overlap: {has_overlap}")
    print("="*60)
    
    # Generate all segments first
    print("\nStep 1: Generating individual audio segments...")
    segments = []
    
    for i, turn in enumerate(conversation):
        speaker = turn["speaker"]
        text = turn["text"]
        emotion = turn["emotion"]
        
        print(f"  [{i+1}/{len(conversation)}] {speaker.title()} ({emotion}): {text[:50]}...")
        
        voice_id = AGENT_VOICE if speaker == "agent" else CLIENT_VOICE
        audio = generate_audio_segment(text, voice_id, emotion)
        
        segments.append({
            "audio": audio,
            "speaker": speaker,
            "emotion": emotion,
            "overlap": turn.get("overlap", None) if has_overlap else None,
            "text": text
        })
        
        time.sleep(0.3)  # Rate limiting
    
    # Mix segments with overlaps
    print("\nStep 2: Mixing segments with natural timing...")
    final_audio = AudioSegment.silent(duration=0)
    current_position = 0
    
    for i, segment in enumerate(segments):
        audio = segment["audio"]
        overlap = segment["overlap"]
        
        if i == 0:
            # First segment starts at beginning
            final_audio = audio
            current_position = len(audio)
            print(f"  [{i+1}] {segment['speaker'].title()} starts at 0.0s (duration: {len(audio)/1000:.1f}s)")
        else:
            prev_emotion = segments[i-1]["emotion"]
            curr_emotion = segment["emotion"]
            
            if overlap:
                # Overlap: cut into previous speaker
                overlap_ms = int(overlap * 1000)
                insert_position = current_position - overlap_ms
                
                # Create overlap by mixing
                pre_overlap = final_audio[:insert_position]
                overlap_section = final_audio[insert_position:current_position]
                
                # Mix the overlapping part
                mixed_overlap = overlap_section.overlay(audio[:len(overlap_section)])
                remaining_audio = audio[len(overlap_section):]
                
                final_audio = pre_overlap + mixed_overlap + remaining_audio
                current_position = len(final_audio)
                
                print(f"  [{i+1}] {segment['speaker'].title()} INTERRUPTS at {insert_position/1000:.1f}s (overlap: {overlap:.1f}s)")
            else:
                # Normal pause
                pause_duration = get_pause_duration(prev_emotion, curr_emotion)
                pause = AudioSegment.silent(duration=pause_duration)
                
                final_audio = final_audio + pause + audio
                current_position = len(final_audio)
                
                print(f"  [{i+1}] {segment['speaker'].title()} at {(current_position - len(audio))/1000:.1f}s (pause: {pause_duration}ms)")
    
    # Export
    print("\nStep 3: Exporting final audio...")
    final_audio.export(output_file, format="mp3", bitrate="192k")
    
    print("\n" + "="*60)
    print(f"✓ SUCCESS!")
    print(f"  File: {output_file}")
    print(f"  Duration: {len(final_audio) / 1000:.1f}s")
    print(f"  Segments: {len(segments)}")
    print("="*60)


def estimate_characters():
    """Estimate total character count for API usage."""
    print(f"\n📊 Character Estimate for All Scenarios:")
    print("="*60)
    
    total_all = 0
    for key, scenario in SCENARIOS.items():
        conv_chars = sum(len(turn["text"]) for turn in scenario["conversation"])
        total_all += conv_chars
        print(f"  {scenario['name']:40s} {conv_chars:>5,} chars")
    
    print("="*60)
    print(f"  TOTAL for all 6 scenarios: {total_all:>5,} characters")
    print(f"  Free tier limit:           10,000/month")
    print(f"  Usage: {total_all/10000*100:.1f}% of free tier")
    
    if total_all <= 10000:
        print(f"  ✅ Remaining: {10000-total_all:,} characters")
    else:
        print(f"  ⚠️  WARNING: Exceeds free tier by {total_all-10000:,} characters")
        print(f"  Consider: $5/month for 30,000 characters")


def main():
    parser = argparse.ArgumentParser(
        description="Generate multiple test audio files with varying difficulty",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Scenarios generated:
  1. easy_no_overlap - Short, clear conversation
  2. medium_no_overlap - Standard support call
  3. hard_no_overlap - Complex multi-issue call
  4. easy_overlap - Single interruption
  5. medium_overlap - Multiple interruptions  
  6. hard_overlap - Frequent interruptions
        """
    )
    parser.add_argument("--scenario", type=str, choices=list(SCENARIOS.keys()),
                        help="Generate only one specific scenario")
    parser.add_argument("--estimate-only", action="store_true",
                        help="Only estimate character count, don't generate")
    
    args = parser.parse_args()
    
    if not ELEVENLABS_API_KEY:
        print("❌ ERROR: ELEVENLABS_API_KEY not found in .env file")
        print("\nSteps to fix:")
        print("1. Get API key from: https://elevenlabs.io")
        print("2. Create .env file in project root")
        print("3. Add: ELEVENLABS_API_KEY=your_key_here")
        return
    
    # Fetch available voices
    if not get_available_voices():
        print("\n❌ Failed to get voices. Cannot generate audio.")
        return
    
    if args.estimate_only:
        estimate_characters()
        return
    
    # Create output folder
    output_dir = Path("generated_audio")
    output_dir.mkdir(exist_ok=True)
    print(f"\n📁 Output folder: {output_dir.absolute()}")
    
    estimate_characters()
    print("\nPress Enter to continue or Ctrl+C to cancel...")
    input()
    
    # Generate scenarios
    if args.scenario:
        scenarios_to_gen = [args.scenario]
    else:
        scenarios_to_gen = list(SCENARIOS.keys())
    
    print(f"\n🎙️  Generating {len(scenarios_to_gen)} scenario(s)...\n")
    
    for i, scenario_key in enumerate(scenarios_to_gen, 1):
        scenario = SCENARIOS[scenario_key]
        output_file = output_dir / f"{scenario_key}.mp3"
        
        print(f"\n[{i}/{len(scenarios_to_gen)}] Starting: {scenario_key}")
        generate_call_with_overlap(scenario_key, str(output_file))
        print(f"✅ Saved: {output_file}\n")
        
        # Pause between generations
        if i < len(scenarios_to_gen):
            time.sleep(2)
    
    print("\n" + "="*60)
    print("🎉 ALL SCENARIOS GENERATED SUCCESSFULLY!")
    print("="*60)
    print(f"\n📁 Audio files saved in: {output_dir.absolute()}")
    print("\nFiles:")
    for key in scenarios_to_gen:
        print(f"  - {key}.mp3")


if __name__ == "__main__":
    main()
