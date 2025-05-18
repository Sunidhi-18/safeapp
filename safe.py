import streamlit as st
import sounddevice as sd
import numpy as np
import smtplib
import requests
import time
from scipy.io.wavfile import write
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# ---------------- Configuration ----------------
sender_email = "uptoskillssunidhi@gmail.com"
sender_password = "wnpf blvr vcll evnu"
receiver_emails = ["khushangowda@jnnce.ac.in"]
THRESHOLD = 0.00015  # Lowered threshold for sensitivity
RECORD_SECONDS = 15
SAMPLE_RATE = 44100
FILENAME = "detected_threat.wav"

# ---------------- Get Location ----------------
def get_location():
    try:
        response = requests.get("https://ipinfo.io", timeout=5)
        data = response.json()
        return data.get("loc", "Unknown")
    except Exception:
        return "Location not found"

# ---------------- Detect Threat ----------------
def detect_threat(audio_data):
    amplified = audio_data * 10  # Amplify audio to detect quieter sounds
    rms = np.sqrt(np.mean(amplified**2))
    st.write(f"Amplified RMS amplitude in audio: {rms:.6f}")
    return rms > THRESHOLD

# ---------------- Send Email with Attachment ----------------
def send_sos_email_with_audio(location):
    subject = "🚨 Audio Threat Detected!"
    body = f"""An audio threat has been detected.\n\nLocation: https://www.google.com/maps?q={location}
Please review the attached recording."""

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = ", ".join(receiver_emails)
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    try:
        with open(FILENAME, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename= {FILENAME}")
            message.attach(part)
    except FileNotFoundError:
        st.error("Audio file not found.")
        return False

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_emails, message.as_string())
        return True
    except Exception as e:
        st.error(f"Failed to send alert: {e}")
        return False

# ---------------- Record Audio ----------------
def record_audio():
    try:
        st.info(f"Recording audio for {RECORD_SECONDS} seconds... Please be silent or play test audio near your mic.")
        audio_data = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        write(FILENAME, SAMPLE_RATE, audio_data)
        return audio_data.flatten()
    except Exception as e:
        st.error(f"Error recording audio: {e}")
        return np.array([])

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Audio Threat Detection")
st.title("🎙️ Passive Audio Threat Detection")

if st.button("▶️ Start Audio Threat Detection"):
    audio = record_audio()
    if len(audio) == 0:
        st.error("⚠️ Audio capture failed.")
    else:
        if detect_threat(audio):
            loc = get_location()
            st.error("🚨 Threat Detected! Sending alert...")
            if send_sos_email_with_audio(loc):
                st.success("✅ Alert sent successfully with audio.")
            else:
                st.error("❌ Failed to send the alert.")
        else:
            st.info("No threat detected in the recording.")
