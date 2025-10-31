import streamlit as st
import requests
import os
from datetime import datetime

# Page configuration - Force light theme
st.set_page_config(
    page_title="Meditalk",
    page_icon="./assets/brain.svg",
    layout="wide"
)

# Custom CSS for bright modern styling
st.markdown("""
    <style>
    /* Force light theme */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e3edf7 100%);
    }
    
    /* Main content area */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1200px;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f0f4f8 100%);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #2d3748;
    }
    
    /* Headers - bright and bold */
    h1 {
        color: #1a365d !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    h2, h3 {
        color: #2d3748 !important;
        font-weight: 600 !important;
    }
    
    /* Input fields - clean white */
    .stTextInput input, .stTextArea textarea {
        background-color: #ffffff !important;
        color: #2d3748 !important;
        border: 2px solid #cbd5e0 !important;
        border-radius: 12px !important;
        font-size: 16px !important;
    }
    
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    /* Buttons - vibrant colors */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 0.6rem 2rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Ensure button text and icons are white */
    .stButton button p, .stButton button span, .stButton button div {
        color: white !important;
    }
    
    /* Form submit button - small circular button */
    .stForm button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border-radius: 50% !important;
        width: 45px !important;
        height: 45px !important;
        padding: 0 !important;
        min-width: 45px !important;
        font-size: 18px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3) !important;
    }
    
    .stForm button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.5) !important;
    }
    
    /* Secondary button (form submit) */
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%) !important;
    }
    
    .stButton button[kind="primary"]:hover {
        box-shadow: 0 8px 20px rgba(72, 187, 120, 0.4) !important;
    }
    
    /* User message - vibrant purple gradient */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 1rem 0 1rem 20%;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
    }
    
    /* Assistant message - clean white with border */
    .assistant-message {
        background: white;
        color: #2d3748;
        padding: 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 1rem 20% 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
    }
    
    .message-role {
        font-weight: 700;
        font-size: 11px;
        margin-bottom: 0.8rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .message-content {
        line-height: 1.7;
        font-size: 15px;
    }
    
    /* Checkbox and select boxes */
    .stCheckbox, .stSelectbox {
        color: #2d3748 !important;
    }
    
    .stSelectbox > div > div {
        background-color: white !important;
        border: 2px solid #cbd5e0 !important;
        border-radius: 10px !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: white !important;
        border: 2px dashed #cbd5e0 !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #667eea !important;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 2px solid #e2e8f0;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: white !important;
        border-radius: 10px !important;
        color: #2d3748 !important;
        border: 1px solid #e2e8f0 !important;
    }
    
    /* Caption text */
    .caption {
        color: #718096 !important;
    }
    
    /* Audio player */
    audio {
        width: 100%;
        margin-top: 1rem;
    }
    
    /* Info/Success/Error boxes */
    .stAlert {
        border-radius: 10px !important;
    }
    
    /* Chat input styling */
    [data-testid="stChatInput"] {
        background-color: white !important;
        border: 2px solid #cbd5e0 !important;
        border-radius: 12px !important;
    }
    
    [data-testid="stChatInput"]:focus-within {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    [data-testid="stChatInput"] textarea {
        font-size: 16px !important;
        color: #2d3748 !important;
    }
    
    [data-testid="stChatInput"] button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    
    [data-testid="stChatInput"] button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.5) !important;
    }
    
    /* Custom send button styling */
    .send-button-wrapper {
        position: relative;
    }
    
    .send-icon-button {
        position: absolute;
        right: 12px;
        bottom: 12px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 8px;
        padding: 8px;
        cursor: pointer;
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    
    .send-icon-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.5);
    }
    
    .send-icon-button svg {
        width: 20px;
        height: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Service URLs
MULTIMEDITRON_URL = os.getenv("MULTIMEDITRON_URL", "http://localhost:5009")
MEDITRON_URL = os.getenv("MEDITRON_URL", "http://localhost:5006")
ORPHEUS_URL = os.getenv("ORPHEUS_URL", "http://localhost:5005")
BARK_URL = os.getenv("BARK_URL", "http://localhost:5008")
CSM_URL = os.getenv("CSM_URL", "http://localhost:5010")
WHISPER_URL = os.getenv("WHISPER_URL", "http://localhost:5007")

# Helper function to load SVG icons
def load_icon(icon_name, width=20, height=20, color="#667eea"):
    try:
        with open(f"assets/{icon_name}.svg", "r") as f:
            svg = f.read()
            # Replace currentColor with the specified color
            svg = svg.replace('stroke="currentColor"', f'stroke="{color}"')
            svg = svg.replace('width="24"', f'width="{width}"')
            svg = svg.replace('height="24"', f'height="{height}"')
            return svg
    except:
        return ""

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'voices' not in st.session_state:
    st.session_state.voices = {}

# Sidebar - Settings
with st.sidebar:
    settings_icon = load_icon("settings", 24, 24, "#667eea")
    st.markdown(f'<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 1rem;">{settings_icon}<h2 style="margin: 0;">Settings</h2></div>', unsafe_allow_html=True)
    
    # AI Model Selection
    brain_icon = load_icon("brain", 18, 18, "#667eea")
    st.markdown(f'<div style="display: flex; align-items: center; gap: 8px;"><span>{brain_icon}</span><span style="font-weight: 600; color: #2d3748;">AI Model</span></div>', unsafe_allow_html=True)
    ai_model = st.selectbox(
        "Select AI Model",
        ["multimeditron", "meditron"],
        format_func=lambda x: "MultiMeditron (Multimodal)" if x == "multimeditron" else "Meditron (Text-only)",
        help="MultiMeditron supports images and conversation context"
    )
    
    st.divider()
    
    # Audio Settings
    volume_icon = load_icon("volume", 18, 18, "#667eea")
    st.markdown(f'<div style="display: flex; align-items: center; gap: 8px;"><span>{volume_icon}</span><span style="font-weight: 600; color: #2d3748;">Audio Settings</span></div>', unsafe_allow_html=True)
    generate_audio = st.checkbox("Generate Audio Response", value=True)
    
    if generate_audio:
        tts_service = st.selectbox(
            "TTS Service",
            ["orpheus", "bark", "csm"],
            format_func=lambda x: "Orpheus TTS (Recommended)" if x == "orpheus" else ("Bark TTS" if x == "bark" else "CSM (Conversational - Context-Aware)")
        )
        
        # CSM doesn't use pre-defined voices (it's context-based)
        if tts_service == "csm":
            voice = '0'  # Store speaker ID as string for compatibility
        else:
            # Load voices if not already loaded for Orpheus and Bark
            if tts_service not in st.session_state.voices:
                try:
                    voice_url = f"{ORPHEUS_URL if tts_service == 'orpheus' else BARK_URL}/voices"
                    response = requests.get(voice_url, timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        if tts_service == "orpheus":
                            st.session_state.voices[tts_service] = {
                                v['id']: f"{v['name']} ({v['gender']})" 
                                for v in data.get('voices', [])
                            }
                        else:
                            english_voices = data.get('voices', {}).get('english', [])
                            st.session_state.voices[tts_service] = {
                                v: v for v in english_voices
                            }
                except:
                    st.session_state.voices[tts_service] = {"tara": "Tara (default)"}
            
            voices = st.session_state.voices.get(tts_service, {"tara": "Tara (default)"})
            voice = st.selectbox(
                "Voice",
                list(voices.keys()),
                format_func=lambda x: voices[x]
            )
    else:
        tts_service = "orpheus"
        voice = "tara"
    
    st.divider()
    
    # Voice Input Settings
    mic_icon = load_icon("microphone", 18, 18, "#667eea")
    st.markdown(f'<div style="display: flex; align-items: center; gap: 8px; margin-top: 1rem;"><span>{mic_icon}</span><span style="font-weight: 600; color: #2d3748;">Voice Input</span></div>', unsafe_allow_html=True)
    use_voice_input = st.checkbox("Enable Voice Input", value=True)
    
    if use_voice_input:
        # Live microphone recording
        st.markdown("**Record from Microphone**")
        audio_bytes = st.audio_input("Click to record your question")
        
        if audio_bytes:
            # Create a hash of the audio to track if we've already processed it
            import hashlib
            # Read the bytes for hashing
            audio_bytes.seek(0)
            audio_data = audio_bytes.read()
            audio_hash = hashlib.md5(audio_data).hexdigest()
            audio_bytes.seek(0)  # Reset for later use
            
            # Initialize last_processed_audio in session state if not exists
            if 'last_processed_audio' not in st.session_state:
                st.session_state.last_processed_audio = None
            
            # Only process if this is a new recording
            if st.session_state.last_processed_audio != audio_hash:
                st.session_state.last_processed_audio = audio_hash
                
                with st.spinner("Transcribing your voice..."):
                    try:
                        # Reset the audio bytes for reading
                        audio_bytes.seek(0)
                        audio_data_for_whisper = audio_bytes.read()
                        
                        # Send audio to Whisper for transcription
                        files = {'audio_file': ('recording.wav', audio_data_for_whisper, 'audio/wav')}
                        response = requests.post(f"{WHISPER_URL}/transcribe", files=files, timeout=60)
                        
                        if response.status_code == 200:
                            data = response.json()
                            transcribed_text = data.get('text', '')
                            st.session_state.transcribed_text = transcribed_text
                            
                            # Upload audio to CSM for context (if CSM is selected)
                            audio_url = None
                            if tts_service == "csm":
                                try:
                                    # Upload the user's voice to CSM context
                                    files_csm = {'audio_file': ('user_recording.wav', audio_data_for_whisper, 'audio/wav')}
                                    csm_response = requests.post(f"{CSM_URL}/upload_context_audio", files=files_csm, timeout=30)
                                    
                                    if csm_response.status_code == 200:
                                        csm_data = csm_response.json()
                                        audio_url = csm_data.get('relative_path')
                                        st.success(f"✓ Voice recorded for context")
                                except Exception as e:
                                    st.warning(f"Could not upload voice for context: {str(e)}")
                            
                            # Auto-submit the transcribed text with audio URL
                            user_message = {
                                "role": "user",
                                "content": transcribed_text
                            }
                            
                            # Add audio URL if available
                            if audio_url:
                                user_message["audio_url"] = audio_url
                            
                            st.session_state.conversation_history.append(user_message)
                            st.success(f"✓ Transcribed: {transcribed_text}")
                            st.rerun()
                        else:
                            st.error(f"Transcription failed: {response.status_code} - {response.text}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())
        
        st.divider()
        
        # File upload option
        # upload_file_icon = load_icon("upload_file", 18, 18, "#667eea")
        # st.markdown(f'<div style="display: flex; align-items: center; gap: 8px; margin-top: 1rem;"><span>{upload_file_icon}</span><span style="font-weight: 600; color: #2d3748;">Or upload audio file</span></div>', unsafe_allow_html=True)
        # audio_file = st.file_uploader(
        #     "Upload audio file (WAV, MP3, M4A)",
        #     type=['wav', 'mp3', 'm4a'],
        #     help="Upload a pre-recorded audio file"
        # )
        
        # if audio_file and st.button("Transcribe Uploaded File", use_container_width=True):
        #     with st.spinner("Transcribing..."):
        #         try:
        #             files = {'audio_file': (audio_file.name, audio_file.getvalue(), audio_file.type)}
        #             response = requests.post(f"{WHISPER_URL}/transcribe", files=files, timeout=60)
                    
        #             if response.status_code == 200:
        #                 data = response.json()
        #                 transcribed_text = data.get('text', '')
        #                 st.session_state.transcribed_text = transcribed_text
        #                 st.success("Transcription complete!")
        #                 st.write(f"**Transcribed:** {transcribed_text}")
        #             else:
        #                 st.error(f"Transcription failed: {response.status_code} - {response.text}")
        #         except Exception as e:
        #             st.error(f"Error: {str(e)}")
    
    # st.divider()
    
    # Advanced Settings
    with st.expander("Advanced Settings"):
        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Higher values make output more random"
        )
        
        max_length = st.slider(
            "Max Response Length",
            min_value=128,
            max_value=1024,
            value=512,
            step=128,
            help="Maximum tokens in response"
        )
    
    st.divider()
    
    # Conversation Controls
    if st.button("New Conversation", use_container_width=True, icon=":material/add_comment:"):
        if len(st.session_state.conversation_history) > 0:
            st.session_state.conversation_history = []
            st.rerun()
        else:
            st.info("No conversation to clear")
    
    # Show conversation stats
    msg_count = len(st.session_state.conversation_history)
    st.caption(f"Messages: {msg_count}")

# Main content area
st.title("MediTalk")
st.markdown("### Medical AI Assistant with Voice Response")

# Display conversation history
if len(st.session_state.conversation_history) > 0:
    st.markdown("---")
    for msg in st.session_state.conversation_history:
        if msg['role'] == 'user':
            st.markdown(f"""
                <div class="user-message">
                    <div class="message-role">You</div>
                    <div class="message-content">{msg['content']}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="assistant-message">
                    <div class="message-role">MediTalk</div>
                    <div class="message-content">{msg['content']}</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Show audio player if available
            if 'audio_url' in msg and msg['audio_url']:
                with st.expander("Listen to response"):
                    # Show warning if context was skipped
                    if msg.get('context_skipped', False):
                        st.warning(":material/warning: Audio was generated without conversation context due to length constraints. The voice may sound less natural.", icon=":material/warning:")
                    
                    audio_url = msg['audio_url']
                    if not audio_url.startswith('http'):
                        audio_url = f"http://localhost:8080{audio_url}"
                    
                    try:
                        st.audio(audio_url)
                    except Exception as e:
                        st.error(f"Could not load audio: {e}")
    
    st.markdown("---")

# Pre-fill with transcribed text if available
default_text = st.session_state.get('transcribed_text', '')
if default_text:
    st.session_state.transcribed_text = ''
    st.session_state.prefill_text = default_text

# Use chat_input for a cleaner interface
question = st.chat_input(
    "Ask your medical question...",
    key="question_input"
)

# Handle chat input submission
if question and question.strip():
    # Add user message to history immediately
    st.session_state.conversation_history.append({
        "role": "user",
        "content": question
    })
    st.rerun()

# Check if we need to generate a response (last message is from user and no processing flag)
if (len(st.session_state.conversation_history) > 0 and 
    st.session_state.conversation_history[-1]['role'] == 'user' and
    not st.session_state.get('processing', False)):
    
    st.session_state.processing = True
    last_question = st.session_state.conversation_history[-1]['content']
    
    with st.spinner("Thinking... This may take a few minutes..."):
        try:
            endpoint = f"{MULTIMEDITRON_URL if ai_model == 'multimeditron' else MEDITRON_URL}/ask"
            
            payload = {
                "question": last_question,
                "generate_audio": generate_audio,
                "voice": voice,
                "tts_service": tts_service,
                "temperature": temperature,
                "max_length": max_length,
                "conversation_history": st.session_state.conversation_history[:-1]  # Exclude the last user message
            }
            
            # Make request
            response = requests.post(endpoint, json=payload, timeout=600)
            
            if response.status_code == 200:
                data = response.json()
                
                # Add assistant response to history
                assistant_msg = {
                    "role": "assistant",
                    "content": data.get("answer", "")
                }
                
                if generate_audio and data.get("audio_url"):
                    assistant_msg["audio_url"] = data["audio_url"]
                    # Include context_skipped flag if present (for CSM warnings)
                    if data.get("context_skipped"):
                        assistant_msg["context_skipped"] = True
                
                st.session_state.conversation_history.append(assistant_msg)
                st.session_state.processing = False
                st.rerun()
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
                st.session_state.conversation_history.pop()  # Remove user message on error
                st.session_state.processing = False
                
        except requests.exceptions.Timeout:
            st.error("Request timed out. Please try again.")
            if st.session_state.conversation_history:
                st.session_state.conversation_history.pop()
            st.session_state.processing = False
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            if st.session_state.conversation_history:
                st.session_state.conversation_history.pop()
            st.session_state.processing = False
