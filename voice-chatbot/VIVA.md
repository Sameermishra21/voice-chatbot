# Viva Preparation – Voice-Enabled Chatbot

### 1. Explain your project in one minute.
It is a web chatbot you talk to. On the Streamlit website I click the microphone, the browser asks for permission, and my voice is recorded live. When I click stop, the recording goes to the Python app, which converts speech to text using a speech recognition service. The text then goes to a deep learning model I trained myself (an Embedding layer plus a Bidirectional LSTM) that classifies it into one of 17 intents, such as *courses*, *fees* or *greeting*. The app picks a response for that intent and shows the recognized text, the intent, the confidence and the answer, and can speak the answer aloud. It is deployed on Streamlit Community Cloud over HTTPS.

### 2. Explain the architecture.
Three layers:
- **UI** (Streamlit, `streamlit_app.py`): the `st.audio_input` microphone widget, result display, and text-to-speech through the browser's `speechSynthesis`.
- **Processing** (Python): `speech_to_text.py` validates the WAV and calls speech-to-text; `nlp_utils.py` cleans, tokenizes and pads the text and runs the model. The model is loaded once with `st.cache_resource`.
- **Model** (TensorFlow/Keras): trained offline by `train_model.py`; saved as `chatbot_model.keras` plus `model_metadata.json` (vocabulary and label map).
Data flow: mic → 16 kHz WAV → Streamlit server → STT → text → tokenize/pad → BiLSTM → softmax → intent + confidence → response → page.
I also built an alternative version with a custom HTML/JavaScript frontend and a FastAPI `POST /process-audio` endpoint that reuses the same Python modules.

### 3. Why speech recognition?
Voice is faster and more natural than typing, works hands-free, and helps users who find typing hard. Speech recognition is the bridge that turns the audio signal into text, which the NLP model can then understand.

### 4. How does microphone input reach the backend?
In the Streamlit app:
1. Clicking the microphone in `st.audio_input` calls the browser's `navigator.mediaDevices.getUserMedia({audio: true})`. The browser shows the permission prompt and returns a live `MediaStream`.
2. The widget records until I click stop, then encodes the audio as a **16 kHz WAV** (`sample_rate=16000`).
3. Streamlit sends it to the Python server over its WebSocket connection and reruns the script; `st.audio_input` returns the recording as bytes.
4. I hash the bytes so each recording is processed only once, then pass them to `transcribe_wav()`.
In the alternative HTML version, an `AudioWorklet` collects raw samples, JavaScript builds the WAV itself, and `fetch()` posts it to `/process-audio`.
In both cases the user never picks a file.

### 5. How does speech-to-text work?
The audio is split into short frames (about 25 ms) and converted to spectral features (log-mel spectrogram). An acoustic neural network maps these features to probabilities of speech sounds and characters, and a language model plus a decoder (beam search) chooses the most probable word sequence. I use the Google Web Speech API through the `SpeechRecognition` library. It returns the best transcript, or raises `UnknownValueError` if nothing intelligible was found; I turn that into a "speech not recognized" message.

### 6. Why a BiLSTM?
- Word **order** matters in sentences, and an LSTM processes words in sequence while keeping memory through its gates (input, forget, output). That solves the vanishing-gradient problem of plain RNNs.
- **Bidirectional** means one LSTM reads forwards and one backwards, so each position sees both left and right context. In short questions the key word can be at the start or the end.
- It is small and trains in seconds on a CPU, which suits a small dataset. A Transformer like BERT would be more accurate but is much heavier.

### 7. What is intent classification?
It is a text classification task: mapping a user's sentence to the *goal* behind it. "How much do I pay per year?" and "tell me the fee structure" have different words but the same intent, `fees`. The chatbot answers based on the intent, not the exact words.

### 8. How is the confidence calculated?
The last layer is a softmax: *pᵢ = e^{zᵢ} / Σⱼ e^{zⱼ}*, which gives 17 probabilities summing to 1. The predicted intent is the one with the highest probability, and that probability is the confidence (for example 0.954 = 95.4 %). If it is below 0.60, or no word is in the vocabulary, the bot says *"I'm not sure I understood that. Could you please say it again?"*

### 9. How does the chatbot generate the response?
It is retrieval-based. Each intent in `intents.json` has a list of responses, and after prediction the server picks one at random (`random.choice`) from the predicted intent's list. The model does not generate new sentences; this keeps answers accurate and controlled.

### 10. How does the frontend communicate with the backend?
In Streamlit, the browser and the Python server are connected by a WebSocket. Each widget interaction (for example a finished recording) is sent to the server, the script reruns from top to bottom, and the new page elements are sent back to the browser. The model is cached with `st.cache_resource`, so it is not reloaded on every rerun. Errors (empty or silent recording, speech not recognized, service down) are shown with `st.error`.
In the alternative version the frontend uses `fetch()`: `POST /process-audio` with multipart form data, and JSON back (`recognized_text`, `intent`, `confidence`, `response`).

### 11. Why is HTTPS required for microphone access?
Browsers only expose `getUserMedia` in a **secure context** (`https://` or `http://localhost`). Over plain HTTP, anyone on the network could modify the page and secretly record the user, and HTTPS guarantees the page really comes from the site the user granted permission to. The permission is remembered per origin. Streamlit Community Cloud serves apps over HTTPS, and locally `http://localhost` counts as secure.

### 12. How is the application deployed?
The code, dataset and trained model are in a public GitHub repository. On Streamlit Community Cloud I connected my GitHub account, selected the repository, branch `main` and main file `streamlit_app.py`, and chose Python 3.12. Streamlit Cloud installs `requirements.txt`, starts the app and serves it at `https://<name>.streamlit.app` over HTTPS. Every push to GitHub redeploys it automatically.

---

### Likely follow-up questions
- **What is an embedding?** A trainable lookup table that maps each word index to a dense vector (64 numbers here). Words used in similar contexts get similar vectors, unlike one-hot vectors, which are sparse and treat every word as equally different.
- **What is padding / masking?** Sentences have different lengths, so I pad with 0 to length 12. With `mask_zero=True` the LSTM skips those zeros.
- **What is OOV?** An out-of-vocabulary word, one not seen in training. It maps to index 1.
- **What is Dropout?** During training it randomly sets 50 % of the Dense outputs to zero, forcing the network not to rely on individual neurons. This reduces over-fitting. It is disabled at prediction time.
- **Which loss and optimiser?** Sparse categorical cross-entropy (for integer labels with multiple classes) and Adam (adaptive learning rates).
- **How did you avoid over-fitting?** Dropout, a stratified validation set, a vocabulary built only from training data, and early stopping on validation loss with the best weights restored (epoch 16).
- **Why is validation accuracy 83 % and not 99 %?** The dataset is small, and many validation sentences contain words never seen in training. The gap shows mild over-fitting. More data or pre-trained embeddings would help.
- **Why not retrain on every page load?** Training is slow and non-deterministic. The server loads the saved model once, so every request is fast and consistent.
- **Why WAV?** The `SpeechRecognition` library reads WAV directly, so no ffmpeg conversion is needed. `st.audio_input` already returns WAV. In my HTML version I avoided MediaRecorder because it outputs different codecs per browser (WebM/Opus in Chrome, MP4/AAC in Safari).
- **Why Streamlit?** It is pure Python, deploys free with HTTPS from GitHub, and its `st.audio_input` widget gives live microphone recording without writing JavaScript.
- **What if the internet is down?** If the server cannot reach Google, the app shows that the speech recognition service is unavailable instead of inventing a transcript.
- **Where is the TTS done?** In the browser (`window.speechSynthesis`), so no audio is sent back from the server.
