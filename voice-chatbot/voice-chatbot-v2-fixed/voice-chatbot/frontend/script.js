/*
 * script.js — browser side of the Voice-Enabled Chatbot
 *
 * Flow:
 *   🎤 Start  -> navigator.mediaDevices.getUserMedia()  (browser shows the
 *                "Allow microphone?" prompt the first time)
 *             -> Web Audio API captures raw PCM samples from the Mac microphone
 *   ⏹ Stop   -> samples are down-sampled to 16 kHz mono and encoded as a WAV file
 *             -> POST /process-audio  (multipart/form-data)
 *             -> JSON {recognized_text, intent, confidence, response}
 *             -> displayed on the page, optionally spoken with speechSynthesis
 *
 * Why Web Audio instead of MediaRecorder?  MediaRecorder produces WebM/Opus
 * (Chrome) or MP4/AAC (Safari), which the server would have to convert with
 * ffmpeg.  Capturing PCM ourselves gives a plain WAV that works identically in
 * Safari, Chrome, Edge and Firefox on macOS and needs no conversion.
 */

(() => {
  "use strict";

  const TARGET_RATE = 16000;     // speech recognisers expect 16 kHz
  const MAX_SECONDS = 15;        // auto-stop safety limit
  const MIN_SECONDS = 0.4;       // shorter = "empty recording"
  const API_URL = "/process-audio";

  // ---- DOM ------------------------------------------------------------------
  const $ = (id) => document.getElementById(id);
  const startBtn = $("startBtn"), stopBtn = $("stopBtn"), playBtn = $("playBtn");
  const statusEl = $("status"), meterFill = $("meterFill"), timerEl = $("timer");
  const recognizedEl = $("recognizedText"), intentEl = $("intent");
  const confEl = $("confidence"), confFill = $("confFill"), responseEl = $("response");
  const topPredsEl = $("topPreds"), timingEl = $("timing"), chatLog = $("chatLog");
  const autoPlay = $("autoPlay"), serverStatus = $("serverStatus");

  // ---- Recording state --------------------------------------------------------
  let audioCtx = null, mediaStream = null, sourceNode = null, workletNode = null, processorNode = null;
  let chunks = [], inputRate = 44100, startTime = 0, timerId = null, isRecording = false;
  let lastResponse = "";

  // ---- UI helpers -------------------------------------------------------------
  function setStatus(text, kind) {
    statusEl.textContent = text;
    statusEl.className = "status " + kind;
  }

  function setField(el, text, isPlaceholder = false) {
    el.textContent = text;
    el.classList.toggle("placeholder", isPlaceholder);
  }

  function addMessage(text, who, tag) {
    const div = document.createElement("div");
    div.className = "msg " + who;
    div.textContent = text;
    if (tag) {
      const span = document.createElement("span");
      span.className = "tag";
      span.textContent = tag;
      div.appendChild(span);
    }
    chatLog.appendChild(div);
    chatLog.scrollTop = chatLog.scrollHeight;
  }

  function resetResultFields() {
    setField(recognizedEl, "Listening…", true);
    setField(intentEl, "—", true);
    intentEl.classList.remove("unknown");
    setField(confEl, "—", true);
    confFill.style.width = "0%";
    setField(responseEl, "Waiting for your question…", true);
    topPredsEl.innerHTML = "";
    timingEl.textContent = "";
    playBtn.disabled = true;
  }

  // ---- Server health check ------------------------------------------------------
  async function checkServer() {
    try {
      const r = await fetch("/health", { cache: "no-store" });
      const j = await r.json();
      if (j.model_loaded) {
        serverStatus.textContent = `Server online · ${j.intents.length} intents loaded`;
        serverStatus.className = "server-status ok";
      } else {
        serverStatus.textContent = "Server starting – model loading…";
        serverStatus.className = "server-status checking";
        setTimeout(checkServer, 3000);
      }
    } catch {
      serverStatus.textContent = "Backend unavailable";
      serverStatus.className = "server-status down";
    }
  }

  // ---- Microphone -----------------------------------------------------------------
  function micSupportProblem() {
    if (!window.isSecureContext) {
      return "Microphone access needs a secure page. Open this site over HTTPS (or http://localhost when running locally).";
    }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      return "This browser does not support microphone capture. Please use a recent Chrome, Safari, Edge or Firefox.";
    }
    return null;
  }

  function explainMicError(err) {
    switch (err && err.name) {
      case "NotAllowedError":
      case "SecurityError":
        return "Microphone permission was denied. Click the 🔒/camera icon in the address bar, allow the microphone, then try again. (macOS: System Settings → Privacy & Security → Microphone → enable your browser.)";
      case "NotFoundError":
      case "OverconstrainedError":
        return "No microphone was found. Connect or enable a microphone and try again.";
      case "NotReadableError":
      case "AbortError":
        return "The microphone is being used by another application or could not be started. Close other apps using it and try again.";
      default:
        return "Could not access the microphone: " + (err && err.message ? err.message : err);
    }
  }

  // AudioWorklet processor source (loaded from a Blob so no extra file is needed)
  const WORKLET_SRC = `
    class RecorderProcessor extends AudioWorkletProcessor {
      process(inputs) {
        const ch = inputs[0] && inputs[0][0];
        if (ch) this.port.postMessage(ch.slice(0));
        return true;
      }
    }
    registerProcessor("recorder-processor", RecorderProcessor);
  `;

  function handleSamples(float32) {
    chunks.push(float32);
    // Level meter (RMS)
    let sum = 0;
    for (let i = 0; i < float32.length; i++) sum += float32[i] * float32[i];
    const rms = Math.sqrt(sum / float32.length);
    meterFill.style.width = Math.min(100, rms * 400) + "%";
  }

  async function startRecording() {
    const problem = micSupportProblem();
    if (problem) { setStatus("❌ " + problem, "error"); return; }

    startBtn.disabled = true;
    setStatus("Requesting microphone permission…", "processing");

    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
    } catch (err) {
      setStatus("❌ " + explainMicError(err), "error");
      startBtn.disabled = false;
      return;
    }

    try {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      audioCtx = new Ctx();
      if (audioCtx.state === "suspended") await audioCtx.resume();
      inputRate = audioCtx.sampleRate;
      sourceNode = audioCtx.createMediaStreamSource(mediaStream);
      chunks = [];

      if (audioCtx.audioWorklet && window.AudioWorkletNode) {
        const url = URL.createObjectURL(new Blob([WORKLET_SRC], { type: "application/javascript" }));
        await audioCtx.audioWorklet.addModule(url);
        URL.revokeObjectURL(url);
        workletNode = new AudioWorkletNode(audioCtx, "recorder-processor");
        workletNode.port.onmessage = (e) => handleSamples(e.data);
        sourceNode.connect(workletNode);
        // Connected through a muted gain so the graph runs without echoing the mic.
        const mute = audioCtx.createGain(); mute.gain.value = 0;
        workletNode.connect(mute).connect(audioCtx.destination);
      } else {
        // Fallback for older browsers
        processorNode = audioCtx.createScriptProcessor(4096, 1, 1);
        processorNode.onaudioprocess = (e) => handleSamples(new Float32Array(e.inputBuffer.getChannelData(0)));
        sourceNode.connect(processorNode);
        processorNode.connect(audioCtx.destination);
      }
    } catch (err) {
      cleanupAudio();
      setStatus("❌ Could not start audio capture: " + err.message, "error");
      startBtn.disabled = false;
      return;
    }

    isRecording = true;
    startTime = performance.now();
    timerId = setInterval(() => {
      const s = (performance.now() - startTime) / 1000;
      timerEl.textContent = s.toFixed(1) + " s";
      if (s >= MAX_SECONDS) stopRecording();
    }, 100);

    resetResultFields();
    startBtn.classList.add("recording");
    stopBtn.disabled = false;
    setStatus("🔴 Recording… speak now, then press ⏹ Stop Speaking", "recording");
  }

  function cleanupAudio() {
    try { if (sourceNode) sourceNode.disconnect(); } catch {}
    try { if (workletNode) { workletNode.port.onmessage = null; workletNode.disconnect(); } } catch {}
    try { if (processorNode) { processorNode.onaudioprocess = null; processorNode.disconnect(); } } catch {}
    if (mediaStream) mediaStream.getTracks().forEach((t) => t.stop());   // turns the mic indicator off
    if (audioCtx && audioCtx.state !== "closed") audioCtx.close();
    audioCtx = mediaStream = sourceNode = workletNode = processorNode = null;
    clearInterval(timerId);
    meterFill.style.width = "0%";
  }

  async function stopRecording() {
    if (!isRecording) return;
    isRecording = false;
    const duration = (performance.now() - startTime) / 1000;
    cleanupAudio();
    startBtn.classList.remove("recording");
    stopBtn.disabled = true;

    const samples = mergeChunks(chunks);
    chunks = [];
    if (samples.length < inputRate * MIN_SECONDS || duration < MIN_SECONDS) {
      setStatus("❌ Empty recording – press Start, speak your question, then press Stop.", "error");
      setField(recognizedEl, "No speech captured.", true);
      startBtn.disabled = false;
      return;
    }

    const wavBlob = encodeWav(downsample(samples, inputRate, TARGET_RATE), TARGET_RATE);
    await sendAudio(wavBlob);
    startBtn.disabled = false;
  }

  // ---- Audio encoding ---------------------------------------------------------------
  function mergeChunks(list) {
    const total = list.reduce((n, c) => n + c.length, 0);
    const out = new Float32Array(total);
    let offset = 0;
    for (const c of list) { out.set(c, offset); offset += c.length; }
    return out;
  }

  // Average-based down-sampling (acts as a simple low-pass filter).
  function downsample(buffer, fromRate, toRate) {
    if (toRate >= fromRate) return buffer;
    const ratio = fromRate / toRate;
    const outLength = Math.floor(buffer.length / ratio);
    const out = new Float32Array(outLength);
    let pos = 0;
    for (let i = 0; i < outLength; i++) {
      const next = Math.floor((i + 1) * ratio);
      let sum = 0, count = 0;
      for (; pos < next && pos < buffer.length; pos++) { sum += buffer[pos]; count++; }
      out[i] = count ? sum / count : 0;
    }
    return out;
  }

  // 16-bit PCM mono WAV (44-byte RIFF header + samples)
  function encodeWav(samples, rate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const v = new DataView(buffer);
    const writeStr = (off, s) => { for (let i = 0; i < s.length; i++) v.setUint8(off + i, s.charCodeAt(i)); };
    writeStr(0, "RIFF");
    v.setUint32(4, 36 + samples.length * 2, true);
    writeStr(8, "WAVE");
    writeStr(12, "fmt ");
    v.setUint32(16, 16, true);          // fmt chunk size
    v.setUint16(20, 1, true);           // PCM
    v.setUint16(22, 1, true);           // mono
    v.setUint32(24, rate, true);        // sample rate
    v.setUint32(28, rate * 2, true);    // byte rate
    v.setUint16(32, 2, true);           // block align
    v.setUint16(34, 16, true);          // bits per sample
    writeStr(36, "data");
    v.setUint32(40, samples.length * 2, true);
    let off = 44;
    for (let i = 0; i < samples.length; i++, off += 2) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      v.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return new Blob([v], { type: "audio/wav" });
  }

  // ---- Backend call -----------------------------------------------------------------
  async function sendAudio(wavBlob) {
    setStatus("⏳ Processing… (speech-to-text + intent model)", "processing");
    setField(recognizedEl, "Recognizing speech…", true);

    const form = new FormData();
    form.append("audio", wavBlob, "speech.wav");

    let res, data;
    try {
      res = await fetch(API_URL, { method: "POST", body: form });
    } catch {
      setStatus("❌ Backend unavailable – the server could not be reached. Check your connection or that the server is running.", "error");
      setField(recognizedEl, "Not processed.", true);
      serverStatus.textContent = "Backend unavailable";
      serverStatus.className = "server-status down";
      return;
    }
    try { data = await res.json(); } catch { data = null; }

    if (!res.ok || !data) {
      const msg = (data && (data.message || data.detail)) || `Server error (HTTP ${res.status}).`;
      setStatus("❌ " + msg, "error");
      setField(recognizedEl, data && data.error === "speech_not_recognized" ? "Speech not recognized." : "No text recognized.", true);
      setField(responseEl, "—", true);
      return;
    }
    showResult(data);
  }

  function showResult(d) {
    setField(recognizedEl, `“${d.recognized_text}”`);
    const pct = (d.confidence * 100).toFixed(1) + "%";

    if (d.out_of_scope) {
      setField(intentEl, "out_of_scope (not a college question)");
      intentEl.classList.add("unknown");
      setStatus("🟡 Response ready – question is outside the college topics", "ready");
    } else if (d.low_confidence) {
      setField(intentEl, `unknown (closest: ${d.predicted_intent})`);
      intentEl.classList.add("unknown");
      setStatus("🟡 Response ready – low confidence, please rephrase", "ready");
    } else {
      setField(intentEl, d.intent);
      intentEl.classList.remove("unknown");
      setStatus("🟢 Response ready", "ready");
    }
    setField(confEl, pct);
    confFill.style.width = pct;
    confFill.classList.toggle("low", d.low_confidence);
    setField(responseEl, d.response);

    topPredsEl.innerHTML = "";
    (d.top_predictions || []).forEach((p) => {
      const li = document.createElement("li");
      li.textContent = `${p.intent} — ${(p.confidence * 100).toFixed(1)}%`;
      topPredsEl.appendChild(li);
    });
    if (d.timing_ms) {
      timingEl.textContent = `Speech-to-text: ${d.timing_ms.speech_to_text} ms · Intent model: ${d.timing_ms.intent_model} ms · Audio: ${d.audio ? d.audio.duration_s + " s" : "-"}`;
    }

    addMessage(d.recognized_text, "user");
    addMessage(d.response, "bot", `${d.intent} · ${pct}`);

    lastResponse = d.response;
    playBtn.disabled = !("speechSynthesis" in window);
    if (autoPlay.checked) speak(lastResponse);
  }

  // ---- Text-to-Speech (browser built-in) ---------------------------------------------
  function speak(text) {
    if (!("speechSynthesis" in window) || !text) return;
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.lang = "en-US";
    u.rate = 1;
    const voices = window.speechSynthesis.getVoices();
    const v = voices.find((x) => x.lang && x.lang.startsWith("en") && /Samantha|Google|Daniel|Karen/i.test(x.name))
           || voices.find((x) => x.lang && x.lang.startsWith("en"));
    if (v) u.voice = v;
    window.speechSynthesis.speak(u);
  }

  // ---- Wire up ----------------------------------------------------------------------
  startBtn.addEventListener("click", startRecording);
  stopBtn.addEventListener("click", stopRecording);
  playBtn.addEventListener("click", () => speak(lastResponse));
  if ("speechSynthesis" in window) window.speechSynthesis.getVoices(); // preload voices
  else autoPlay.disabled = true;

  const problem = micSupportProblem();
  if (problem) setStatus("⚠️ " + problem, "error");
  checkServer();
})();
