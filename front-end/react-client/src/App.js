import React, { useState, useRef, useEffect } from "react";
import "./App.css";

function App() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]); // { id, sender: "user" | "bot", text }
  const messagesEndRef = useRef(null);

  // AUDIO + FADE STATE
  const audioRef = useRef(null);
  const fadeActive = useRef(false);
  const fadeTargetRef = useRef(1.0);
  const fadeRafRef = useRef(null);
  const lastTRef = useRef(null);
  const [volume, setVolume] = useState(1); // 0..1
  const [tempo, setTempo] = useState(1);   // placeholder for future tempo handling
  const [fading, setFading] = useState(false);
  const [fadeSpeed, setFadeSpeed] = useState(50); // 0..100, higher = faster

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Keep element volume in sync with slider
  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.volume = Math.max(0, Math.min(1, volume));
    }
  }, [volume]);

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userText = input;
    const userMsg = {
      id: Date.now(),
      sender: "user",
      text: userText,
    };

    // Show the user message immediately
    setMessages((prev) => [...prev, userMsg]);
    setInput("");

    try {
      const res = await fetch("http://localhost:5000/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userText }),
      });
      const data = await res.json();

      const botMsg = {
        id: Date.now() + 1,
        sender: "bot",
        text: data.reply,
      };

      setMessages((prev) => [...prev, botMsg]);
    } catch (err) {
      console.error("Error talking to backend:", err);
    }
  };

  // Local audio playback controls (front-end only)
  const play = () => {
    if (audioRef.current) {
      audioRef.current.play();
    }
  };
  const pause = () => {
    audioRef.current?.pause();
  };

  const MIN_FADE_MS = 2500;   // fastest ~2.5s
  const MAX_FADE_MS = 10000;  // slowest ~10s
  const fadeDurationMs = (pct) => {
    const t = Math.max(0, Math.min(100, pct)) / 100; // normalize 0..1
    // Higher speed value = shorter duration (invert interpolation)
    return Math.round(MAX_FADE_MS - (MAX_FADE_MS - MIN_FADE_MS) * t);
  };

  // ---- Fade helpers ----
  const fadeTo = (target) => {
    if (!audioRef.current) return;

    fadeTargetRef.current = Math.max(0, Math.min(1, target));
    fadeActive.current = true;
    setFading(true);
    lastTRef.current = null;

    const step = (tNow) => {
      if (!fadeActive.current || !audioRef.current) return;

      const cur = audioRef.current.volume;
      const target = fadeTargetRef.current;
      const remaining = target - cur;

      // Finish condition
      if (Math.abs(remaining) <= 0.001) {
        audioRef.current.volume = target;
        setVolume(target);
        fadeActive.current = false;
        setFading(false);
        fadeRafRef.current = null;
        return;
      }

      const dt = lastTRef.current ? (tNow - lastTRef.current) / 1000 : 0;
      lastTRef.current = tNow;

      // Recompute desired duration from the *current* slider value
      const durSec = Math.max(0.05, fadeDurationMs(fadeSpeed) / 1000);
      // Velocity chosen so remaining reaches target in ~durSec seconds
      const vPerSec = remaining / durSec;

      let next = cur + vPerSec * dt;

      // Clamp and avoid overshoot
      if ((remaining > 0 && next > target) || (remaining < 0 && next < target)) {
        next = target;
      }
      next = Math.max(0, Math.min(1, next));

      audioRef.current.volume = next;
      setVolume(next);

      fadeRafRef.current = requestAnimationFrame(step);
    };

    if (fadeRafRef.current) cancelAnimationFrame(fadeRafRef.current);
    fadeRafRef.current = requestAnimationFrame(step);
  };

  const fadeIn = () => fadeTo(1.0);
  const fadeOut = () => fadeTo(0.0);
  const stopFade = () => {
    fadeActive.current = false;
    setFading(false);
    if (fadeRafRef.current) {
      cancelAnimationFrame(fadeRafRef.current);
      fadeRafRef.current = null;
    }
  };

  return (
    <div className="chat-page">
      {/* Inline audio element. Put a file at /public/music/track.wav */}
      <audio ref={audioRef} src="/music/track.wav" preload="auto" />

      <div className="slider-container">
        <h2>DJ Controls</h2>

        <div className="slider-child">
          <label htmlFor="volume">Volume:</label>
          <input
            type="range"
            id="volume"
            name="volume-slider"
            min="0"
            max="100"
            value={Math.round(volume * 100)}
            onChange={(e) => setVolume(parseInt(e.target.value, 10) / 100)}
            className="slider"
          />
        </div>

        <div className="slider-child">
          <label htmlFor="tempo">Tempo:</label>
          <input
            type="range"
            id="tempo"
            name="tempo-slider"
            min="50"
            max="150"
            step="1"
            value={Math.round(tempo * 100)}
            onChange={(e) => setTempo(parseInt(e.target.value, 10) / 100)}
            className="slider"
          />
        </div>

        <div className="slider-child">
          <label htmlFor="fadeSpeed">Fade Speed:</label>
          <input
            type="range"
            id="fadeSpeed"
            name="fade-speed-slider"
            min="0"
            max="100"
            step="1"
            value={fadeSpeed}
            onChange={(e) => setFadeSpeed(parseInt(e.target.value, 10))}
            className="slider"
          />
          <div style={{ fontSize: "0.85em", opacity: 0.7 }}>
            {(fadeDurationMs(fadeSpeed) / 1000).toFixed(1)}s
          </div>
        </div>

        <div>
          <button onClick={play}>Play</button>
          <button onClick={pause}>Pause</button>
          <button onClick={fadeIn} disabled={fading}>Fade In</button>
          <button onClick={fadeOut} disabled={fading}>Fade Out</button>
          <button onClick={stopFade} disabled={!fading}>Stop Fade</button>
        </div>
      </div>

      <div className="chat-window">
        <div className="chat-header">AI DJ</div>
        <div className="messages">
          {messages.map((m) => (
            <div
              key={m.id}
              className={`message-row ${m.sender === "user" ? "user" : "bot"}`}
            >
              <div className="bubble">{m.text}</div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        <form className="input-row" onSubmit={sendMessage}>
          <input
            type="text"
            placeholder="Type a message…"
            value={input}
            onChange={(e) => setInput(e.target.value)}
          />
          <button type="submit">Send</button>
        </form>
      </div>
    </div>
  );
}

export default App;
