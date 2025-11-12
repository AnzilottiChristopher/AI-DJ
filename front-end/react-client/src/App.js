import React, { useState, useRef, useEffect } from "react";
import "./App.css";


//Resources:
//https://www.w3schools.com/howto/howto_js_rangeslider.asp

function App() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]); // { id, sender: "user" | "bot", text }
  const messagesEndRef = useRef(null);
  var volume = document.getElementById("volume");
  var tempo = document.getElementById("tempo");

  const scrollToBottom = () => {
  messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };
  useEffect(() => {
  scrollToBottom();
  }, [messages]);

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

  const play = () => {
    fetch("http://localhost:5000/api/play", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    })
    .then((res) => res.json())
    .then((data) => {
      console.log("Play response:", data);
    })
    .catch((err) => {
      console.error("Error sending play command:", err);
    });
  };

  return (
    <div className="chat-page">
      <div className="slider-container">
        <h2>DJ Controls</h2>
        <div className = "slider-child">
          <label for="volume">Volume:</label>
          <input type="range" id="volume" name="volume-slider" min="0" max="100" class="slider"/>
        </div>
        <div className = "slider-child">
          <label for="tempo">Tempo:</label>
          <input type="range" id="tempo" name="tempo-slider" min="0" max="100" class="slider"/>
        </div>
        <div>
          <button onClick="play()">Play</button>
          <button onClick={() => {}}>Pause</button>
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


