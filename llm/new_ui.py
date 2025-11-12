# new_ui.py
import customtkinter as ctk
import textwrap
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import queue

from new_llm import LlamaLLM

APP_TITLE = "AI DJ"
ASSISTANT_NAME = "DJ"
USER_NAME = "You"

BUBBLE_WRAP = 720
MAX_CHARS_PER_LINE = 80

ASSISTANT_BG = "#1f2937"
ASSISTANT_FG = "#e5e7eb"
USER_BG = "#2563eb"
USER_FG = "#ffffff"
THREAD_BG = "transparent"
INPUT_BG = None

def nice_time(dt: datetime) -> str:
    return dt.strftime("%I:%M %p").lstrip("0")

class ChatApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.llm = LlamaLLM()

        # background executor & result queue
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.results_q = queue.Queue()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.title(APP_TITLE)
        self.geometry("980x660")
        self.minsize(720, 480)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Thread
        self.thread_frame = ctk.CTkScrollableFrame(self, fg_color=THREAD_BG)
        self.thread_frame.grid(row=0, column=0, sticky="nsew", padx=12, pady=(12, 6))
        self.thread_frame.grid_columnconfigure(0, weight=1)

        self._add_message("assistant", f"Hi! I’m your AI DJ. Ask me anything.", timestamp=True)

        # Input
        input_container = ctk.CTkFrame(self, fg_color=THREAD_BG)
        input_container.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 12))
        input_container.grid_columnconfigure(0, weight=1)

        self.input_box = ctk.CTkTextbox(input_container, height=72, wrap="word",
                                        activate_scrollbars=False, fg_color=INPUT_BG)
        self.input_box.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.input_box.focus_set()

        self.send_btn = ctk.CTkButton(input_container, text="Send", command=self.on_send_click, width=100)
        self.send_btn.grid(row=0, column=1, sticky="e")

        self.input_box.bind("<Return>", self._on_enter)
        self.input_box.bind("<Shift-Return>", lambda e: self._insert_newline())

        # start polling for background results
        self.after(60, self._poll_results)

    # ---------- UI bits ----------
    def _add_message(self, role: str, text: str, timestamp: bool = False):
        text = text.strip("\n")
        if not text:
            return
        normalized = textwrap.fill(text, width=MAX_CHARS_PER_LINE)

        row = self._next_row()
        container = ctk.CTkFrame(self.thread_frame, fg_color="transparent")
        container.grid(row=row, column=0, sticky="ew", pady=4, padx=2)
        container.grid_columnconfigure(0, weight=1)
        container.grid_columnconfigure(1, weight=1)

        is_user = role.lower() in ("user", "you", "me")
        side_col = 1 if is_user else 0
        spacer_col = 0 if is_user else 1

        ctk.CTkLabel(container, text="").grid(row=0, column=spacer_col, sticky="ew")

        bubble = ctk.CTkFrame(container, corner_radius=16,
                              fg_color=USER_BG if is_user else ASSISTANT_BG)
        bubble.grid(row=0, column=side_col, sticky="w" if not is_user else "e")

        header_text = (USER_NAME if is_user else ASSISTANT_NAME)
        if timestamp:
            header_text += " · " + nice_time(datetime.now())

        ctk.CTkLabel(bubble, text=header_text,
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=USER_FG if is_user else ASSISTANT_FG).grid(
            row=0, column=0, sticky="w", padx=10, pady=(8, 0)
        )

        ctk.CTkLabel(bubble, text=normalized, wraplength=BUBBLE_WRAP, justify="left",
                     font=ctk.CTkFont(size=13),
                     text_color=USER_FG if is_user else ASSISTANT_FG).grid(
            row=1, column=0, sticky="w", padx=10, pady=(4, 8)
        )

        try:
            self.thread_frame._parent_canvas.yview_moveto(1.0)
        except Exception:
            pass

    def _next_row(self) -> int:
        children = [w for w in self.thread_frame.grid_slaves()]
        if not children:
            return 0
        rows = [w.grid_info().get("row", 0) for w in children]
        return max(rows) + 1

    def _on_enter(self, event):
        self.on_send_click()
        return "break"

    def _insert_newline(self):
        self.input_box.insert("insert", "\n")

    def _set_busy(self, busy: bool):
        if busy:
            self.send_btn.configure(state="disabled", text="Thinking…")
        else:
            self.send_btn.configure(state="normal", text="Send")

    # ---------- send & background ----------
    def on_send_click(self):
        content = self.input_box.get("1.0", "end").strip()
        if not content:
            return
        self.input_box.delete("1.0", "end")
        self._add_message("user", content, timestamp=True)
        self._set_busy(True)

        # hand off to background
        future = self.executor.submit(self.generate_llm_response, content)
        future.add_done_callback(lambda f: self.results_q.put(("assistant", f.result())))

    def _poll_results(self):
        try:
            while True:
                role, text = self.results_q.get_nowait()
                self._add_message(role, text, timestamp=True)
                self._set_busy(False)
        except queue.Empty:
            pass
        self.after(60, self._poll_results)

    def generate_llm_response(self, user_text: str) -> str:
        try:
            return self.llm.process_response(user_text)
        except Exception as e:
            # keep UI alive even if model errors
            return f"[ERROR] {type(e).__name__}: {e}"

if __name__ == "__main__":
    app = ChatApp()
    app.mainloop()
