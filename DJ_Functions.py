import re, os, tkinter as tk
from pyo import Server, SfPlayer, Fader
from pathlib import Path
import sys
class DJFunctions:
    def __init__(self, paths=None):
        try:        
            # self.server = Server().boot()
            self.server = Server(
                sr=44100,
                nchnls=2,
                buffersize=512,
                duplex=0,
                winhost="wasapi",  # or "wasapi" if your test used that
            ).boot()
            self.server.start()

            self.tracks = {}
            if paths:
                for path in paths:
                    title = path.stem
                    player = SfPlayer(str(path), speed=1.0, loop=True, mul=0.5)
                    fader = Fader(fadein=0.01, fadeout=0.01, mul=1.0).play()
                    player.mul = fader
                    self.tracks[title] = {
                        'path': path,
                        'player': player,
                        'fader': fader,
                        'speed': 1.0,
                    }
            self.current_title = None
        except Exception as e:
            print("Error initializing DJFunctions:", e)
            sys.exit(1)

    def play(self, title):
        if title not in self.tracks:
            print("Track not included")
            return
        self.current_title = title
        self.tracks[title]['player'].out()
    
    def stop(self):
        if self.current_title:
            self.tracks[self.current_title]['player'].stop()
            self.current_title = None
    
    def change_tempo(self, rate):
        if self.current_title:
            self.tracks[self.current_title]['player'].speed = rate
    
    def change_pitch(self, semitones):
        if self.current_title:
            factor = 2 ** (semitones / 12)
            self.tracks[self.current_title]['player'].speed = factor
    
    def set_volume(self, vol):
        if self.current_title:
            self.tracks[self.current_title]['fader'].mul = vol

    def get_song(self, title):
        return self.tracks.get(title, None)

    def launch_gui(self):
        """Launch simple pyo GUI sliders for tempo and volume."""
        if not self.current_title:
            print("Play a track first!")
            return
        track = self.tracks[self.current_title]

        root = tk.Tk()
        root.title(f"DJ Controls - {self.current_title}")

        # Tempo slider
        tk.Label(root, text="Tempo").pack()
        tempo_slider = tk.Scale(root, from_=0.5, to=2.0, resolution=0.01, orient='horizontal',
                                command=lambda val: self.change_tempo(float(val)))
        tempo_slider.set(track['player'].speed)
        tempo_slider.pack()

        # Volume slider
        tk.Label(root, text="Volume").pack()
        vol_slider = tk.Scale(root, from_=0.0, to=1.0, resolution=0.01, orient='horizontal',
                              command=lambda val: self.set_volume(float(val)))
        vol_slider.set(track['fader'].mul)
        vol_slider.pack()

        root.mainloop()

if __name__ == "__main__":
    
    track_path = Path("wav_files/wakemeup-avicii.wav")
    title = track_path.stem
    dj = DJFunctions([track_path])
    dj.play(title)

    dj.launch_gui()
