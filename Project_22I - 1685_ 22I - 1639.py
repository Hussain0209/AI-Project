import tkinter as tk
from tkinter import ttk, messagebox
from gtts import gTTS
import playsound
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import MarianMTModel, MarianTokenizer
import time

class MovieGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Movie Summary Analyzer")
        self.root.geometry("600x500")

        # Define languages and translation models
        self.languages = {
            'ar': {'model_name': 'Helsinki-NLP/opus-mt-en-ar', 'name': 'Arabic', 'gtts_lang': 'ar'},
            'ur': {'model_name': 'Helsinki-NLP/opus-mt-en-ur', 'name': 'Urdu', 'gtts_lang': 'ur'},
            'ja': {'model_name': 'Helsinki-NLP/opus-mt-en-jap', 'name': 'Japanese', 'gtts_lang': 'ja'}
        }

        # Initialize translation models
        self.translators = {}
        for lang_code, lang_info in self.languages.items():
            try:
                tokenizer = MarianTokenizer.from_pretrained(lang_info['model_name'])
                model = MarianMTModel.from_pretrained(lang_info['model_name'])
                self.translators[lang_code] = {'tokenizer': tokenizer, 'model': model}
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load {lang_info['name']} translation model: {str(e)}")
                self.root.quit()

        # Load dataset
        try:
            self.df = pd.read_csv('cleaned_movie_data.csv', usecols=['cleaned_summary', 'genres'], dtype={'cleaned_summary': 'string', 'genres': 'string'})
            self.df['cleaned_summary'] = self.df['cleaned_summary'].fillna('')
            self.df['genres'] = self.df['genres'].apply(lambda x: x.split(',') if isinstance(x, str) and x else [])
            self.df['genres'] = self.df['genres'].apply(lambda x: [g.strip() for g in x if g.strip()])
        except FileNotFoundError:
            messagebox.showerror("Error", "cleaned_movie_data.csv not found. Run preprocess_split.py first.")
            self.root.quit()

        # Initialize TF-IDF vectorizer for similarity
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        self.summary_vectors = self.vectorizer.fit_transform(self.df['cleaned_summary'])

        # GUI Elements
        # Summary Input
        tk.Label(root, text="Enter Movie Summary:").pack(pady=5)
        self.summary_text = tk.Text(root, height=5, width=50)
        self.summary_text.pack(pady=5)

        # Action Selection with Buttons
        tk.Label(root, text="Select Action:").pack(pady=5)
        self.action_frame = tk.Frame(root)
        self.action_frame.pack(pady=5)
        
        self.audio_button = tk.Button(self.action_frame, text="Convert Summary to Audio", command=lambda: self.execute_action("audio"))
        self.audio_button.pack(side=tk.LEFT, padx=5)
        
        self.genre_button = tk.Button(self.action_frame, text="Predict Genre", command=lambda: self.execute_action("genre"))
        self.genre_button.pack(side=tk.LEFT, padx=5)
        
        self.exit_button = tk.Button(self.action_frame, text="Exit", command=lambda: self.execute_action("exit"))
        self.exit_button.pack(side=tk.LEFT, padx=5)

        # Language Selection
        tk.Label(root, text="Select Audio Language:").pack(pady=5)
        self.language_var = tk.StringVar(value="Arabic")
        self.language_menu = ttk.Combobox(root, textvariable=self.language_var, values=[lang_info['name'] for lang_info in self.languages.values()], state="readonly")
        self.language_menu.pack()

        # Buttons Frame for Execute and Clear
        self.button_frame = tk.Frame(root)
        self.button_frame.pack(pady=10)
        
        tk.Button(self.button_frame, text="Execute", command=lambda: self.execute_action("execute")).pack(side=tk.LEFT, padx=5)
        tk.Button(self.button_frame, text="Clear Summary", command=self.clear_summary).pack(side=tk.LEFT, padx=5)

        # Output Area
        tk.Label(root, text="Results:").pack(pady=5)
        self.output_text = tk.Text(root, height=10, width=50, state="disabled")
        self.output_text.pack(pady=5)

    def preprocess_summary(self, text):
        # Simple preprocessing to match preprocess_split.py
        text = text.lower()
        text = ''.join(c for c in text if c.isalpha() or c.isspace())
        return text.strip()

    def translate_text(self, text, lang_code):
        if not isinstance(text, str) or not text.strip():
            return ''
        try:
            tokenizer = self.translators[lang_code]['tokenizer']
            model = self.translators[lang_code]['model']
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            translated = model.generate(**inputs)
            translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
            return translated_text
        except Exception as e:
            messagebox.showerror("Error", f"Failed to translate to {self.languages[lang_code]['name']}: {str(e)}")
            return ''

    def create_audio(self, text, lang_code):
        if not text.strip():
            return None
        try:
            # Translate text to the target language
            translated_text = self.translate_text(text, lang_code)
            if not translated_text:
                return None
            # Create language-specific directory
            output_dir = f"audio_{lang_code}"
            os.makedirs(output_dir, exist_ok=True)
            # Generate unique movie_id using timestamp
            movie_id = f"gui_{int(time.time())}"
            output_file = os.path.join(output_dir, f"{movie_id}_{lang_code}.mp3")
            # Create audio using gTTS
            tts = gTTS(text=translated_text, lang=self.languages[lang_code]['gtts_lang'])
            tts.save(output_file)
            return output_file
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create audio: {str(e)}")
            return None

    def play_audio(self, audio_file):
        try:
            playsound.playsound(audio_file)
            os.remove(audio_file)  # Clean up audio file
        except Exception as e:
            messagebox.showerror("Error", f"Failed to play audio: {str(e)}")

    def predict_genres(self, text):
        try:
            # Preprocess and vectorize input summary
            cleaned_text = self.preprocess_summary(text)
            input_vector = self.vectorizer.transform([cleaned_text])
            # Compute cosine similarity with dataset summaries
            similarities = cosine_similarity(input_vector, self.summary_vectors)
            # Find the most similar summary
            max_sim_idx = np.argmax(similarities[0])
            # Retrieve genres
            genres = self.df.iloc[max_sim_idx]['genres']
            return genres if genres else ["No genres found"]
        except Exception as e:
            messagebox.showerror("Error", f"Failed to predict genres: {str(e)}")
            return []

    def update_output(self, message):
        self.output_text.config(state="normal")
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, message)
        self.output_text.config(state="disabled")

    def clear_summary(self):
        self.summary_text.delete("1.0", tk.END)
        self.update_output("Summary cleared.")

    def execute_action(self, action):
        if action == "exit":
            self.root.quit()
            return

        summary = self.summary_text.get("1.0", tk.END).strip()
        if not summary:
            messagebox.showwarning("Warning", "Please enter a movie summary.")
            return

        if action == "audio":
            language_name = self.language_var.get()
            lang_code = next((code for code, info in self.languages.items() if info['name'] == language_name), 'ar')
            audio_file = self.create_audio(summary, lang_code)
            if audio_file:
                self.update_output(f"Audio generated and playing in {language_name}...")
                self.play_audio(audio_file)
                self.update_output(f"Audio played successfully in {language_name}.")
        elif action == "genre":
            genres = self.predict_genres(summary)
            self.update_output("Predicted Genres:\n" + "\n".join(genres))

if __name__ == "__main__":
    root = tk.Tk()
    app = MovieGUI(root)
    root.mainloop()