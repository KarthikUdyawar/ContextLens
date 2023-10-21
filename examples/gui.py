"""Tkinter-based app for text sentiment analysis"""
import tkinter as tk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from src.pipeline.predict import TextSentimentClassifier


class SentimentAnalysisApp:
    """A simple Tkinter-based app for text sentiment analysis."""

    def __init__(self, root):
        """
        Initialize the SentimentAnalysisApp.

        Args:
            root: The Tkinter root window.
        """
        self.root = root
        self.root.title("Text Sentiment Classifier for Sentiment Analysis")
        self.root.iconbitmap("img/icon.ico")  # Use the path to your icon file

        # Create a frame to hold the widgets
        self.frame = tk.Frame(root)
        self.frame.pack(expand=True, fill="both")

        self.label = tk.Label(self.frame, text="Enter text:")
        self.label.grid(row=0, column=0, padx=(10, 0), pady=(10, 0), sticky="w")

        self.text_widget = tk.Text(self.frame, wrap="word", width=50, height=10)
        self.text_widget.grid(
            row=1, column=0, padx=(10, 10), pady=(0, 10), sticky="nsew"
        )

        self.classifier = TextSentimentClassifier("src/model/0.2v/model.pth")

        self.result_label = tk.Label(self.frame, text="")
        self.result_label.grid(row=2, column=0, padx=(10, 0), pady=(0, 10), sticky="w")

        # Analyze button moved to center
        self.analyze_button = tk.Button(
            self.frame, text="Analyze", command=self.analyze_sentiment
        )
        self.analyze_button.grid(
            row=3, column=0, padx=(0, 0), pady=(0, 10), sticky="nsew", columnspan=2
        )  # Set columnspan to 2

        self.frame.grid_rowconfigure(1, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)

        # Matplotlib radar chart
        self.figure = Figure(figsize=(4, 5))
        self.ax = self.figure.add_subplot(111, polar=True)

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True)

    def analyze_sentiment(self):
        """
        Analyze the sentiment of the entered text.
        """
        user_text = self.text_widget.get("1.0", "end-1c")
        clean_text = self.classifier.preprocess_text(user_text)
        result_prob = self.classifier.classify_sentiment(
            clean_text, return_probabilities=True
        )
        result = self.classifier.classify_sentiment(clean_text)
        self.result_label.config(text=f"Sentiment: {result}")

        # Update the radar chart
        self.update_radar_chart(result_prob)

    def update_radar_chart(self, probabilities):
        """Update the radar chart based on sentiment probabilities.

        Args:
            probabilities: Sentiment probabilities.
        """
        self.ax.clear()

        categories = ["Negative", "Neutral", "Positive"]
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        self.ax.set_theta_offset(np.pi / 2)
        self.ax.set_theta_direction(-1)
        self.ax.set_rlabel_position(0)

        # Find the index of the maximum probability
        max_index = np.argmax(probabilities)

        # Define colors based on max probability
        colors = ["red", "yellow", "green"]

        self.ax.plot(
            angles,
            probabilities.tolist() + probabilities.tolist()[:1],
            "black",
            linewidth=1,
            linestyle="solid",
        )
        self.ax.fill(
            angles,
            probabilities.tolist() + probabilities.tolist()[:1],
            alpha=0.5,
            color=colors[max_index],
        )

        self.ax.set_xticks(angles[:-1])
        self.ax.set_xticklabels(categories)
        self.ax.set_yticks([0.2, 0.4, 0.6, 0.8])

        self.canvas.draw()


def main():
    """
    Main function to run the sentiment analysis app.
    """
    root = tk.Tk()
    SentimentAnalysisApp(root)
    root.geometry("800x600")  # Set the initial window size
    root.mainloop()


if __name__ == "__main__":
    main()
