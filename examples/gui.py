"""Tkinter-based app for text sentiment analysis."""

import tkinter as tk
from collections.abc import Sequence
from typing import cast

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from src.pipeline.predict import ModelNotReadyError, TextSentimentClassifier


class SentimentAnalysisApp:
    """A simple Tkinter-based app for text sentiment analysis."""

    def __init__(self, root: tk.Tk) -> None:
        """Initialize the SentimentAnalysisApp.

        Args:
            root: The Tkinter root window.
        """
        self.root = root
        self.root.title("Text Sentiment Classifier for Sentiment Analysis")
        # tkinter stubs leave iconbitmap untyped
        self.root.iconbitmap("img/icon.ico")  # type: ignore[no-untyped-call]

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

    def analyze_sentiment(self) -> None:
        """Analyze the sentiment of the entered text."""
        user_text = self.text_widget.get("1.0", "end-1c")
        clean_text = self.classifier.preprocess_text(user_text)
        try:
            result_prob = self.classifier.classify_sentiment(
                clean_text, return_probabilities=True
            )
            result = self.classifier.classify_sentiment(clean_text)
        except ModelNotReadyError as exc:
            self.result_label.config(text=f"Error: {exc}")
            self.ax.clear()
            self.canvas.draw()
            return
        self.result_label.config(text=f"Sentiment: {result}")

        # Update the radar chart
        # classify_sentiment(..., return_probabilities=True) returns list[float]
        probs = cast(list[float], result_prob)
        self.update_radar_chart(probs)

    def update_radar_chart(self, probabilities: Sequence[float]) -> None:
        """Update the radar chart based on sentiment probabilities.

        Args:
            probabilities: Sentiment probabilities.
        """
        self.ax.clear()

        probs_arr = np.asarray(probabilities)
        categories = ["Negative", "Neutral", "Positive"]
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        self.ax.set_theta_offset(np.pi / 2)
        self.ax.set_theta_direction(-1)
        self.ax.set_rlabel_position(0)

        # Find the index of the maximum probability
        max_index = int(np.argmax(probs_arr))

        # Define colors based on max probability
        colors = ["red", "yellow", "green"]

        self.ax.plot(
            angles,
            probs_arr.tolist() + probs_arr.tolist()[:1],
            "black",
            linewidth=1,
            linestyle="solid",
        )
        self.ax.fill(
            angles,
            probs_arr.tolist() + probs_arr.tolist()[:1],
            alpha=0.5,
            color=colors[max_index],
        )

        self.ax.set_xticks(angles[:-1])
        self.ax.set_xticklabels(categories)
        self.ax.set_yticks([0.2, 0.4, 0.6, 0.8])

        self.canvas.draw()


def main() -> None:
    """Main function to run the sentiment analysis app."""
    root = tk.Tk()
    SentimentAnalysisApp(root)
    root.geometry("800x600")  # Set the initial window size
    root.mainloop()


if __name__ == "__main__":
    main()
