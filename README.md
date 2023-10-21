# ContextLens

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/downloads/release/python-31012/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter%20Notebook-6.5-orange)](https://jupyter.org/install)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68.1-red)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-20.10.8-blue)](https://www.docker.com/)

[![Logo][product-screenshot]](https://github.com/KarthikUdyawar/ContextLens)

[product-screenshot]: img\logo.png

Reveal Emotions with Context

[**Explore the docs »**](https://github.com/KarthikUdyawar/ContextLens)

[View Demo](https://github.com/KarthikUdyawar/ContextLens)
·
[Report Bug](https://github.com/KarthikUdyawar/ContextLens/issues)
·
[Request Feature](https://github.com/KarthikUdyawar/c/pulls)

</div>

## Table of Contents

- [About The Project](#about-the-project)
  - [Built With](#built-with)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [GUI Usage](#gui-usage)
  - [API Usage](#api-usage)
- [Docker Image](#docker-image)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## About The Project

[![Product Name Screen Shot][product-screenshot-1]](https://github.com/KarthikUdyawar/ContextLens)

[product-screenshot-1]: img/meter.png

ContextLens is a comprehensive project designed to empower users with the ability to analyze and understand sentiment in textual data. In the age of vast information exchange, the project leverages diverse datasets, including Reddit and Twitter data, to create a consolidated and meticulously cleaned dataset that serves as the foundation for sentiment analysis.

The project is structured around the following key steps, each aligned with the respective project notebooks:

**Data Collection and Preparation:** (Notebook 01_get_data) The project commences with data collection, where we gather data from various sources such as Reddit and Twitter. The data is then meticulously processed, cleaned, and consolidated into a uniform format, setting the stage for subsequent analysis.

**Text Data Preprocessing:** (Notebook 02_clean_text) To ensure the text is suitable for analysis and modeling, text data preprocessing is essential. This step involves handling HTML entities, removing special characters, and converting emojis, among other tasks, to prepare the text data for further analysis.

**Exploratory Data Analysis (EDA):** (Notebook 03_eda) EDA is a critical step in understanding the dataset comprehensively. We delve into various aspects, including text sentiment, text length distribution, and word frequencies. This exploratory phase equips us with insights and readies the data for subsequent tasks such as model building and feature engineering.

**Sentiment Analysis and BERT Model Exploration:** (Notebook 04_sentiment_BERT) This notebook marks a pivotal phase in our journey. We explore sentiment analysis, a crucial component of Natural Language Processing (NLP) that focuses on deciphering the emotional tone of textual data. Additionally, we delve into BERT (Bidirectional Encoder Representations from Transformers), an influential language model renowned for its exceptional performance in a variety of language understanding tasks.

With ContextLens, users can harness the power of sentiment analysis and cutting-edge NLP techniques, including BERT, to gain invaluable insights from textual data. The project offers a step-by-step guide, complete with code implementations and explanations in each notebook, enabling users to not only understand sentiment analysis but also adapt the ContextLens project to suit their specific needs. Whether it's for business intelligence, social media monitoring, or sentiment research, ContextLens is your gateway to understanding the sentiments embedded in textual information.

### Built With

This project was developed using the following technologies, libraries, and frameworks:

- [Python](https://www.python.org/): A versatile and widely-used programming language that forms the foundation of the project.
- [Jupyter Notebook](https://jupyter.org/): An interactive development environment for data analysis and experimentation, facilitating the creation of code, visualizations, and narrative documentation.
- [Docker](https://www.docker.com/): A containerization platform that ensures consistency and portability of the project's execution environment, making deployment and scaling more efficient.

The project is enriched with the capabilities of several Python libraries, including:

- [Pandas](https://pandas.pydata.org/): A powerful data manipulation and analysis library, essential for handling and processing datasets.
- [NumPy](https://numpy.org/): A fundamental library for numerical computations, providing support for arrays and mathematical operations.
- [PyTorch](https://pytorch.org/): A powerful deep learning framework that plays a pivotal role in building and training machine learning models, particularly neural networks.
- [Scikit-learn](https://scikit-learn.org/): A versatile machine learning library, offering various algorithms and evaluation metrics for model development and analysis.
- [Matplotlib](https://matplotlib.org/): A comprehensive data visualization library, enabling the creation of informative charts and graphs.
- [Seaborn](https://seaborn.pydata.org/): A high-level interface to Matplotlib, enhancing data visualization with a more attractive and informative style.
- [TextBlob](https://textblob.readthedocs.io/en/dev/): A library for processing textual data, including sentiment analysis, part-of-speech tagging, and more.
- [Transformers](https://huggingface.co/transformers/): A library by Hugging Face that provides access to pre-trained natural language processing models, including BERT.
- [FastAPI](https://fastapi.tiangolo.com/): A modern web framework for building APIs with Python, enabling efficient and robust integration of machine learning models.

These technologies, libraries, and frameworks collectively empower "ContextLens" to perform sentiment analysis and handle text data effectively, offering a wide range of features and capabilities for data collection, preprocessing, analysis, and model development.

## Getting Started

This section provides instructions on how to set up the project locally. Follow the steps below to get a local copy up and running.

### Prerequisites

Before proceeding with the installation, ensure that you have the following prerequisites installed:

- Python (version 3.10.6)
- Jupyter Notebook (version 6.5.2)
- Docker (version 20.10.8)

You can check the versions of Python and Jupyter Notebook by running the following commands in the terminal:

```bash
python --version

jupyter notebook --version

docker --version
```

### Installation

Follow the steps below to install and set up the project:

1. **Clone the repository:**

   Clone the project repository using the following command:

   ```bash
   git clone https://github.com/KarthikUdyawar/ContextLens.git
   ```

2. **Create a virtual environment and activate it:**

   To isolate project dependencies, create a virtual environment and activate it:

   ```bash
   python3 -m venv env
   source env/bin/activate   # On Windows: env\Scripts\activate
   ```

3. **Install package**

   Install the package using the following command:

   ```bash
   pip install .
   ```

## Usage

You can use the following code snippet as an example to understand how to use the project:

```python
"""Text Sentiment Classifier for Sentiment Analysis"""

from src.pipeline.predict import TextSentimentClassifier


def main():
    """Main function for the interactive program."""
    MODEL_FILE_PATH = "src/model/0.2v/model.pth"
    classifier = TextSentimentClassifier(MODEL_FILE_PATH)
    while True:
        print("Text Sentiment Classifier for Sentiment Analysis")
        user_text = str(input("> "))
        print(f"User Text: {user_text}", end="\n\n")

        clean_text = classifier.preprocess_text(user_text)
        print(f"Cleaned Text: {clean_text}", end="\n\n")

        result = classifier.classify_sentiment(clean_text)
        print(f"Sentiment: {result}", end="\n\n")

        result_prob = classifier.classify_sentiment(
            clean_text, return_probabilities=True
        )
        print(f"Sentiment probability: {result_prob}", end="\n\n")

        choice = input("Try again (Y/n): ")
        if choice.lower() == "n":
            break


if __name__ == "__main__":
    main()
```

The provided code snippet offers an interactive way for users to input text and receive sentiment analysis results. Users can interact with the program by entering text and observing the sentiment classification and associated probabilities. You can customize this code to fit your specific project requirements and user interaction preferences.

The code snippet concludes with a reference to additional examples available in the provided GitHub repository for more in-depth usage scenarios.

_For more examples, please refer to the [Code](https://github.com/KarthikUdyawar/ContextLens)_

### GUI Usage

[![GUI Screen Shot][gui-screenshot-1]](https://github.com/KarthikUdyawar/ContextLens)

[gui-screenshot-1]: img/screenshot.png

This Python script provides a graphical user interface (GUI) application for text sentiment analysis using the Tkinter library. The application allows you to enter text, analyze its sentiment, and visualize the results on a radar chart.

### API Usage

The project now includes an API powered by [FastAPI](https://fastapi.tiangolo.com/), which allows you to interact with nlp-related functionalities programmatically. To use the API, follow these steps:

1. **Run the FastAPI server:**

   Navigate to the project directory and run the following command:

   ```bash
   uvicorn src.app.main:app --port 8000
   ```

   This will start the FastAPI server, making the API endpoints accessible at [http://localhost:8000](http://localhost:8000).

2. **Access the API documentation:**

   Open your web browser and go to [http://localhost:8000/docs](http://localhost:8000/docs) to access the Swagger documentation for the API. Here, you can explore the available endpoints, view request and response schemas, and interact with the API using the built-in interface.

3. **API Endpoints:**

   - `POST /predict:` Endpoint for predicting the sentiment of a given text.

   - `POST /predict-prob:` Endpoint for predicting the sentiment probabilities of a given text.

   - `POST /clean:` Endpoint for cleaning a given text.

   _For more details, please refer to the [API documentation](http://localhost:8000/docs)._

## Docker Image

A Docker image for the contextlens API is available on [Docker Hub](https://hub.docker.com/r/kstar123/contextlens). You can pull and run the image using the following command:

```bash
docker pull kstar123/contextlens:1.0

docker run -d -p 8000:8000 --name contextlens-api kstar123/contextlens:1.0
```

This will start the FastAPI server inside a Docker container, and you can access the API endpoints at [http://localhost:8000](http://localhost:8000).

_For more details on using Docker, refer to the [Docker documentation](https://docs.docker.com/)._

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request against the `develop` branch. You can also simply open an issue with the tag "enhancement". Don't forget to give the project a star! Thanks again!

1. **Open an Issue:** Start by opening an issue to discuss your proposed changes or enhancements.
2. **Fork the Project:** Create your own fork of the project repository.
3. **Create a Feature Branch:** Create a feature branch in your fork (`git checkout -b feature/AmazingFeature`).
4. **Commit your Changes:** Make your desired changes and commit them (`git commit -m 'Add some AmazingFeature'`).
5. **Push to the Branch:** Push your changes to the feature branch (`git push origin feature/AmazingFeature`).
6. **Open a Pull Request:** Create a pull request against the `develop` branch of the original repository.

## License

Distributed under the MIT License. See [`LICENSE.txt`](https://github.com/KarthikUdyawar/contextlens/blob/master/LICENSE) for more information.

## Contact

If you have any questions, suggestions, or feedback about Contextlens, feel free to reach out to us:

- **Project Author:** [Karthik Udyawar](mailto:karthikajitudy@gmail.com)
- **GitHub Repo:** [Contextlens](https://github.com/KarthikUdyawar/Contextlens)
- **Kaggle:** [Karthik Udyawar](https://www.kaggle.com/karthikudyawar)
- **Docker Hub** [kstar123](https://hub.docker.com/r/kstar123/contextlens)

We are open to collaboration and appreciate any contributions to the project. If you encounter any issues or have ideas for enhancements, please don't hesitate to create an issue or pull request on the GitHub repository.

We value your input and look forward to hearing from you!
