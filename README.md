# IMDB Movie Review Sentiment Analysis

## Overview
This repository contains a Streamlit web application that performs sentiment analysis on movie reviews. The app uses a Simple RNN (Recurrent Neural Network) model trained on the IMDB dataset to classify movie reviews as either positive or negative.

## Live Demo
Try the app live at: [IMDB Sentiment Analysis App](https://rnn-movie-review-analyzer-3gx9p4hi9qpyvdrrif3uqv.streamlit.app/)

## Features
- Interactive web interface for entering movie reviews
- Real-time sentiment prediction (positive/negative)
- Confidence score visualization with progress bar
- Responsive design that works on both desktop and mobile

## Model Architecture
The model uses a Simple RNN architecture trained on the IMDB dataset, which consists of 50,000 movie reviews labeled as positive or negative. The model processes sequences of words and learns to recognize patterns that indicate sentiment.

## Tech Stack
- **Python**: Primary programming language
- **TensorFlow/Keras**: Deep learning framework for model training and inference
- **Streamlit**: Web application framework for the user interface
- **IMDB Dataset**: Used for training the sentiment analysis model

## Installation and Local Setup
1. Clone this repository:
```bash
git clone https://github.com/yourusername/rnn-movie-review-analyzer.git
cd rnn-movie-review-analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run main.py
```

4. Open your browser and navigate to `http://localhost:8501`

## File Structure
- `main.py`: Main Streamlit application code
- `simple_rnn_imdb.h5`: Pre-trained sentiment analysis model
- `requirements.txt`: Dependencies for the project
- `README.md`: Project documentation

## Model Training
The model was trained on the IMDB dataset using a Simple RNN architecture. Training details include:
- 25,000 training samples
- Word embeddings with 100 dimensions
- Simple RNN layer with 32 units
- Binary cross-entropy loss function
- Adam optimizer

## Future Improvements
- Add support for multiple languages
- Implement more advanced model architectures (LSTM, Transformer)
- Include explainability features to highlight influential words
- Add comparative analysis with other sentiment analysis models

## Troubleshooting
If you encounter any issues with model loading, ensure you have the correct TensorFlow version installed. The model was saved with TensorFlow 2.x and includes a custom SimpleRNN implementation to handle compatibility issues.

## License
MIT License

## Contact
For questions or suggestions, please open an issue on this repository.

---

*Note: This application is for educational purposes and demonstrates the use of RNNs for natural language processing tasks.*
