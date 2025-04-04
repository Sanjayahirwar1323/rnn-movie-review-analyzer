# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Create a custom SimpleRNN class that ignores the 'time_major' parameter
class CustomSimpleRNN(tf.keras.layers.SimpleRNN):
    def __init__(self, *args, **kwargs):
        # Remove the problematic parameter if present
        if 'time_major' in kwargs:
            kwargs.pop('time_major')
        super().__init__(*args, **kwargs)

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with custom objects
model = load_model('simple_rnn_imdb.h5', custom_objects={'SimpleRNN': CustomSimpleRNN})

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Step 3: Streamlit App
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# User input
user_input = st.text_area('Movie Review', height=150)

if st.button('Classify'):
    if user_input:
        try:
            preprocessed_input = preprocess_text(user_input)
            # Make prediction
            prediction = model.predict(preprocessed_input)
            sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
            
            # Display the result with styling
            if sentiment == 'Positive':
                st.success(f'Sentiment: {sentiment}')
            else:
                st.error(f'Sentiment: {sentiment}')
                
            # Create a progress bar for visualization
            st.write(f'Confidence Score: {prediction[0][0]:.4f}')
            st.progress(float(prediction[0][0]))
            
            # Add some context about the prediction
            confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
            if confidence > 0.9:
                st.write("The model is very confident about this prediction.")
            elif confidence > 0.7:
                st.write("The model is reasonably confident about this prediction.")
            else:
                st.write("The model is less certain about this prediction.")
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
    else:
        st.warning('Please enter a movie review.')