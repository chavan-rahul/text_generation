import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load the text generation model
model = load_model('text_generation_model.h5')


# Text generation function
def generate_text(seed_text, max_words=50):
    generated_text = seed_text
    for _ in range(max_words):
        # Convert input text to sequence of token indices
        input_sequence = tokenizer.texts_to_sequences([seed_text])[0]
        input_sequence = pad_sequences([input_sequence], maxlen=MAX_SEQUENCE_LENGTH, padding='pre')

        # Predict the next token index using the model
        predicted_index = model.predict_classes(input_sequence, verbose=0)

        # Convert token index to word and append to generated text
        predicted_word = tokenizer.index_word[predicted_index[0]]
        generated_text += ' ' + predicted_word

        # Update the seed_text for the next iteration
        seed_text = seed_text + ' ' + predicted_word

    return generated_text


# Streamlit app
def main():
    st.title('Text Generation with Streamlit')
    seed_text = st.text_input('Enter your seed text:')
    num_words = st.number_input('Number of words to generate:', min_value=10, max_value=200, value=50, step=10)

    if st.button('Generate Text'):
        generated_text = generate_text(seed_text, num_words)
        st.write('Generated Text:')
        st.write(generated_text)


if __name__ == '__main__':
    main()
