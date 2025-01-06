# WhatsApp Template Prediction

## Overview
The **WhatsApp Template Prediction** project is an AI-powered chatbot designed to assist businesses in creating Meta-compliant WhatsApp message templates. It categorizes templates into Utility, Authentication, and Marketing, providing optimized suggestions to reduce rejection rates and streamline the approval process. 

---

## Features
- Interactive chatbot interface for requesting and downloading templates.
- AI-powered prediction engine for generating templates based on user input.
- Pre-trained models and encoders ensure accurate and compliant templates.
- Easy-to-use interface for business users to streamline communication with clients.

---

## How to Run the Application

### Prerequisites
1. Python installed (>= 3.8 recommended).
2. Install the required dependencies from the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure the following files are in the project directory:
   - `my_neural_network_model.h5` (Trained model file)
   - `tfidf_vectorizer.pkl` (TF-IDF vectorizer file)
   - `label_encoder.pkl` (Label encoder file)

### Running the Application
1. Navigate to the project directory in your terminal.
2. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```
3. Open the application in your browser. By default, it will run at `http://localhost:8501`.

---

## Explanation of Major Code Components

### 1. **Model Loading and Caching**
   ```python
   @st.cache_resource
   def load_neural_network_model():
       model = load_model('my_neural_network_model.h5')
       return model

   @st.cache_resource
   def load_vectorizer():
       vectorizer = joblib.load('tfidf_vectorizer.pkl')
       return vectorizer

   @st.cache_resource
   def load_label_encoder():
       label_encoder = joblib.load('label_encoder.pkl')
       return label_encoder
   ```
   - These functions load the neural network model, TF-IDF vectorizer, and label encoder.
   - Streamlit's `@st.cache_resource` ensures efficient reloading and prevents unnecessary computations.

### 2. **Prediction Logic**
   ```python
   def main(new_input):
       new_input_vectorized = vectorizer.transform([new_input]).toarray()
       new_input_pred_prob = model.predict(new_input_vectorized)
       new_input_pred_class = argmax(new_input_pred_prob, axis=1).numpy()
       predicted_json_output = label_encoder.inverse_transform(new_input_pred_class)
       return predicted_json_output[0]
   ```
   - Accepts user input, vectorizes it using the TF-IDF model, and predicts the template category using the neural network.
   - Decodes the prediction to return a JSON-formatted template.

### 3. **Chatbot Interaction**
   ```python
   def chatbot(new_input):
       output = main(new_input)
       st.session_state.output = output
       data1 = json.loads(output)
       title = data1.get('name')
       st.markdown(f" :blue[**TITLE**] : :green-background[{title}] ")
       st.code(output)
       st.download_button(label="Download Template...",
                          file_name="template.json",
                          data=output,
                          mime="json",
                          use_container_width=True,
                          help="Download the above template in JSON format")
   ```
   - Handles user input, processes predictions, and displays the output in a user-friendly format.
   - Includes a download button to save the template as a JSON file.

### 4. **Streamlit App Initialization**
   ```python
   if __name__ == "__main__":
       st.title("TEMPLATE SELECTION MODEL :smile:")
       st.markdown("#### <span style='color: blue;'>**VINITH KUMAR **</span>", unsafe_allow_html=True)
       new_input = st.chat_input("Enter the template you want, more than 25 characters or 5 words...")
       app_run()
   ```
   - Sets up the Streamlit application with a title and user prompt.
   - Invokes the main application logic via `app_run()`.

---

## Output
1. User enters a message template idea via the chatbot.
2. The chatbot processes the input and generates a JSON-formatted WhatsApp template.
3. Users can view, edit, or download the template for Meta submission.

---

## Benefits
- **Efficiency:** Reduces time spent on designing compliant templates.
- **Accuracy:** Minimizes rejection rates by providing pre-tested, optimized templates.
- **User-Friendly:** Intuitive chatbot interface for ease of use.

For questions or assistance, contact **Vinith Kumar**.
