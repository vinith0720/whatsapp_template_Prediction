import os
import logging
import joblib
import json
import streamlit as st

# Suppress TensorFlow's INFO and WARNING messages

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # '2' to suppress INFO and WARNING, '3' to suppress ERROR

# Disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


from tensorflow import argmax

# Configure Python logging to suppress TensorFlow warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)
from tensorflow.keras.models import load_model # type: ignore



st.set_page_config(
    page_title="whatsapp_Template_Prediction",
    page_icon="🖥️",
    # layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "VINITH KUMAR M" 
    }
    )
# Cache the model loading function
@st.cache_resource
def load_neural_network_model():
    model = load_model('my_neural_network_model.h5')
    return model

# Cache the vectorizer loading function
@st.cache_resource
def load_vectorizer():
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return vectorizer

# Cache the label encoder loading function
@st.cache_resource
def load_label_encoder():
    label_encoder = joblib.load('label_encoder.pkl')
    return label_encoder

# Load the resources from cache
model = load_neural_network_model()
vectorizer = load_vectorizer()
label_encoder = load_label_encoder()

# Input from the user

def main(new_input):
    
    new_input_vectorized = vectorizer.transform([new_input]).toarray()
    # Make prediction
    new_input_pred_prob = model.predict(new_input_vectorized)
    new_input_pred_class = argmax(new_input_pred_prob, axis=1).numpy()
    # Use your label encoder to decode the prediction
    predicted_json_output = label_encoder.inverse_transform(new_input_pred_class)
    return predicted_json_output[0]




def chatbot(new_input):
    output = main(new_input)
    st.session_state.output = output
    data1 = json.loads(output)
    title =data1.get('name')
    
    st.markdown(f" :blue[**TITLE**] : :green-background[{title}] ")
    st.code(output)
    st.download_button(label="Download Template...",
                       file_name="template.json",
                       data=output,type="primary",on_click=st.balloons,
                       mime="json",use_container_width=True,help="download the above template in json format")

    # st.button(on_click=edit_the_json,label="Edit the template(JSON)",use_container_width=True,type="primary")



def app_run():
    if new_input and len(new_input) >= 20:
        chatbot(new_input)

    elif new_input and len(new_input) < 20:
    # st.markdown(f" :blue[**YOU**] : :green-background[type something related ..] ")
        st.markdown(" Please provide a valid prompt more than 25 character or  5 words...")
    else:
        st.write("I am a chatbot to provide a template for your WhatsApp campaign and redesign your template as you wish.")


if __name__ =="__main__":
    st.title("TEMPLATE SELECTION MODEL :smile:")
    st.markdown("#### <span style='color: blue;'>**VINITH KUMAR **</span>", unsafe_allow_html=True)
    new_input =st.chat_input("enter the template  you want more than 25 character or  5 words...")
    app_run()