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
## Chatbot Input Reference

Here is a list of available template names for chatbot inputs:

- login_alert
- new_features_update
- exclusive_offer_just_for_you
- promotional_offer_template
- Event Invitation
- Newsletter Confirmation
- Feedback Request Template
- Seasonal Sale Announcement Template
- Product Launch Notification Template
- Loyalty Program Update Template
- Cart Abandonment Reminder Template
- Special Birthday Offer Template
- Flash Sale Notification Template
- Membership Renewal Reminder Template
- Survey Participation Invitation Template
- Referral Program Announcement Template
- New Store Opening Announcement Template
- Win a Prize Contest Template
- Customer Appreciation Message Template
- Seasonal Product Reminder Template
- Flashback Friday Offer Template
- Exclusive VIP Access Template
- Limited-Time Bundle Offer Template
- Product Review Request Template
- Anniversary Sale Reminder Template
- End of Season Clearance Template
- New Collection Launch Template
- Flash Sale Reminder Template
- Subscription Template
- Customer Feedback Request Template
- Special Holiday Offer Template
- Product Restock Notification Template
- Event Invitation Template
- Free Shipping Offer Template
- back_to_school_promotion_template
- birthday_discount_offer
- E-Commerce: New Product Launch template
- Software Development: Project Completion Notification template
- Healthcare: Appointment Reminder Template
- Travel: Booking Confirmation Template
- Real Estate: Property Listing Alert Template
- Fitness: Membership Renewal Reminder
- Real Estate: Property Listing Alert
- Fitness: Membership Renewal Reminder
- Food Delivery: Order Confirmation
- Course Enrollment Confirmation
- Service Reminder
- Logistics: Shipment Tracking Update
- Finance: Loan Approval Notification
- Hospitality: Reservation Confirmation
- Appointment Confirmation
- SaaS: Subscription Renewal Reminder
- E-commerce: Account Login Authentication
- Two-Factor Authentication (2FA)
- Secure Account Access Code
- Payment Verification Code
- Password Reset Authentication
- Loyalty Program Authentication
- Secure Booking Authentication
- Policy Update Authentication
- Secure Exam Authentication
- Property Viewing Authentication
- Secure Document Access Authentication
- SIM Activation Authentication
- Vehicle Service Authentication
- Employee Portal Access Authentication
- Secure Class Booking Authentication
- Loan Application Authentication
- Appointment Verification Authentication
- Property Rental Authentication
- Secure Reservation Authentication
- Service Activation Authentication
- Order Delivery Update
- Appointment Reminder
- Bill Payment Reminder
- Flight Status Update
- Rent Payment Reminder
- Data Usage Alert
- Class Schedule Reminder
- Package Out for Delivery
- Service Appointment Reminder
- Subscription Renewal Reminder
- Retail: In-Store Pickup Reminder
- Hospitality: Hotel Check-in Reminder
- Utilities: Scheduled Power Maintenance Notice
- Fitness: Class Booking Confirmation
- Transportation: Ride Confirmation
- Vehicle Service Authentication
- Food Delivery: Order Confirmation
- Healthcare: Prescription Refill Reminder
- E-Learning: Course Progress Reminder
- Transportation: Vehicle Pickup Notification
- Subscription Service: Free Trial Expiry Notification
- Banking: Account Balance Alert
- Telecom: Plan Expiry Alert
- Real Estate: Rent Due Reminder
- Event Management: Event Reminder
- E-commerce: New Product Launch Announcement
- Beauty & Wellness: Limited-Time Offer
- Fashion: Seasonal Collection Launch
- Fitness: Membership Promotion
- Automotive: Service Promotion
- Travel: Holiday Package Promotion
- Real Estate: Property Open House Invitation
- Software: Free Trial Offer
- Healthcare: Wellness Program Enrollment
- Telecom: Upgrade to a New Plan
- Beauty & Wellness: Limited-Time Offer
- E-commerce: Abandoned Cart Reminder
- Restaurant: Special Weekend Menu
- Healthcare: Health Checkup Promo
- Education: Webinar Registration
- Retail: Flash Sale Alert
- Fitness: Free Fitness Class Invite
- Hospitality: Hotel Booking Discount
- Automotive: New Car Launch Invite
- Software: Product Feature Update
- Education: Course Enrollment Reminder
- Travel Agency: Exclusive Travel Deal
- GYM: Membership Renewal Reminder Template


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
# Contact Information

For any inquiries or support, feel free to reach out:

- **LinkedIn:** [VINITH KUMAR](https://www.linkedin.com/in/vinith-kumar-m)
- **Email:** [vinithkumar0720@gmail.com.com](mailto:vinithkumar0720@gmail.com)

