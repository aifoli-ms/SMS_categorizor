import streamlit as st
import pandas as pd
from categorization import SMSCategorizer

st.title('SMS Categorization App')

st.write('Upload a CSV file containing SMS messages. The app will categorize each message using the SMSCategorizer model.')

uploaded_file = st.file_uploader('Choose a CSV file', type='csv')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write('Preview of uploaded data:')
    st.dataframe(df.head())

    # Let user select the message column
    message_column = st.selectbox('Select the column containing the messages:', df.columns)

    # Initialize categorizer
    categorizer = SMSCategorizer()

    # Categorize messages
    with st.spinner('Categorizing messages...'):
        df['predicted_category'] = df[message_column].apply(lambda x: categorizer.pattern_based_categorization(categorizer.preprocess_text(x)))

    st.success('Categorization complete!')
    st.write('Results:')
    st.dataframe(df[[message_column, 'predicted_category']])

    # Option to download results
    csv = df[[message_column, 'predicted_category']].to_csv(index=False).encode('utf-8')
    st.download_button('Download categorized results as CSV', data=csv, file_name='categorized_sms.csv', mime='text/csv') 