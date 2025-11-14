import streamlit as st
import pandas as pd
from main import SMSCategorizer  # This correctly imports from your main.py
import plotly.express as px
from datetime import datetime
import io

# Configure page
st.set_page_config(
    page_title="SMS Categorization App",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metrics-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
    .stAlert > div {
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">📱 SMS Categorization App</h1>', unsafe_allow_html=True)

# Sidebar for app info and settings
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This app categorizes SMS messages using pattern-based analysis.
    
    **Features:**
    - Support for CSV and Excel files
    - Automatic text column detection
    - Interactive visualizations
    - Downloadable results
    """)
    
    st.header("📊 Settings")
    show_preview_rows = st.slider("Preview rows to display", 5, 20, 10)
    chart_theme = st.selectbox("Chart theme", ["plotly", "plotly_white", "plotly_dark"])

# Main content
st.write("""
Upload a CSV or Excel file containing SMS messages. The app will categorize each message 
using the SMSCategorizer model and provide detailed visualizations of the results.
""")

# File uploader
uploaded_file = st.file_uploader(
    'Choose a CSV or Excel file',
    type=['csv', 'xlsx', 'xls'],
    help="Supported formats: CSV, Excel (.xlsx, .xls)"
)

if uploaded_file is not None:
    try:
        # File type detection and reading
        file_details = {
            "filename": uploaded_file.name,
            "filetype": uploaded_file.type,
            "filesize": uploaded_file.size
        }
        
        with st.expander("📁 File Details"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Filename", file_details["filename"])
            with col2:
                st.metric("File Type", file_details["filetype"])
            with col3:
                st.metric("File Size", f"{file_details['filesize']} bytes")
        
        # Read file based on extension
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error('❌ Unsupported file type!')
            st.stop()
        
        st.success(f"✅ Successfully loaded {len(df)} rows and {len(df.columns)} columns")
        
        # Preview of uploaded data
        st.subheader('📋 Data Preview')
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.dataframe(df.head(show_preview_rows), use_container_width=True)
        with col2:
            st.metric("Total Rows", len(df))
            st.metric("Total Columns", len(df.columns))
            if df.isnull().sum().sum() > 0:
                st.metric("Missing Values", df.isnull().sum().sum())
        
        # Column selection with smart detection
        st.subheader('🎯 Column Selection')
        
        text_col_candidates = [col for col in df.columns if 'text' in col.lower() or 'message' in col.lower() or 'sms' in col.lower()]
        
        if text_col_candidates:
            default_column = text_col_candidates[0]
            st.info(f"🎯 Auto-detected potential text column: '{default_column}'")
        else:
            default_column = df.columns[0]
        
        message_column = st.selectbox(
            'Select the column containing the SMS messages:',
            df.columns,
            index=list(df.columns).index(default_column) if default_column in df.columns else 0
        )
        
        # Data validation
        if df[message_column].isnull().sum() > 0:
            st.warning(f"⚠️ Found {df[message_column].isnull().sum()} missing values in selected column. These will be skipped.")
            df = df.dropna(subset=[message_column])
        
        with st.expander("👀 Sample Messages from Selected Column", expanded=False):
            sample_messages = df[message_column].sample(min(5, len(df))).tolist()
            for i, msg in enumerate(sample_messages, 1):
                st.write(f"{i}. {msg}")
        
        df_processed = df[[message_column]].copy()
        
        # Optional Error Analysis Section
        st.subheader("⚙️ Optional Error Analysis")
        if st.checkbox("📊 Show Error Distribution Analysis", help="Check this box to analyze the 'ErrorName' column if it exists."):
            df_for_analysis = df.copy()
            df_for_analysis.columns = [col.strip().lower().replace(' ', '') for col in df_for_analysis.columns]

            if 'errorname' in df_for_analysis.columns:
                error_series = df_for_analysis['errorname'].dropna()
                error_series = error_series[error_series.str.strip() != 'No Error (code 0 )']
                error_series = error_series[error_series.str.strip() != '']

                total_rows = len(df_for_analysis)
                error_count = len(error_series)

                if error_count > 0:
                    st.info(f"Found {error_count} messages with errors (out of {total_rows} total rows).")
                    error_counts_df = error_series.value_counts().reset_index()
                    error_counts_df.columns = ['ErrorName', 'Count']
                    error_counts_df['Percentage'] = (error_counts_df['Count'] / total_rows * 100).round(2)

                    st.subheader('Error Distribution Table')
                    st.dataframe(error_counts_df, use_container_width=True)

                else:
                    st.success("✅ No errors found in the 'ErrorName' column.")
            else:
                st.warning("⚠️ No 'ErrorName' column found to analyze.")
        
        # Categorization section
        st.subheader('🔄 SMS Categorization')
        
        if st.button('🚀 Start Categorization', type="primary"):
            try:
                categorizer = SMSCategorizer()
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text('Initializing categorizer...')
                progress_bar.progress(10)
                
                status_text.text('Processing messages...')
                
                batch_size = max(1, len(df_processed) // 10)
                categories = []
                
                for i in range(0, len(df_processed), batch_size):
                    batch = df_processed[message_column].iloc[i:i+batch_size]
                    batch_categories = batch.apply(
                        lambda x: categorizer.pattern_based_categorization(
                            categorizer.preprocess_text(str(x))
                        )
                    ).tolist()
                    categories.extend(batch_categories)
                    
                    progress = min(90, 20 + (i / len(df_processed)) * 70)
                    progress_bar.progress(int(progress))
                    status_text.text(f'Processed {min(i + batch_size, len(df_processed))}/{len(df_processed)} messages...')
                
                df_processed['predicted_category'] = categories
                
                progress_bar.progress(100)
                status_text.text('✅ Categorization complete!')
                
                st.success(f'🎉 Successfully categorized {len(df_processed)} messages!')
                
                st.subheader('📊 Results')
                
                category_counts = df_processed['predicted_category'].value_counts()
                unique_categories = int(len(category_counts))
                most_common_category = str(category_counts.index[0]) if unique_categories > 0 else "N/A"
                most_common_count = int(category_counts.iloc[0]) if unique_categories > 0 else 0
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Messages", int(len(df_processed)))
                with col2:
                    st.metric("Unique Categories", unique_categories)
                with col3:
                    st.metric("Most Common Category", most_common_category)
                with col4:
                    st.metric("Messages in Top Category", most_common_count)
                
                with st.expander("📋 Detailed Results", expanded=False):
                    st.dataframe(
                        df_processed[[message_column, 'predicted_category']], 
                        use_container_width=True
                    )
                
                st.subheader('📈 Category Distribution')
                category_counts_df = category_counts.reset_index()
                category_counts_df.columns = ['Category', 'Count']
                category_counts_df['Percentage'] = (category_counts_df['Count'] / len(df_processed) * 100).round(2)

                st.dataframe(category_counts_df, use_container_width=True)
                
                st.subheader('📊 Visualizations')
                
                tab1, tab2, tab3 = st.tabs(["🥧 Pie Chart", "📊 Bar Chart", "📈 Horizontal Bar"])
                
                with tab1:
                    fig_pie = px.pie(
                        category_counts_df, 
                        names='Category', 
                        values='Count', 
                        title='Distribution of SMS Categories',
                        template=chart_theme
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with tab2:
                    fig_bar = px.bar(
                        category_counts_df, 
                        x='Category', 
                        y='Count', 
                        title='Number of Messages per Category',
                        text='Count',
                        template=chart_theme,
                        color='Count',
                        color_continuous_scale='viridis'
                    )
                    fig_bar.update_traces(texttemplate='%{text}', textposition='outside')
                    fig_bar.update_layout(showlegend=False)
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                with tab3:
                    fig_hbar = px.bar(
                        category_counts_df.sort_values('Count'), 
                        x='Count', 
                        y='Category', 
                        orientation='h',
                        title='Message Count by Category (Horizontal)',
                        text='Count',
                        template=chart_theme,
                        color='Count',
                        color_continuous_scale='plasma'
                    )
                    fig_hbar.update_traces(texttemplate='%{text}', textposition='outside')
                    fig_hbar.update_layout(showlegend=False)
                    st.plotly_chart(fig_hbar, use_container_width=True)
                
                st.subheader('💾 Download Results')
                
                download_df = df_processed[[message_column, 'predicted_category']].copy()
                download_df.columns = ['Message', 'Predicted_Category']
                
                download_df['Processing_Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                col1, col2 = st.columns(2)
                
                with col1:
                    csv = download_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label='📥 Download as CSV',
                        data=csv,
                        file_name=f'categorized_sms_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                        mime='text/csv'
                    )
                
                with col2:
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        download_df.to_excel(writer, sheet_name='Categorized_SMS', index=False)
                        category_counts_df.to_excel(writer, sheet_name='Category_Summary', index=False)
                    buffer.seek(0)
                    st.download_button(
                        label='📥 Download as Excel',
                        data=buffer.getvalue(),
                        file_name=f'categorized_sms_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )
                
            except Exception as e:
                st.error(f'❌ An error occurred during categorization: {str(e)}')
                st.info('Please check that your SMSCategorizer class is properly implemented and accessible.')
    
    except Exception as e:
        st.error(f'❌ Error reading or processing file: {str(e)}')
        st.info('Please ensure your file is properly formatted and not corrupted.')

else:
    st.info('👆 Please upload a file to get started')
    
    with st.expander("📋 Expected File Format", expanded=False):
        st.write("""
        Your file should contain SMS messages in one of these formats:
        
        **Option 1: CSV with 'text' column**
        ```
        text
        "Hello, how are you?"
        "Your order has been confirmed"
        "Meeting at 3 PM today"
        ```
        
        **Option 2: CSV with any column name**
        ```
        message,sender
        "Hello, how are you?",John
        "Your order has been confirmed",System
        ```
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666666; font-size: 0.9em;'>
    Fido Streamlit SMS Categorization App
</div>
""", unsafe_allow_html=True)
