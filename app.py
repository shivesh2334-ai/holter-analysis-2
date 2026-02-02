import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io
import os
import tempfile
import json
import textwrap
from typing import List, Dict, Any
import base64

# Try to import PDF processing libraries
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    st.warning("PyPDF2 not available - PDF uploads will be limited")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

# Try to import optional visualization libraries
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Philips Holter Monitor Analysis Guide with AI Assistant",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.philips.com/healthcare',
        'Report a bug': 'https://github.com/yourusername/holter-guide/issues',
        'About': """
        # Philips Holter Analysis Guide v2.0
        
        Clinical decision support tool with AI-powered PDF analysis.
        
        For educational purposes. Always consult official documentation.
        """
    }
)

# Custom CSS with enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #00539B;
        font-weight: 800;
        text-align: center;
        margin-bottom: 1.5rem;
        padding-bottom: 15px;
        border-bottom: 3px solid #00539B;
        background: linear-gradient(90deg, #00539B, #0083B0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .sub-header {
        font-size: 1.8rem;
        color: #00539B;
        font-weight: 700;
        margin-top: 1.8rem;
        margin-bottom: 1.2rem;
        padding-left: 15px;
        border-left: 5px solid #00539B;
    }
    
    .ai-response {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
        margin: 20px 0;
        box-shadow: 0 2px 4px rgba(76,175,80,0.2);
    }
    
    .pdf-content {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        font-family: 'Courier New', monospace;
        font-size: 0.9em;
        max-height: 300px;
        overflow-y: auto;
    }
    
    .code-block {
        background: #2d2d2d;
        color: #f8f8f2;
        padding: 15px;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        font-size: 0.9em;
        overflow-x: auto;
        margin: 10px 0;
    }
    
    .task-card {
        background: linear-gradient(135deg, #ffffff 0%, #f5f9ff 100%);
        padding: 25px;
        border-radius: 12px;
        border-left: 6px solid #00539B;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .tip-box {
        background: linear-gradient(135deg, #fff8e1 0%, #fff3cd 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ffb300;
        margin: 20px 0;
    }
    
    .step-number {
        display: inline-block;
        background: linear-gradient(135deg, #00539B, #0083B0);
        color: white;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        text-align: center;
        line-height: 32px;
        font-weight: bold;
        margin-right: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'user_data' not in st.session_state:
    st.session_state.user_data = {
        'role': 'Cardiac Technician',
        'bookmarks': [],
        'uploaded_pdfs': [],
        'chat_history': [],
        'extracted_texts': {}
    }

if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = {
        'current_patient': None,
        'analysis_started': False
    }

# PDF Processing Functions
def extract_text_from_pdf(pdf_file):
    """Extract text from PDF using available libraries"""
    text = ""
    
    try:
        if PDFPLUMBER_AVAILABLE:
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        elif PDF_AVAILABLE:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        else:
            return "PDF processing libraries not available. Please install PyPDF2 or pdfplumber."
    
    except Exception as e:
        return f"Error extracting text: {str(e)}"
    
    return text

def analyze_pdf_content(text, max_length=5000):
    """Analyze PDF content and extract key information"""
    if len(text) > max_length:
        text = text[:max_length] + "... [truncated]"
    
    # Basic analysis - in a real app, you might use NLP here
    lines = text.split('\n')
    
    analysis = {
        'total_lines': len(lines),
        'total_words': len(text.split()),
        'estimated_pages': len(text) // 2500 + 1,
        'has_ecg_terms': any(term in text.lower() for term in ['ecg', 'ekg', 'holter', 'arrhythmia', 'cardiac']),
        'has_medical_terms': any(term in text.lower() for term in ['patient', 'diagnosis', 'treatment', 'medication', 'clinical']),
        'first_lines': lines[:10],
        'sample_text': text[:1000] if len(text) > 1000 else text
    }
    
    return analysis

def generate_function_from_description(description):
    """Generate Python function based on description"""
    # This is a simplified version - in production, you'd use AI/ML models
    
    # Template for common Holter analysis functions
    function_templates = {
        'detect': """
def {function_name}(ecg_data, threshold={threshold}):
    \"\"\"
    Detect {condition} in ECG data.
    
    Parameters:
    -----------
    ecg_data : pd.DataFrame
        ECG data with columns: ['time', 'lead_i', 'lead_ii', 'lead_v5']
    threshold : float
        Detection threshold
        
    Returns:
    --------
    dict
        Dictionary with detection results
    \"\"\"
    import numpy as np
    import pandas as pd
    
    results = {{
        'detected': False,
        'episodes': [],
        'total_duration': 0,
        'confidence': 0.0
    }}
    
    # Placeholder detection logic
    # In real implementation, add your detection algorithm here
    
    return results
""",
        'analyze': """
def {function_name}(data, parameters={parameters}):
    \"\"\"
    Analyze {analysis_type} from Holter data.
    
    Parameters:
    -----------
    data : dict or pd.DataFrame
        Input data for analysis
    parameters : dict
        Analysis parameters
        
    Returns:
    --------
    dict
        Analysis results
    \"\"\"
    results = {{
        'analysis_type': '{analysis_type}',
        'parameters_used': parameters,
        'results': {{}},
        'status': 'completed'
    }}
    
    # Placeholder analysis logic
    # Add your analysis code here
    
    return results
""",
        'calculate': """
def {function_name}(*args, **kwargs):
    \"\"\"
    Calculate {calculation} from input parameters.
    
    Returns:
    --------
    float or dict
        Calculated value or results
    \"\"\"
    # Placeholder calculation
    # Add your calculation logic here
    
    return 0.0
"""
    }
    
    # Simple keyword matching to choose template
    description_lower = description.lower()
    
    if any(word in description_lower for word in ['detect', 'find', 'identify']):
        template_key = 'detect'
        function_name = 'detect_' + description_lower.split()[0] if len(description_lower.split()) > 0 else 'detect_pattern'
        threshold = 0.5
        condition = description_lower.replace('detect', '').strip() or 'pattern'
        
        return function_templates[template_key].format(
            function_name=function_name,
            threshold=threshold,
            condition=condition
        )
    
    elif any(word in description_lower for word in ['analyze', 'process', 'evaluate']):
        template_key = 'analyze'
        function_name = 'analyze_' + description_lower.split()[0] if len(description_lower.split()) > 0 else 'analyze_data'
        analysis_type = description_lower.replace('analyze', '').strip() or 'data'
        parameters = "{'window_size': 10, 'sampling_rate': 200}"
        
        return function_templates[template_key].format(
            function_name=function_name,
            analysis_type=analysis_type,
            parameters=parameters
        )
    
    else:
        # Default template for calculations
        return """
def process_{description}(input_data):
    \"\"\"
    Process: {description}
    
    This function was generated based on your description.
    Modify it according to your specific needs.
    \"\"\"
    # TODO: Implement the logic for: {description}
    
    # Example structure:
    results = {
        'input_processed': True,
        'description': '{description}',
        'output': None  # Replace with actual output
    }
    
    return results
""".format(description=description.replace(' ', '_').lower())

def answer_question_based_on_text(question, pdf_text, context=""):
    """Generate answer based on PDF text and question"""
    
    # Simple keyword-based answering
    # In production, you would use more sophisticated NLP
    
    keywords = {
        'ecg': ['electrocardiogram', 'heart rate', 'rhythm', 'beat', 'qrs', 'qt'],
        'holter': ['24-hour', 'monitor', 'recording', 'ambulatory', 'portable'],
        'arrhythmia': ['irregular', 'abnormal', 'tachycardia', 'bradycardia', 'afib', 'flutter'],
        'analysis': ['process', 'analyze', 'evaluate', 'interpret', 'review']
    }
    
    answer_template = """
Based on the provided PDF content and your question about "{question}", here's what I can determine:

**PDF Analysis:**
- Total content extracted: {word_count} words
- Medical relevance: {medical_relevance}
- Contains ECG/Holter terms: {has_ecg}

**Answer:**
{answer_text}

**Suggested Actions:**
- Review the complete PDF for detailed information
- Consult clinical guidelines for protocol specifics
- Verify measurements with supervising physician
"""

    # Generate answer based on keywords
    pdf_lower = pdf_text.lower()
    question_lower = question.lower()
    
    has_ecg = any(term in pdf_lower for term in keywords['ecg'])
    has_holter = any(term in pdf_lower for term in keywords['holter'])
    
    if has_ecg and has_holter:
        answer_text = "The document appears to contain Holter monitor/ECG related content. "
    elif has_ecg:
        answer_text = "The document contains ECG/cardiac related information. "
    else:
        answer_text = "The document may not be specifically about ECG/Holter monitoring. "
    
    # Add specific responses based on question
    if 'how' in question_lower:
        answer_text += "For procedural questions, please refer to the step-by-step instructions in the manual sections."
    elif 'what' in question_lower:
        answer_text += "The document describes various aspects of cardiac monitoring and analysis."
    elif 'why' in question_lower:
        answer_text += "Clinical justifications are typically provided in guideline sections."
    
    # Count words
    word_count = len(pdf_text.split())
    
    return answer_template.format(
        question=question,
        word_count=word_count,
        medical_relevance="High" if has_ecg else "Low/Medium",
        has_ecg="Yes" if has_ecg else "No",
        answer_text=answer_text
    )

# Main Application Functions
def main():
    """Main application function"""
    
    # Header
    st.markdown('<div class="main-header">🫀 Philips Holter Guide with AI Assistant</div>', unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.title("📋 Navigation")
        
        # User profile
        with st.expander("👤 User Profile", expanded=False):
            st.session_state.user_data['role'] = st.selectbox(
                "Role:",
                ["Cardiologist", "Cardiac Technician", "Trainee", "Researcher", "Administrator"]
            )
        
        # Main navigation
        st.markdown("---")
        page = st.radio(
            "### 📚 Select Page:",
            [
                "🏠 Home Dashboard",
                "📄 PDF Analysis & Q&A",
                "🤖 AI Function Generator",
                "💓 AF Detection Guide",
                "📊 ST Analysis",
                "📝 Report Generation",
                "📚 Reference Guide"
            ]
        )
        
        # Uploaded PDFs
        if st.session_state.user_data['uploaded_pdfs']:
            st.markdown("---")
            st.markdown("### 📚 Uploaded PDFs")
            for pdf in st.session_state.user_data['uploaded_pdfs'][:3]:
                st.caption(f"• {pdf}")
        
        # Quick stats
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("PDFs", len(st.session_state.user_data['uploaded_pdfs']))
        with col2:
            st.metric("Functions", "12")
    
    # Page routing
    if page == "🏠 Home Dashboard":
        home_dashboard()
    elif page == "📄 PDF Analysis & Q&A":
        pdf_analysis_page()
    elif page == "🤖 AI Function Generator":
        ai_function_generator_page()
    elif page == "💓 AF Detection Guide":
        af_detection_page()
    elif page == "📝 Report Generation":
        report_generation_page()
    elif page == "📚 Reference Guide":
        reference_guide_page()
    elif page == "📊 ST Analysis":
        st_analysis_page()
    
    # Footer
    display_footer()

def home_dashboard():
    """Home dashboard page"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Active Patients", "24", "+3")
    with col2:
        st.metric("PDFs Analyzed", len(st.session_state.user_data['uploaded_pdfs']))
    with col3:
        st.metric("AI Functions", "8", "+2")
    
    st.markdown('<div class="sub-header">🚀 Quick Actions</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📤 Upload PDF", use_container_width=True):
            st.session_state.current_page = "📄 PDF Analysis & Q&A"
            st.rerun()
    with col2:
        if st.button("🤖 Generate Function", use_container_width=True):
            st.session_state.current_page = "🤖 AI Function Generator"
            st.rerun()
    with col3:
        if st.button("📊 Analyze Data", use_container_width=True):
            st.info("Data analysis started...")
    
    # Recent activity
    st.markdown('<div class="sub-header">📈 Recent Activity</div>', unsafe_allow_html=True)
    
    if st.session_state.user_data['chat_history']:
        for chat in st.session_state.user_data['chat_history'][-3:]:
            st.text_area("", f"Q: {chat['question'][:100]}...\nA: {chat['answer'][:200]}...", 
                        height=100, disabled=True)
    else:
        st.info("No recent activity. Upload a PDF or ask a question to get started.")

def pdf_analysis_page():
    """PDF Analysis and Q&A page"""
    st.markdown('<div class="sub-header">📄 PDF Analysis & Q&A Assistant</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📤 Upload PDF", "❓ Ask Questions", "📊 PDF Insights"])
    
    with tab1:
        st.markdown("### Upload PDF for Analysis")
        
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", 
                                       help="Upload Philips Holter manuals, research papers, or clinical guidelines")
        
        if uploaded_file is not None:
            # Save file info
            file_details = {
                "filename": uploaded_file.name,
                "filetype": uploaded_file.type,
                "filesize": uploaded_file.size
            }
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**File Details:**")
                st.write(f"Name: {file_details['filename']}")
                st.write(f"Size: {file_details['filesize']} bytes")
            
            with col2:
                # Extract text button
                if st.button("🔍 Extract & Analyze Text", use_container_width=True):
                    with st.spinner("Extracting text from PDF..."):
                        # Extract text
                        text = extract_text_from_pdf(uploaded_file)
                        
                        # Analyze content
                        analysis = analyze_pdf_content(text)
                        
                        # Store in session state
                        if uploaded_file.name not in st.session_state.user_data['extracted_texts']:
                            st.session_state.user_data['extracted_texts'][uploaded_file.name] = {
                                'text': text,
                                'analysis': analysis,
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                        
                        # Add to uploaded PDFs list
                        if uploaded_file.name not in st.session_state.user_data['uploaded_pdfs']:
                            st.session_state.user_data['uploaded_pdfs'].append(uploaded_file.name)
                        
                        st.success("✅ PDF analyzed successfully!")
                        
                        # Show analysis results
                        st.markdown("### 📊 Analysis Results")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Words", analysis['total_words'])
                            st.metric("Estimated Pages", analysis['estimated_pages'])
                        with col2:
                            st.metric("Medical Relevance", "High" if analysis['has_medical_terms'] else "Low")
                            st.metric("ECG Content", "Yes" if analysis['has_ecg_terms'] else "No")
            
            # Show sample text if available
            if uploaded_file.name in st.session_state.user_data['extracted_texts']:
                st.markdown("### 📄 Sample Text (First 1000 characters)")
                sample_text = st.session_state.user_data['extracted_texts'][uploaded_file.name]['text'][:1000]
                st.markdown(f'<div class="pdf-content">{sample_text}...</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### ❓ Ask Questions About Uploaded PDFs")
        
        # Select PDF to query
        if st.session_state.user_data['uploaded_pdfs']:
            selected_pdf = st.selectbox(
                "Select a PDF to query:",
                st.session_state.user_data['uploaded_pdfs']
            )
            
            question = st.text_area("Enter your question:", 
                                  placeholder="e.g., What does this PDF say about AF detection thresholds?")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                context = st.text_area("Additional context (optional):",
                                     placeholder="e.g., I'm particularly interested in R-R interval analysis...")
            
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🤖 Get Answer", use_container_width=True):
                    if selected_pdf in st.session_state.user_data['extracted_texts']:
                        pdf_text = st.session_state.user_data['extracted_texts'][selected_pdf]['text']
                        
                        with st.spinner("Analyzing PDF and generating answer..."):
                            answer = answer_question_based_on_text(question, pdf_text, context)
                            
                            # Store in chat history
                            st.session_state.user_data['chat_history'].append({
                                'pdf': selected_pdf,
                                'question': question,
                                'answer': answer,
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                            
                            # Display answer
                            st.markdown('<div class="ai-response">', unsafe_allow_html=True)
                            st.markdown(answer)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Option to generate function from answer
                            if st.checkbox("Generate a function based on this answer?"):
                                function_code = generate_function_from_description(answer[:200])
                                st.markdown("### 🐍 Generated Function")
                                st.code(function_code, language='python')
                                
                                # Download option
                                st.download_button(
                                    label="📥 Download Function",
                                    data=function_code,
                                    file_name="generated_function.py",
                                    mime="text/x-python"
                                )
                    else:
                        st.error("Please extract text from the PDF first (go to Upload PDF tab).")
        else:
            st.info("Please upload a PDF first to ask questions.")
    
    with tab3:
        st.markdown("### 📊 PDF Insights Dashboard")
        
        if st.session_state.user_data['extracted_texts']:
            # Create insights dashboard
            insights_data = []
            
            for filename, data in st.session_state.user_data['extracted_texts'].items():
                insights_data.append({
                    'PDF Name': filename,
                    'Word Count': data['analysis']['total_words'],
                    'Lines': data['analysis']['total_lines'],
                    'ECG Content': 'Yes' if data['analysis']['has_ecg_terms'] else 'No',
                    'Medical Terms': 'Yes' if data['analysis']['has_medical_terms'] else 'No',
                    'Last Analyzed': data['timestamp']
                })
            
            insights_df = pd.DataFrame(insights_data)
            st.dataframe(insights_df, use_container_width=True, hide_index=True)
            
            # Show word cloud or other visualizations
            st.markdown("### 📈 Content Overview")
            
            col1, col2 = st.columns(2)
            with col1:
                if insights_df is not None and not insights_df.empty:
                    st.bar_chart(insights_df.set_index('PDF Name')['Word Count'])
            
            with col2:
                ecg_count = insights_df['ECG Content'].value_counts()
                if not ecg_count.empty:
                    st.metric("PDFs with ECG Content", 
                            f"{ecg_count.get('Yes', 0)}/{len(insights_df)}")
        else:
            st.info("No PDFs analyzed yet. Upload and analyze a PDF to see insights.")

def ai_function_generator_page():
    """AI Function Generator page"""
    st.markdown('<div class="sub-header">🤖 AI-Powered Function Generator</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["💬 Describe Function", "📚 Use PDF Context", "📦 Function Library"])
    
    with tab1:
        st.markdown("### 💬 Describe Your Function")
        
        function_description = st.text_area(
            "Describe the function you need:",
            height=150,
            placeholder="""Example: "Create a function to detect atrial fibrillation episodes in ECG data based on R-R interval variability with configurable threshold parameters.""""
        )
        
        # Additional parameters
        col1, col2 = st.columns(2)
        with col1:
            function_type = st.selectbox(
                "Function Type:",
                ["Detection", "Analysis", "Calculation", "Visualization", "Utility", "Report"]
            )
            return_type = st.selectbox(
                "Return Type:",
                ["Dictionary", "DataFrame", "Boolean", "List", "String", "Number", "Plot"]
            )
        
        with col2:
            language = st.selectbox("Language:", ["Python", "R", "MATLAB"])
            complexity = st.select_slider("Complexity:", 
                                        options=["Simple", "Medium", "Advanced", "Expert"])
        
        # Generate function button
        if st.button("🚀 Generate Function", use_container_width=True):
            if function_description:
                with st.spinner("Generating function code..."):
                    # Generate function based on description
                    function_code = generate_function_from_description(function_description)
                    
                    # Display generated function
                    st.markdown("### 🐍 Generated Python Function")
                    st.markdown('<div class="code-block">', unsafe_allow_html=True)
                    st.code(function_code, language='python')
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Function details
                    st.markdown("#### 📋 Function Details")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"**Type:** {function_type}")
                        st.info(f"**Complexity:** {complexity}")
                    with col2:
                        st.info(f"**Return Type:** {return_type}")
                        st.info(f"**Language:** {language}")
                    
                    # Download options
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.download_button(
                            label="📥 Download .py",
                            data=function_code,
                            file_name="generated_function.py",
                            mime="text/x-python"
                        )
                    with col2:
                        # Create test template
                        test_template = f"""
import pytest
from generated_function import {function_code.split('def ')[1].split('(')[0]}

def test_generated_function():
    \"\"\"Test the generated function\"\"\"
    # TODO: Add test cases
    test_data = {{}}
    result = {function_code.split('def ')[1].split('(')[0]}(test_data)
    assert result is not None
"""
                        st.download_button(
                            label="🧪 Download Test",
                            data=test_template,
                            file_name="test_generated_function.py",
                            mime="text/x-python"
                        )
                    with col3:
                        # Create documentation
                        doc_template = f"""
# {function_code.split('def ')[1].split('(')[0]} Function Documentation

## Purpose
{function_description}

## Parameters
- **parameter1**: Description
- **parameter2**: Description

## Returns
{return_type}: Description of return value

## Example Usage
```python
result = {function_code.split('def ')[1].split('(')[0]}(example_data)
