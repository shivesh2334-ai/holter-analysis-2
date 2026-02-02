import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io
import os
import tempfile
import json
import re

# Page configuration
st.set_page_config(
    page_title="Philips Holter Analysis Guide",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #00539B;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1.5rem;
        padding-bottom: 10px;
        border-bottom: 3px solid #00539B;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #00539B;
        font-weight: bold;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-left: 10px;
        border-left: 4px solid #00539B;
    }
    .task-card {
        background: linear-gradient(135deg, #f5f9ff 0%, #e6f0ff 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #00539B;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .tip-box {
        background-color: #fff8e1;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #ffb300;
        margin: 15px 0;
    }
    .warning-box {
        background-color: #ffebee;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #f44336;
        margin: 15px 0;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #4caf50;
        margin: 15px 0;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #2196f3;
        margin: 15px 0;
    }
    .ai-response {
        background-color: #f3e5f5;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #9c27b0;
        margin: 20px 0;
    }
    .pdf-content {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        font-size: 0.9em;
        max-height: 300px;
        overflow-y: auto;
        border: 1px solid #ddd;
    }
    .code-block {
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 15px;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        font-size: 0.9em;
        overflow-x: auto;
        margin: 10px 0;
    }
    .step-number {
        display: inline-block;
        background-color: #00539B;
        color: white;
        width: 30px;
        height: 30px;
        border-radius: 50%;
        text-align: center;
        line-height: 30px;
        font-weight: bold;
        margin-right: 10px;
    }
    .metric-box {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        border-top: 5px solid #00539B;
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
        'extracted_texts': {},
        'generated_functions': []
    }

if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = {
        'current_patient': None,
        'analysis_started': False,
        'patient_data': {}
    }

# PDF Processing Functions
def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    text = ""
    try:
        # Try to import PyPDF2
        try:
            import PyPDF2
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except ImportError:
            text = "PyPDF2 not available. Please install with: pip install PyPDF2"
        except Exception as e:
            text = f"Error extracting text: {str(e)}"
    except Exception as e:
        text = f"Error: {str(e)}"
    
    return text

def analyze_pdf_content(text, max_length=5000):
    """Analyze PDF content and extract key information"""
    if len(text) > max_length:
        text = text[:max_length] + "... [truncated]"
    
    # Basic analysis
    lines = text.split('\n')
    
    analysis = {
        'total_lines': len(lines),
        'total_words': len(text.split()),
        'estimated_pages': len(text) // 2500 + 1,
        'has_ecg_terms': any(term in text.lower() for term in ['ecg', 'ekg', 'holter', 'arrhythmia', 'cardiac', 'heart']),
        'has_medical_terms': any(term in text.lower() for term in ['patient', 'diagnosis', 'treatment', 'medication', 'clinical', 'therapy']),
        'sample_text': text[:1000] if len(text) > 1000 else text
    }
    
    return analysis

def generate_function_from_description(description):
    """Generate Python function based on description"""
    
    # Clean description
    desc_lower = description.lower().strip()
    
    # Detect function type
    if any(word in desc_lower for word in ['detect', 'find', 'identify', 'check']):
        func_type = 'detection'
        func_name = 'detect_' + desc_lower.split()[1] if len(desc_lower.split()) > 1 else 'detect_pattern'
    elif any(word in desc_lower for word in ['calculate', 'compute', 'measure', 'quantify']):
        func_type = 'calculation'
        func_name = 'calculate_' + desc_lower.split()[1] if len(desc_lower.split()) > 1 else 'calculate_value'
    elif any(word in desc_lower for word in ['analyze', 'process', 'evaluate', 'assess']):
        func_type = 'analysis'
        func_name = 'analyze_' + desc_lower.split()[1] if len(desc_lower.split()) > 1 else 'analyze_data'
    elif any(word in desc_lower for word in ['generate', 'create', 'build', 'make']):
        func_type = 'generation'
        func_name = 'generate_' + desc_lower.split()[1] if len(desc_lower.split()) > 1 else 'generate_output'
    else:
        func_type = 'utility'
        func_name = 'process_' + desc_lower.split()[0] if desc_lower.split() else 'custom_function'
    
    # Clean function name
    func_name = re.sub(r'[^a-zA-Z0-9_]', '_', func_name)
    
    # Generate appropriate function
    if func_type == 'detection':
        function_code = f'''
def {func_name}(data, threshold=0.5, window_size=10):
    """
    Detect {description}
    
    Parameters:
    data : numpy.ndarray or pd.DataFrame
        Input data for detection
    threshold : float, default=0.5
        Detection threshold
    window_size : int, default=10
        Analysis window size
        
    Returns:
    dict with detection results
    """
    import numpy as np
    
    results = {{
        'detected': False,
        'confidence': 0.0,
        'locations': [],
        'metadata': {{
            'threshold_used': threshold,
            'window_size': window_size,
            'description': "{description}"
        }}
    }}
    
    # Convert to numpy array if needed
    if hasattr(data, 'values'):
        data_array = data.values
    else:
        data_array = np.array(data)
    
    # Simple detection logic (placeholder)
    if len(data_array) > window_size:
        variances = []
        for i in range(0, len(data_array) - window_size + 1, window_size):
            window = data_array[i:i+window_size]
            variances.append(np.var(window))
        
        avg_variance = np.mean(variances)
        if avg_variance > threshold:
            results['detected'] = True
            results['confidence'] = min(avg_variance, 1.0)
            
    return results
'''
    
    elif func_type == 'calculation':
        function_code = f'''
def {func_name}(*args, **kwargs):
    """
    Calculate {description}
    
    Parameters:
    *args : variable length arguments
    **kwargs : keyword arguments
        
    Returns:
    float or dict with calculated value(s)
    """
    import numpy as np
    
    method = kwargs.get('method', 'standard')
    precision = kwargs.get('precision', 4)
    
    try:
        if args:
            values = np.array(args)
            if method == 'mean':
                result = np.mean(values)
            elif method == 'sum':
                result = np.sum(values)
            elif method == 'median':
                result = np.median(values)
            else:
                result = np.mean(values)
        
        result = round(float(result), precision)
        
        return {{
            'value': result,
            'method': method,
            'precision': precision,
            'description': "{description}"
        }}
        
    except Exception as e:
        return {{
            'error': str(e),
            'value': None,
            'description': "{description}"
        }}
'''
    
    elif func_type == 'analysis':
        function_code = f'''
def {func_name}(data, parameters=None):
    """
    Analyze {description}
    
    Parameters:
    data : various
        Input data for analysis
    parameters : dict, optional
        Analysis parameters
        
    Returns:
    dict with analysis results
    """
    import numpy as np
    import pandas as pd
    
    if parameters is None:
        parameters = {{
            'analysis_type': 'basic',
            'normalize': True,
            'return_stats': True
        }}
    
    results = {{
        'analysis_type': parameters.get('analysis_type', 'basic'),
        'parameters': parameters,
        'description': "{description}",
        'results': {{}},
        'success': False
    }}
    
    try:
        if isinstance(data, pd.DataFrame):
            df = data
        elif isinstance(data, np.ndarray):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame({{'data': data}})
        
        stats = df.describe().to_dict()
        
        results['results'] = {{
            'summary_statistics': stats,
            'data_shape': df.shape,
            'data_types': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict()
        }}
        
        results['success'] = True
        
    except Exception as e:
        results['error'] = str(e)
    
    return results
'''
    
    else:
        function_code = f'''
def {func_name}(input_data, config=None):
    """
    Process: {description}
    
    This function was generated based on your description.
    
    Parameters:
    input_data : various
        Input data to process
    config : dict, optional
        Configuration parameters
        
    Returns:
    dict with processing results
    """
    if config is None:
        config = {{
            'verbose': False,
            'validate_input': True
        }}
    
    results = {{
        'function': "{func_name}",
        'description': "{description}",
        'config': config,
        'input_received': True if input_data is not None else False,
        'output': None,
        'status': 'completed'
    }}
    
    # Add your processing logic here
    
    results['output'] = {{
        'processed': True,
        'note': 'Template function - implement your logic here'
    }}
    
    return results
'''
    
    return function_code.strip()

def answer_question_based_on_text(question, pdf_text, context=""):
    """Generate answer based on PDF text and question"""
    
    pdf_lower = pdf_text.lower()
    question_lower = question.lower()
    
    # Check for ECG/Holter terms
    ecg_terms = ['ecg', 'ekg', 'electrocardiogram', 'holter', 'cardiac', 'heart rate', 'arrhythmia']
    has_ecg = any(term in pdf_lower for term in ecg_terms)
    
    # Check for analysis terms
    analysis_terms = ['analyze', 'analysis', 'detect', 'measure', 'calculate', 'evaluate']
    is_analysis = any(term in question_lower for term in analysis_terms)
    
    # Generate appropriate response
    if has_ecg:
        base_answer = "Based on the ECG/Holter-related content in the document, "
    else:
        base_answer = "Based on the document content, "
    
    if is_analysis:
        answer = base_answer + f"I can help you {question_lower}. "
        answer += "For analysis procedures, you can use the AI Function Generator to create custom analysis functions. "
    elif question_lower.startswith('how'):
        answer = base_answer + "The procedure typically involves: \n\n"
        answer += "1. Data acquisition and preprocessing\n"
        answer += "2. Feature extraction from signals\n"
        answer += "3. Applying detection algorithms\n"
        answer += "4. Validating results with clinical standards\n"
    elif question_lower.startswith('what'):
        answer = base_answer + "This refers to cardiac monitoring techniques for detecting arrhythmias and other heart conditions over extended periods."
    elif question_lower.startswith('why'):
        answer = base_answer + "This is important for early detection of cardiac abnormalities, monitoring treatment efficacy, and preventing serious cardiac events."
    else:
        answer = base_answer + "The document contains relevant information for Holter monitor analysis. For specific details, please refer to the extracted text."
    
    # Add context if provided
    if context:
        answer += f"\n\n**Context provided:** {context}"
    
    # Add suggestions
    answer += "\n\n**Suggested Actions:**\n"
    answer += "1. Review the extracted PDF text for detailed information\n"
    answer += "2. Use the AI Function Generator for automated analysis\n"
    answer += "3. Consult clinical guidelines for validation\n"
    
    return answer

# Sample Data Generation
def generate_sample_ecg_data():
    """Generate synthetic ECG data"""
    fs = 200
    t = np.arange(0, 10, 1/fs)
    
    # Generate ECG components
    p_wave = 0.3 * np.sin(2 * np.pi * 1 * t)
    qrs = 1.5 * np.exp(-((t % 1) - 0.3)**2 / 0.001)
    t_wave = 0.4 * np.exp(-((t % 1) - 0.5)**2 / 0.002)
    noise = 0.05 * np.random.randn(len(t))
    
    ecg = p_wave + qrs + t_wave + noise
    
    return pd.DataFrame({
        'Time (s)': t,
        'ECG Signal (mV)': ecg,
        'Heart Rate (bpm)': 60 + 20 * np.sin(2 * np.pi * 0.1 * t)
    })

def generate_sample_patient_data():
    """Generate sample patient data"""
    patients = []
    conditions = ['Normal', 'AF', 'PVC', 'Bradycardia', 'Tachycardia']
    
    for i in range(8):
        condition = np.random.choice(conditions)
        patients.append({
            'ID': f'PAT-{1000 + i}',
            'Name': f'Patient {i+1}',
            'Age': np.random.randint(35, 85),
            'Gender': np.random.choice(['M', 'F']),
            'Condition': condition,
            'Recording Hours': np.random.choice([24, 48, 72]),
            'AF Burden (%)': round(np.random.uniform(0, 30), 1) if condition == 'AF' else 0,
            'Status': np.random.choice(['Completed', 'In Progress', 'Needs Review'])
        })
    
    return pd.DataFrame(patients)

def generate_sample_report():
    """Generate a sample Holter report"""
    report = f"""
PHILIPS HOLTER MONITOR ANALYSIS REPORT
=======================================

Report Date: {datetime.now().strftime('%Y-%m-%d')}
Report ID: HLR-{np.random.randint(10000, 99999)}

PATIENT INFORMATION
-------------------
Patient ID: PAT-{np.random.randint(1000, 9999)}
Name: [Patient Name]
Age: {np.random.randint(40, 80)}
Gender: {np.random.choice(['Male', 'Female'])}
Recording Duration: {np.random.choice(['24 hours', '48 hours', '72 hours'])}

ANALYSIS SUMMARY
----------------
Total Beats Analyzed: {np.random.randint(80000, 120000):,}
Average Heart Rate: {np.random.randint(60, 85)} bpm
Maximum Heart Rate: {np.random.randint(120, 180)} bpm
Minimum Heart Rate: {np.random.randint(40, 55)} bpm

ARRHYTHMIA FINDINGS
-------------------
- Atrial Fibrillation: {np.random.choice(['Present', 'Absent', 'Occasional'])}
- Ventricular Ectopy: {np.random.randint(0, 1000)} beats
- Supraventricular Ectopy: {np.random.randint(0, 500)} beats
- Pauses: {np.random.randint(0, 5)} > 2.0 seconds

ST SEGMENT ANALYSIS
-------------------
- ST Deviation: {np.random.choice(['Normal', 'Minimal', 'Significant'])}
- Ischemic Episodes: {np.random.randint(0, 3)}

HEART RATE VARIABILITY
----------------------
- SDNN: {np.random.randint(80, 180)} ms
- RMSSD: {np.random.randint(20, 60)} ms
- pNN50: {np.random.randint(5, 25)}%

CLINICAL IMPRESSION
-------------------
{np.random.choice([
    'Normal Holter study',
    'Intermittent atrial fibrillation noted',
    'Frequent ventricular ectopy',
    'Sinus rhythm with occasional PACs/PVCs',
    'No significant arrhythmias detected'
])}

RECOMMENDATIONS
---------------
1. {np.random.choice([
    'Clinical correlation recommended',
    'Consider cardiology referral',
    'Repeat Holter in 6 months',
    'No further action required'
])}

---
Report Generated by: Philips Holter Analysis System
"""
    return report

# Page Functions
def home_dashboard():
    """Home dashboard page"""
    st.markdown('<div class="main-header">🫀 Philips Holter Analysis Guide</div>', unsafe_allow_html=True)
    
    # Welcome message
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"### Welcome, {st.session_state.user_data['role']}!")
        st.markdown("*Your comprehensive guide for Philips Holter monitor analysis*")
    with col2:
        st.metric("Today's Date", datetime.now().strftime('%Y-%m-%d'))
    
    # Dashboard metrics
    st.markdown('<div class="sub-header">📊 Dashboard Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Analysis Tasks", "12")
    with col2:
        st.metric("Scanning Modes", "8")
    with col3:
        st.metric("PDFs Uploaded", len(st.session_state.user_data['uploaded_pdfs']))
    with col4:
        st.metric("Support", "24/7")
    
    # Quick actions
    st.markdown('<div class="sub-header">🚀 Quick Actions</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📤 Upload PDF Document", use_container_width=True):
            st.info("Navigate to PDF Analysis tab")
    with col2:
        if st.button("🤖 Generate Analysis Function", use_container_width=True):
            st.info("Navigate to AI Function Generator tab")
    with col3:
        if st.button("📊 View Sample Analysis", use_container_width=True):
            st.info("Navigate to AF Detection Guide")
    
    # Recent patients
    st.markdown('<div class="sub-header">👥 Recent Patients</div>', unsafe_allow_html=True)
    patient_df = generate_sample_patient_data()
    st.dataframe(patient_df, use_container_width=True, hide_index=True)
    
    # System overview
    st.markdown('<div class="sub-header">🖥️ System Overview</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-box">
        <strong>Software Capabilities:</strong>
        <ul>
        <li>Automatic arrhythmia detection</li>
        <li>ST segment analysis</li>
        <li>Heart rate variability</li>
        <li>Pacemaker analysis</li>
        <li>Custom report generation</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="tip-box">
        <strong>💡 Quick Tips:</strong>
        <ul>
        <li>Use Retrospective mode for comprehensive AF detection</li>
        <li>Always verify automated detections manually</li>
        <li>Check patient diary for symptom correlation</li>
        <li>Export reports in multiple formats</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

def pdf_analysis_page():
    """PDF Analysis and Q&A page"""
    st.markdown('<div class="sub-header">📄 PDF Analysis & Q&A Assistant</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Extract", "❓ Ask Questions", "📊 PDF Insights"])
    
    with tab1:
        st.markdown("### Upload Philips Holter Documentation")
        
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", 
                                       help="Upload manuals, clinical guidelines, or research papers")
        
        if uploaded_file is not None:
            # Display file info
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(f"**File:** {uploaded_file.name}")
                st.write(f"**Size:** {uploaded_file.size:,} bytes")
            
            with col2:
                if st.button("🔍 Extract Text", use_container_width=True):
                    with st.spinner("Extracting text from PDF..."):
                        # Extract text
                        text = extract_text_from_pdf(uploaded_file)
                        
                        # Analyze content
                        analysis = analyze_pdf_content(text)
                        
                        # Store in session state
                        st.session_state.user_data['extracted_texts'][uploaded_file.name] = {
                            'text': text,
                            'analysis': analysis,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        # Add to uploaded list
                        if uploaded_file.name not in st.session_state.user_data['uploaded_pdfs']:
                            st.session_state.user_data['uploaded_pdfs'].append(uploaded_file.name)
                        
                        st.success(f"✅ Extracted {analysis['total_words']:,} words from PDF")
            
            # Show extracted text if available
            if uploaded_file.name in st.session_state.user_data['extracted_texts']:
                st.markdown("### 📄 Extracted Content Preview")
                
                data = st.session_state.user_data['extracted_texts'][uploaded_file.name]
                analysis = data['analysis']
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Words", f"{analysis['total_words']:,}")
                with col2:
                    st.metric("Lines", analysis['total_lines'])
                with col3:
                    st.metric("ECG Content", "Yes" if analysis['has_ecg_terms'] else "No")
                with col4:
                    st.metric("Medical Terms", "Yes" if analysis['has_medical_terms'] else "No")
                
                # Show sample text
                with st.expander("View Extracted Text", expanded=False):
                    st.markdown(f'<div class="pdf-content">{analysis["sample_text"]}</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### Ask Questions About Your PDFs")
        
        if st.session_state.user_data['uploaded_pdfs']:
            selected_pdf = st.selectbox(
                "Select a PDF to query:",
                st.session_state.user_data['uploaded_pdfs']
            )
            
            question = st.text_area("Your question:", 
                                  placeholder="e.g., What does this document say about AF detection thresholds?")
            
            context = st.text_area("Additional context (optional):",
                                 placeholder="e.g., I'm particularly interested in R-R interval analysis...",
                                 height=100)
            
            if st.button("🤖 Get Answer", use_container_width=True):
                if selected_pdf in st.session_state.user_data['extracted_texts']:
                    pdf_text = st.session_state.user_data['extracted_texts'][selected_pdf]['text']
                    
                    with st.spinner("Analyzing PDF and generating answer..."):
                        answer = answer_question_based_on_text(question, pdf_text, context)
                        
                        # Store in chat history
                        st.session_state.user_data['chat_history'].append({
                            'pdf': selected_pdf,
                            'question': question,
                            'answer': answer[:500] + "..." if len(answer) > 500 else answer,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                        
                        # Display answer
                        st.markdown('<div class="ai-response">', unsafe_allow_html=True)
                        st.markdown(answer)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Option to generate function
                        if st.checkbox("Generate a Python function based on this answer?"):
                            function_code = generate_function_from_description(answer[:100])
                            st.markdown("### 🐍 Generated Function")
                            st.markdown('<div class="code-block">', unsafe_allow_html=True)
                            st.code(function_code, language='python')
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Download button
                            st.download_button(
                                label="📥 Download Function",
                                data=function_code,
                                file_name="generated_function.py",
                                mime="text/x-python"
                            )
                else:
                    st.error("Please extract text from the PDF first.")
        else:
            st.info("Please upload a PDF first in the 'Upload & Extract' tab.")
    
    with tab3:
        st.markdown("### 📊 PDF Analysis Dashboard")
        
        if st.session_state.user_data['extracted_texts']:
            # Create insights table
            insights_data = []
            for filename, data in st.session_state.user_data['extracted_texts'].items():
                analysis = data['analysis']
                insights_data.append({
                    'PDF Name': filename,
                    'Words': analysis['total_words'],
                    'Lines': analysis['total_lines'],
                    'ECG Content': '✓' if analysis['has_ecg_terms'] else '✗',
                    'Medical': '✓' if analysis['has_medical_terms'] else '✗',
                    'Last Analyzed': data['timestamp']
                })
            
            insights_df = pd.DataFrame(insights_data)
            st.dataframe(insights_df, use_container_width=True, hide_index=True)
        else:
            st.info("No PDFs analyzed yet. Upload and extract text from a PDF to see insights.")

def ai_function_generator_page():
    """AI Function Generator page"""
    st.markdown('<div class="sub-header">🤖 AI-Powered Function Generator</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["💬 Describe Function", "📚 Use PDF Context", "📦 Function Library"])
    
    with tab1:
        st.markdown("### Create Custom Analysis Functions")
        
        # Fixed: Using single quotes for the placeholder
        function_description = st.text_area(
            "Describe the function you need:",
            height=150,
            placeholder='Example: "Create a function to detect atrial fibrillation in ECG data based on R-R interval variability"'
        )
        
        # Function parameters
        col1, col2 = st.columns(2)
        with col1:
            function_type = st.selectbox(
                "Function Type:",
                ["Detection", "Analysis", "Calculation", "Visualization", "Utility"]
            )
            language = st.selectbox("Language:", ["Python", "Pseudocode"])
        
        with col2:
            complexity = st.select_slider(
                "Complexity:",
                options=["Simple", "Intermediate", "Advanced"]
            )
        
        # Generate function
        if st.button("🚀 Generate Function Code", use_container_width=True):
            if function_description:
                with st.spinner("Generating function code..."):
                    # Generate function
                    function_code = generate_function_from_description(function_description)
                    
                    # Store in session state
                    func_name = function_code.split('def ')[1].split('(')[0]
                    st.session_state.user_data['generated_functions'].append({
                        'name': func_name,
                        'description': function_description,
                        'code': function_code,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                    # Display function
                    st.markdown("### 🐍 Generated Python Function")
                    st.markdown('<div class="code-block">', unsafe_allow_html=True)
                    st.code(function_code, language='python')
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Function details
                    st.markdown("#### 📋 Function Details")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"**Name:** {func_name}")
                        st.info(f"**Type:** {function_type}")
                    with col2:
                        st.info(f"**Complexity:** {complexity}")
                        st.info(f"**Language:** {language}")
                    
                    # Download options
                    st.markdown("#### 📥 Download Options")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.download_button(
                            label="📄 Python File (.py)",
                            data=function_code,
                            file_name=f"{func_name}.py",
                            mime="text/x-python"
                        )
                    with col2:
                        # Create test template
                        test_template = f'''
"""
Test for {func_name} function
"""

def test_{func_name}():
    """Test the generated function"""
    import numpy as np
    
    # Test data
    test_data = np.random.randn(1000)
    
    # Call function
    result = {func_name}(test_data)
    
    # Basic assertions
    assert result is not None
    assert isinstance(result, dict)
    
    print(f"✅ {func_name} test passed!")
    return True

if __name__ == "__main__":
    test_{func_name}()
'''
                        st.download_button(
                            label="🧪 Test File",
                            data=test_template,
                            file_name=f"test_{func_name}.py",
                            mime="text/x-python"
                        )
                    with col3:
                        # Create documentation
                        doc_template = f"""
# {func_name} Function

## Description
{function_description}

## Usage
```python
from {func_name} import {func_name}

# Example usage
result = {func_name}(your_data)
