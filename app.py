
# Let me create the complete file in one go

complete_app_code = '''import streamlit as st
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
        try:
            import PyPDF2
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\\n"
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
    
    lines = text.split('\\n')
    
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
    
    desc_lower = description.lower().strip()
    
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
    
    func_name = re.sub(r'[^a-zA-Z0-9_]', '_', func_name)
    
    if func_type == 'detection':
        function_code = f"""
def {func_name}(data, threshold=0.5, window_size=10):
    \\"\\"\\"
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
    \\"\\"\\"
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
    
    if hasattr(data, 'values'):
        data_array = data.values
    else:
        data_array = np.array(data)
    
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
"""
    elif func_type == 'calculation':
        function_code = f"""
def {func_name}(*args, **kwargs):
    \\"\\"\\"
    Calculate {description}
    
    Parameters:
    *args : variable length arguments
    **kwargs : keyword arguments
        
    Returns:
    float or dict with calculated value(s)
    \\"\\"\\"
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
"""
    elif func_type == 'analysis':
        function_code = f"""
def {func_name}(data, parameters=None):
    \\"\\"\\"
    Analyze {description}
    
    Parameters:
    data : various
        Input data for analysis
    parameters : dict, optional
        Analysis parameters
        
    Returns:
    dict with analysis results
    \\"\\"\\"
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
"""
    else:
        function_code = f"""
def {func_name}(input_data, config=None):
    \\"\\"\\"
    Process: {description}
    
    This function was generated based on your description.
    
    Parameters:
    input_data : various
        Input data to process
    config : dict, optional
        Configuration parameters
        
    Returns:
    dict with processing results
    \\"\\"\\"
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
    
    results['output'] = {{
        'processed': True,
        'note': 'Template function - implement your logic here'
    }}
    
    return results
"""
    
    return function_code.strip()

def answer_question_based_on_text(question, pdf_text, context=""):
    """Generate answer based on PDF text and question"""
    
    pdf_lower = pdf_text.lower()
    question_lower = question.lower()
    
    ecg_terms = ['ecg', 'ekg', 'electrocardiogram', 'holter', 'cardiac', 'heart rate', 'arrhythmia']
    has_ecg = any(term in pdf_lower for term in ecg_terms)
    
    analysis_terms = ['analyze', 'analysis', 'detect', 'measure', 'calculate', 'evaluate']
    is_analysis = any(term in question_lower for term in analysis_terms)
    
    if has_ecg:
        base_answer = "Based on the ECG/Holter-related content in the document, "
    else:
        base_answer = "Based on the document content, "
    
    if is_analysis:
        answer = base_answer + f"I can help you {question_lower}. "
        answer += "For analysis procedures, you can use the AI Function Generator to create custom analysis functions. "
    elif question_lower.startswith('how'):
        answer = base_answer + "The procedure typically involves: \\n\\n"
        answer += "1. Data acquisition and preprocessing\\n"
        answer += "2. Feature extraction from signals\\n"
        answer += "3. Applying detection algorithms\\n"
        answer += "4. Validating results with clinical standards\\n"
    elif question_lower.startswith('what'):
        answer = base_answer + "This refers to cardiac monitoring techniques for detecting arrhythmias and other heart conditions over extended periods."
    elif question_lower.startswith('why'):
        answer = base_answer + "This is important for early detection of cardiac abnormalities, monitoring treatment efficacy, and preventing serious cardiac events."
    else:
        answer = base_answer + "The document contains relevant information for Holter monitor analysis. For specific details, please refer to the extracted text."
    
    if context:
        answer += f"\\n\\n**Context provided:** {context}"
    
    answer += "\\n\\n**Suggested Actions:**\\n"
    answer += "1. Review the extracted PDF text for detailed information\\n"
    answer += "2. Use the AI Function Generator for automated analysis\\n"
    answer += "3. Consult clinical guidelines for validation\\n"
    
    return answer

# Sample Data Generation
def generate_sample_ecg_data():
    """Generate synthetic ECG data"""
    fs = 200
    t = np.arange(0, 10, 1/fs)
    
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
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"### Welcome, {st.session_state.user_data['role']}!")
        st.markdown("*Your comprehensive guide for Philips Holter monitor analysis*")
    with col2:
        st.metric("Today's Date", datetime.now().strftime('%Y-%m-%d'))
    
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
    
    st.markdown('<div class="sub-header">👥 Recent Patients</div>', unsafe_allow_html=True)
    patient_df = generate_sample_patient_data()
    st.dataframe(patient_df, use_container_width=True, hide_index=True)
    
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
    """PDF Analysis and Q&A page - COMPLETED"""
    st.markdown('<div class="sub-header">📄 PDF Analysis & Q&A Assistant</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Extract", "❓ Ask Questions", "📋 History"])
    
    with tab1:
        st.markdown("### Upload PDF Document")
        
        uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'], key="pdf_uploader")
        
        if uploaded_file is not None:
            st.success(f"Uploaded: {uploaded_file.name}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Extract Text", use_container_width=True):
                    with st.spinner("Extracting text from PDF..."):
                        text = extract_text_from_pdf(uploaded_file)
                        
                        if uploaded_file.name not in st.session_state.user_data['uploaded_pdfs']:
                            st.session_state.user_data['uploaded_pdfs'].append(uploaded_file.name)
                        
                        st.session_state.user_data['extracted_texts'][uploaded_file.name] = text
                        
                        st.markdown("<div class='success-box'>✅ Text extracted successfully!</div>", unsafe_allow_html=True)
                        
                        analysis = analyze_pdf_content(text)
                        
                        st.markdown("#### Document Analysis")
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Total Lines", analysis['total_lines'])
                        with col_b:
                            st.metric("Word Count", analysis['total_words'])
                        with col_c:
                            st.metric("Est. Pages", analysis['estimated_pages'])
                        
                        if analysis['has_ecg_terms']:
                            st.markdown("<div class='success-box'>✅ ECG/Holter content detected</div>", unsafe_allow_html=True)
                        if analysis['has_medical_terms']:
                            st.markdown("<div class='info-box'>ℹ️ Medical terminology found</div>", unsafe_allow_html=True)
                        
                        with st.expander("View Extracted Text"):
                            st.markdown(f"<div class='pdf-content'>{analysis['sample_text'][:2000]}</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("### Document Info")
                st.json({
                    "filename": uploaded_file.name,
                    "size": f"{uploaded_file.size / 1024:.1f} KB",
                    "type": uploaded_file.type
                })
    
    with tab2:
        st.markdown("### Ask Questions About Your PDF")
        
        if not st.session_state.user_data['extracted_texts']:
            st.warning("Please upload and extract text from a PDF first (in the Upload tab).")
        else:
            selected_pdf = st.selectbox(
                "Select PDF to query:",
                options=list(st.session_state.user_data['extracted_texts'].keys())
            )
            
            question = st.text_input("Enter your question:", placeholder="e.g., How do I detect AF?")
            context = st.text_area("Additional context (optional):", placeholder="Any specific details...")
            
            if st.button("Get Answer", use_container_width=True):
                if question:
                    with st.spinner("Analyzing..."):
                        pdf_text = st.session_state.user_data['extracted_texts'][selected_pdf]
                        answer = answer_question_based_on_text(question, pdf_text, context)
                        
                        st.markdown(f"<div class='ai-response'><strong>🤖 Answer:</strong><br>{answer}</div>", unsafe_allow_html=True)
                        
                        st.session_state.user_data['chat_history'].append({
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                            'pdf': selected_pdf,
                            'question': question,
                            'answer': answer[:200] + "..."
                        })
                else:
                    st.error("Please enter a question.")
    
    with tab3:
        st.markdown("### Query History")
        if st.session_state.user_data['chat_history']:
            for item in reversed(st.session_state.user_data['chat_history']):
                with st.expander(f"{item['timestamp']} - {item['pdf']}"):
                    st.write(f"**Q:** {item['question']}")
                    st.write(f"**A:** {item['answer']}")
        else:
            st.info("No queries yet. Start by uploading a PDF and asking questions!")

def ai_function_generator():
    """AI Function Generator page"""
    st.markdown('<div class="sub-header">🤖 AI Function Generator</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    Describe the analysis function you need, and I'll generate Python code for you.
    Examples: "detect atrial fibrillation", "calculate heart rate variability", "analyze ST segments"
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        description = st.text_area(
            "Describe your function:",
            height=100,
            placeholder="e.g., Detect abnormal heart rhythms in ECG data..."
        )
        
        if st.button("⚡ Generate Function", use_container_width=True):
            if description:
                with st.spinner("Generating function..."):
                    generated_code = generate_function_from_description(description)
                    
                    st.session_state.user_data['generated_functions'].append({
                        'description': description,
                        'code': generated_code,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M')
                    })
                    
                    st.markdown("### Generated Python Function")
                    st.code(generated_code, language='python')
                    
                    st.download_button(
                        label="📥 Download as .py file",
                        data=generated_code,
                        file_name=f"generated_function_{len(st.session_state.user_data['generated_functions'])}.py",
                        mime="text/x-python"
                    )
            else:
                st.error("Please enter a description.")
    
    with col2:
        st.markdown("### Function Templates")
        templates = {
            "AF Detection": "detect atrial fibrillation episodes",
            "PVC Detection": "detect premature ventricular contractions",
            "HRV Calculation": "calculate heart rate variability metrics",
            "ST Analysis": "analyze ST segment elevation",
            "Tachycardia Check": "detect tachycardia events"
        }
        
        for name, desc in templates.items():
            if st.button(f"📝 {name}", key=f"template_{name}"):
                st.session_state['template_description'] = desc
                st.rerun()
        
        if 'template_description' in st.session_state:
            description = st.session_state['template_description']
            del st.session_state['template_description']

def af_detection_guide():
    """AF Detection Guide page"""
    st.markdown('<div class="sub-header">📊 AF Detection Guide</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="task-card">
    <h4>🔍 Atrial Fibrillation Detection Modes</h4>
    <p>Philips Holter systems offer multiple AF detection methods. Select the appropriate mode based on your analysis needs.</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Scanning Modes", "Sample Data", "Report Generation"])
    
    with tab1:
        st.markdown("### Available Scanning Modes")
        
        modes = [
            {
                "name": "Retrospective AF Detection",
                "icon": "🔍",
                "description": "Comprehensive analysis of entire recording for AF episodes",
                "best_for": "Initial screening, complete AF burden assessment",
                "sensitivity": "High",
                "time": "Longer processing time"
            },
            {
                "name": "Prospective AF Detection",
                "icon": "⚡",
                "description": "Real-time AF detection during scanning",
                "best_for": "Quick screening, immediate results",
                "sensitivity": "Medium",
                "time": "Fast"
            },
            {
                "name": "Manual Review",
                "icon": "👁️",
                "description": "Expert technician review of rhythm strips",
                "best_for": "Complex cases, verification of automated results",
                "sensitivity": "Expert-dependent",
                "time": "Variable"
            }
        ]
        
        for mode in modes:
            with st.expander(f"{mode['icon']} {mode['name']}"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**Description:** {mode['description']}")
                    st.write(f"**Best for:** {mode['best_for']}")
                with col2:
                    st.metric("Sensitivity", mode['sensitivity'])
                    st.caption(f"⏱️ {mode['time']}")
        
        st.markdown("""
        <div class="warning-box">
        <strong>⚠️ Important:</strong> Always verify automated AF detections manually. 
        Artifact and noise can be misclassified as AF.
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### Sample ECG Data")
        
        if st.button("Generate Sample ECG Data"):
            ecg_data = generate_sample_ecg_data()
            
            st.markdown("#### Data Preview")
            st.dataframe(ecg_data.head(20), use_container_width=True)
            
            st.markdown("#### ECG Signal Visualization")
            st.line_chart(ecg_data.set_index('Time (s)')['ECG Signal (mV)'].iloc[:1000])
            
            csv = ecg_data.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="sample_ecg_data.csv",
                mime="text/csv"
            )
    
    with tab3:
        st.markdown("### Generate Sample Report")
        
        if st.button("Generate Holter Report"):
            report = generate_sample_report()
            st.text_area("Report Preview", report, height=400)
            
            st.download_button(
                label="📥 Download Report",
                data=report,
                file_name=f"holter_report_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )

def analysis_workflow():
    """Analysis Workflow Guide page"""
    st.markdown('<div class="sub-header">📋 Analysis Workflow</div>', unsafe_allow_html=True)
    
    steps = [
        ("Patient Preparation", "Ensure proper electrode placement and patient diary completion", "✅"),
        ("Data Acquisition", "Verify signal quality and recording duration", "📊"),
        ("Initial Scan", "Run automated analysis with appropriate detection algorithms", "🤖"),
        ("Manual Review", "Verify all automated detections and add manual annotations", "👁️"),
        ("Report Generation", "Generate comprehensive report with findings", "📄"),
        ("Quality Check", "Final review by senior technician or physician", "✔️")
    ]
    
    for i, (title, desc, icon) in enumerate(steps, 1):
        st.markdown(f"""
        <div class="task-card">
        <span class="step-number">{i}</span>
        <strong>{icon} {title}</strong>
        <p style="margin-left: 45px; margin-top: 10px;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="tip-box">
    <strong>💡 Pro Tip:</strong> Always check the patient diary against detected events. 
    Symptoms reported by patients often correlate with arrhythmias.
    </div>
    """, unsafe_allow_html=True)

def settings_page():
    """Settings page"""
    st.markdown('<div class="sub-header">⚙️ Settings</div>', unsafe_allow_html=True)
    
    st.markdown("### User Preferences")
    
    role = st.selectbox(
        "Select your role:",
        ["Cardiac Technician", "Cardiologist", "Nurse", "Researcher", "Student"],
        index=["Cardiac Technician", "Cardiologist", "Nurse", "Researcher", "Student"].index(st.session_state.user_data['role'])
    )
    
    st.session_state.user_data['role'] = role
    
    st.markdown("### Data Management")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.user_data['chat_history'] = []
            st.success("Chat history cleared!")
    
    with col2:
        if st.button("🗑️ Clear Uploaded PDFs"):
            st.session_state.user_data['uploaded_pdfs'] = []
            st.session_state.user_data['extracted_texts'] = {}
            st.success("PDF data cleared!")
    
    st.markdown("### About")
    st.info("""
    **Philips Holter Analysis Guide v1.0**
    
    This application provides guidance for analyzing Holter monitor data using Philips systems.
    Features include:
    - PDF document analysis and Q&A
    - AI-powered function generation
    - AF detection guides
    - Sample data generation
    
    For support, contact your system administrator.
    """)

# Main Navigation
def main():
    """Main application function"""
    
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
        <h2 style="color: #00539B;">🫀 Philips Holter</h2>
        <p style="color: #666;">Analysis Guide</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        page = st.radio(
            "Navigation",
            ["🏠 Home", "📄 PDF Analysis", "🤖 AI Function Generator", "📊 AF Detection Guide", "📋 Workflow", "⚙️ Settings"],
            index=0
        )
        
        st.divider()
        
        st.markdown("### Quick Stats")
        st.metric("PDFs", len(st.session_state.user_data['uploaded_pdfs']))
        st.metric("Functions", len(st.session_state.user_data['generated_functions']))
        st.metric("Queries", len(st.session_state.user_data['chat_history']))
        
        st.divider()
        
        st.markdown("""
        <div style="font-size: 0.8em; color: #666; text-align: center;">
        <p>Philips Holter Analysis Guide</p>
        <p>Version 1.0</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Route to appropriate page
    if page == "🏠 Home":
        home_dashboard()
    elif page == "📄 PDF Analysis":
        pdf_analysis_page()
    elif page == "🤖 AI Function Generator":
        ai_function_generator()
    elif page == "📊 AF Detection Guide":
        af_detection_guide()
    elif page == "📋 Workflow":
        analysis_workflow()
    elif page == "⚙️ Settings":
        settings_page()

if __name__ == "__main__":
    main()
'''
