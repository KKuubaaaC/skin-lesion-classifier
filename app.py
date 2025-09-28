import streamlit as st
import numpy as np
from PIL import Image
import os
import json
import onnxruntime as ort
from torchvision import transforms
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import io

# Import PostgreSQL database
from database_postgresql import SkinLesionPostgreSQLDB

# Konfiguracja strony
st.set_page_config(
    page_title="HAM10000 Skin Lesion Classifier", 
    page_icon="🔬",
    layout="wide"
)

# CSS styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    
    .prediction-card {
        background-color: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .high-confidence {
        border-color: #28a745;
        background-color: #d4edda;
    }
    
    .medium-confidence {
        border-color: #ffc107;
        background-color: #fff3cd;
    }
    
    .low-confidence {
        border-color: #6c757d;
        background-color: #f8f9fa;
    }
    
    .patient-card {
        background-color: #e3f2fd;
        border: 1px solid #2196f3;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Opisy klas
CLASS_DESCRIPTIONS = {
    "akiec": {
        "full_name": "Actinic Keratoses",
        "description": "Rogowce słoneczne - zmiany przedrakowe skóry",
        "color": "#ff6b6b",
        "risk": "medium"
    },
    "bcc": {
        "full_name": "Basal Cell Carcinoma", 
        "description": "Rak podstawnokomórkowy - najczęstszy typ raka skóry",
        "color": "#ff4757",
        "risk": "high"
    },
    "bkl": {
        "full_name": "Benign Keratosis-like Lesions",
        "description": "Łagodne zmiany rogowaciejące",
        "color": "#2ed573",
        "risk": "low"
    },
    "df": {
        "full_name": "Dermatofibroma",
        "description": "Dermatofibroma - łagodny guz skóry",
        "color": "#1e90ff",
        "risk": "low"
    },
    "mel": {
        "full_name": "Melanoma",
        "description": "Czerniak - najgroźniejszy typ raka skóry",
        "color": "#8B0000",
        "risk": "very_high"
    },
    "nv": {
        "full_name": "Melanocytic Nevi",
        "description": "Znamiona melanocytowe - zazwyczaj łagodne",
        "color": "#32cd32",
        "risk": "low"
    },
    "vasc": {
        "full_name": "Vascular Lesions",
        "description": "Zmiany naczyniowe skóry",
        "color": "#ff69b4",
        "risk": "low"
    }
}

@st.cache_resource
def init_database():
    """Initialize PostgreSQL database connection"""
    try:
        return SkinLesionPostgreSQLDB()
    except Exception as e:
        st.error(f"❌ Database connection failed: {e}")
        st.error("Make sure PostgreSQL is running: docker-compose up postgres -d")
        return None

@st.cache_resource
def load_model():
    """Załaduj model ONNX"""
    try:
        model_path = "model/ham10000_model.onnx"
        if not os.path.exists(model_path):
            st.error(f"❌ Nie znaleziono modelu: {model_path}")
            return None
            
        ort_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        st.success("✅ Model załadowany pomyślnie!")
        return ort_session
    except Exception as e:
        st.error(f"❌ Błąd ładowania modelu: {e}")
        return None

@st.cache_data
def load_classes():
    """Załaduj mapowanie klas"""
    try:
        classes_path = "model/classes.json"
        if os.path.exists(classes_path):
            with open(classes_path, 'r') as f:
                return json.load(f)
        else:
            st.warning("⚠️ Nie znaleziono classes.json, używam domyślnego mapowania")
    except Exception as e:
        st.warning(f"⚠️ Błąd wczytywania classes.json: {e}")
    
    return {
        "0": "akiec", "1": "bcc", "2": "bkl", "3": "df",
        "4": "mel", "5": "nv", "6": "vasc"
    }

def preprocess_image(image):
    """Przygotuj obraz do predykcji"""
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(image).unsqueeze(0)
        return input_tensor.numpy()
    except Exception as e:
        st.error(f"Błąd przetwarzania obrazu: {e}")
        return None

def predict_lesion(image, ort_session, class_mapping):
    """Wykonaj predykcję"""
    try:
        input_numpy = preprocess_image(image)
        if input_numpy is None:
            return None
        
        input_name = ort_session.get_inputs()[0].name
        outputs = ort_session.run(None, {input_name: input_numpy})
        logits = outputs[0]
        
        probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        probabilities = probabilities[0]
        
        predictions = []
        for idx, prob in enumerate(probabilities):
            class_name = class_mapping.get(str(idx), f"Class_{idx}")
            confidence = float(prob) * 100
            
            predictions.append({
                'class': class_name,
                'confidence': confidence,
                'probability': float(prob),
                'class_info': CLASS_DESCRIPTIONS.get(class_name, {})
            })
        
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        return predictions
        
    except Exception as e:
        st.error(f"Błąd podczas predykcji: {e}")
        return None

def create_confidence_chart(predictions):
    """Utwórz wykres pewności"""
    classes = [pred['class'].upper() for pred in predictions]
    confidences = [pred['confidence'] for pred in predictions]
    colors = [pred['class_info'].get('color', '#3498db') for pred in predictions]
    
    fig = go.Figure(data=[
        go.Bar(
            x=classes,
            y=confidences,
            marker_color=colors,
            text=[f'{conf:.1f}%' for conf in confidences],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Confidence Scores for All Classes",
        xaxis_title="Skin Lesion Type",
        yaxis_title="Confidence (%)",
        yaxis_range=[0, 100],
        height=400,
        showlegend=False
    )
    
    return fig

def display_prediction_cards(predictions):
    """Wyświetl karty z predykcjami"""
    for i, pred in enumerate(predictions[:3]):
        class_name = pred['class']
        confidence = pred['confidence']
        class_info = pred['class_info']
        
        if confidence > 70:
            card_class = "prediction-card high-confidence"
            icon = "🎯"
        elif confidence > 40:
            card_class = "prediction-card medium-confidence"
            icon = "⚠️"
        else:
            card_class = "prediction-card low-confidence"
            icon = "❓"
        
        st.markdown(f"""
        <div class="{card_class}">
            <h4>{icon} #{i+1}: {class_info.get('full_name', class_name.upper())}</h4>
            <p><strong>Confidence:</strong> {confidence:.1f}%</p>
            <p><strong>Risk Level:</strong> {class_info.get('risk', 'unknown').upper()}</p>
            <p><em>{class_info.get('description', 'No description available')}</em></p>
        </div>
        """, unsafe_allow_html=True)

def patient_management_sidebar(db):
    """Patient management panel in sidebar"""
    if db is None:
        st.sidebar.error("❌ Database not connected")
        return None
    
    st.sidebar.header("👤 Patient Management")
    
    try:
        # Patient selection/creation
        existing_patients = db.get_all_patients()
        patient_options = ["New Patient"] + [p['patient_id'] for p in existing_patients]
        
        selected_patient = st.sidebar.selectbox("Select Patient", patient_options)
        
        if selected_patient == "New Patient":
            patient_id = st.sidebar.text_input("Patient ID*", value=f"PAT_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            patient_age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=None)
            patient_gender = st.sidebar.selectbox("Gender", ["", "Male", "Female", "Other"])
            patient_notes = st.sidebar.text_area("Notes", placeholder="Medical history, allergies, etc.")
            
            if st.sidebar.button("➕ Create Patient"):
                if patient_id:
                    if db.add_patient(patient_id, patient_age, patient_gender, patient_notes):
                        st.sidebar.success("✅ Patient created!")
                        st.rerun()
                    else:
                        st.sidebar.error("❌ Error creating patient")
        else:
            patient_id = selected_patient
            # Show patient info
            patient_data = next((p for p in existing_patients if p['patient_id'] == patient_id), None)
            if patient_data:
                st.sidebar.markdown(f"""
                <div class="patient-card">
                    <strong>Patient: {patient_id}</strong><br>
                    Age: {patient_data['age'] or 'N/A'}<br>
                    Gender: {patient_data['gender'] or 'N/A'}<br>
                    Analyses: {patient_data['total_analyses']}<br>
                    Last: {str(patient_data['last_analysis'])[:10] if patient_data['last_analysis'] else 'Never'}
                </div>
                """, unsafe_allow_html=True)
        
        return patient_id if selected_patient != "New Patient" or (selected_patient == "New Patient" and patient_id) else None
    
    except Exception as e:
        st.sidebar.error(f"❌ Database error: {e}")
        return None

def show_database_stats(db):
    """Show database statistics"""
    st.header("📊 Database Statistics")
    
    if db is None:
        st.error("❌ Database not connected")
        return
    
    try:
        stats = db.get_statistics()
        if not stats or not stats.get('basic'):
            st.info("No data in database yet.")
            return
        
        basic = stats['basic']
        
        # Basic metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Patients", basic.get('total_patients', 0))
        with col2:
            st.metric("Total Analyses", basic.get('total_analyses', 0))
        with col3:
            avg_conf = basic.get('avg_confidence')
            st.metric("Avg Confidence", f"{avg_conf:.1f}%" if avg_conf else "N/A")
        with col4:
            st.metric("Database", "PostgreSQL")
        
        # Class distribution
        if stats.get('class_distribution'):
            st.subheader("📋 Diagnosis Distribution")
            
            class_data = stats['class_distribution']
            classes = [row['predicted_class'] for row in class_data]
            counts = [row['count'] for row in class_data]
            
            if classes and counts:
                fig = px.pie(values=counts, names=classes, title="Distribution of Diagnoses")
                st.plotly_chart(fig, use_container_width=True)
                
                # Table with details
                df = pd.DataFrame(class_data)
                st.dataframe(df, use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Error loading statistics: {e}")

def show_patient_history(db, patient_id):
    """Show patient history"""
    st.header(f"📋 History for {patient_id}")
    
    if db is None:
        st.error("❌ Database not connected")
        return
    
    try:
        history = db.get_patient_history(patient_id)
        if not history:
            st.info("No analysis history for this patient.")
            return
        
        # Convert to DataFrame for better display
        df = pd.DataFrame(history)
        df['analysis_date'] = pd.to_datetime(df['analysis_date'])
        df = df.sort_values('analysis_date', ascending=False)
        
        # Display history table
        display_cols = ['analysis_date', 'predicted_class', 'confidence', 'image_filename']
        st.dataframe(
            df[display_cols].rename(columns={
                'analysis_date': 'Date',
                'predicted_class': 'Diagnosis', 
                'confidence': 'Confidence (%)',
                'image_filename': 'Image'
            }),
            use_container_width=True
        )
        
        # Trends chart
        if len(history) > 1:
            st.subheader("📈 Confidence Trend")
            fig = px.line(df, x='analysis_date', y='confidence', 
                         title=f"Confidence Trend for {patient_id}")
            st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Error loading patient history: {e}")

def main():
    # Initialize database
    db = init_database()
    
    # Navigation
    tab1, tab2, tab3 = st.tabs(["🔬 Analysis", "📊 Statistics", "👥 Patients"])
    
    with tab1:
        # Title
        st.markdown("""
        <div class="main-header">
            <h1>🔬 HAM10000 Skin Lesion Classifier</h1>
            <p>AI-powered dermatological image analysis with PostgreSQL database</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize session state
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
        if 'ort_session' not in st.session_state:
            st.session_state.ort_session = None
        
        # Patient management sidebar
        patient_id = patient_management_sidebar(db)
        
        # Model loading
        if st.sidebar.button("🔄 Load Model", type="primary"):
            with st.spinner("Loading model..."):
                st.session_state.ort_session = load_model()
                st.session_state.model_loaded = st.session_state.ort_session is not None
        
        if st.session_state.model_loaded:
            st.sidebar.success("✅ Model Ready!")
        else:
            st.sidebar.warning("❌ Model Not Loaded")
        
        # Database status
        if db is not None:
            st.sidebar.success("✅ PostgreSQL Connected")
        else:
            st.sidebar.error("❌ PostgreSQL Disconnected")
        
        # Main analysis interface
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("📤 Upload Image")
            uploaded_file = st.file_uploader(
                "Choose a skin lesion image", 
                type=['png', 'jpg', 'jpeg']
            )
            
            if uploaded_file is not None:
                try:
                    image = Image.open(uploaded_file).convert('RGB')
                    st.subheader("📸 Your Image")
                    st.image(image, use_column_width=True, caption=f"Uploaded: {uploaded_file.name}")
                    
                    width, height = image.size
                    file_size = len(uploaded_file.getvalue()) / 1024
                    st.info(f"📊 Size: {width}x{height}px, {file_size:.1f} KB")
                except Exception as e:
                    st.error(f"❌ Error loading image: {e}")
                    uploaded_file = None
        
        with col2:
            if uploaded_file is not None and st.session_state.model_loaded:
                st.header("🔬 Analysis Results")
                
                # Doctor notes
                doctor_notes = st.text_area("Doctor Notes (optional)", placeholder="Additional observations...")
                
                if st.button("🔍 Analyze", type="primary", use_container_width=True):
                    with st.spinner("🧠 Analyzing image..."):
                        class_mapping = load_classes()
                        predictions = predict_lesion(image, st.session_state.ort_session, class_mapping)
                        
                        if predictions:
                            top_pred = predictions[0]
                            st.success(f"🎯 **Prediction**: {top_pred['class'].upper()} ({top_pred['confidence']:.1f}%)")
                            
                            display_prediction_cards(predictions)
                            
                            chart = create_confidence_chart(predictions)
                            st.plotly_chart(chart, use_container_width=True)
                            
                            # Save to database
                            if patient_id and db and st.button("💾 Save to Database", type="secondary"):
                                try:
                                    img_bytes = io.BytesIO()
                                    image.save(img_bytes, format='PNG')
                                    img_data = img_bytes.getvalue()
                                    
                                    if db.save_analysis(patient_id, uploaded_file.name, img_data, predictions, doctor_notes):
                                        st.success("✅ Analysis saved to PostgreSQL database!")
                                    else:
                                        st.error("❌ Error saving to database")
                                except Exception as e:
                                    st.error(f"❌ Database save error: {e}")
                            
                            if not patient_id:
                                st.warning("⚠️ Select a patient to save results")
                            elif not db:
                                st.warning("⚠️ Database not connected")
            
            elif uploaded_file is not None:
                st.warning("⚠️ Please load the model first")
            else:
                st.info("👆 Upload an image to start analysis")
    
    with tab2:
        show_database_stats(db)
    
    with tab3:
        st.header("👥 Patient Management")
        
        if db is None:
            st.error("❌ Database not connected")
            return
        
        try:
            patients = db.get_all_patients()
            if patients:
                # Search and filter
                search_term = st.text_input("🔍 Search patients", placeholder="Patient ID or notes...")
                
                # Display patients
                for patient in patients:
                    if not search_term or search_term.lower() in patient['patient_id'].lower():
                        with st.expander(f"Patient: {patient['patient_id']}"):
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.write(f"**Age:** {patient['age'] or 'N/A'}")
                                st.write(f"**Gender:** {patient['gender'] or 'N/A'}")
                                st.write(f"**Total Analyses:** {patient['total_analyses']}")
                                st.write(f"**Last Analysis:** {str(patient['last_analysis'])[:19] if patient['last_analysis'] else 'Never'}")
                                if patient.get('notes'):
                                    st.write(f"**Notes:** {patient['notes']}")
                            
                            with col2:
                                if st.button(f"📋 View History", key=f"history_{patient['patient_id']}"):
                                    show_patient_history(db, patient['patient_id'])
            else:
                st.info("No patients in database yet. Create your first patient in the Analysis tab.")
        
        except Exception as e:
            st.error(f"❌ Error loading patients: {e}")

if __name__ == "__main__":
    main()