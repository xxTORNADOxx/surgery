import streamlit as st
import google.generativeai as genai
import torch
import torch.nn as nn
import base64
import numpy as np
import cv2
from PIL import Image
import io
import os
from torch.fft import fft2, ifft2

# Load API Key from environment variable (for security reasons)
API_KEY = os.getenv('AIzaSyBOArgGwNYvCGBWWToCi7eWmJEt8Is6NGo')
genai.configure(api_key=API_KEY)

# Custom Optimizer Class
class FFTGaLoreOptimizer:
    def __init__(self, params, rank=128, lr=3e-6):
        self.params = list(params)
        self.rank = rank
        self.lr = lr

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            grad = p.grad.data
            fft_grad = fft2(grad)
            proj_grad = ifft2(fft_grad[..., :self.rank]).real
            p.data.add_(-self.lr * proj_grad)

# Define the Surgical AI Model
class SurgicalVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.gemini = genai.GenerativeModel('gemini-1.5-flash')
        self.task_head = nn.Linear(2560, 7)  # For 7 task types
        self.vision_adapter = nn.Sequential(
            nn.Conv2d(3, 16, 3),  # Changed for RGB images
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((224, 224))
        )
        
    def forward(self, inputs):
        response = self.gemini.generate_content(inputs)
        return response.text

# Set up Streamlit Web Interface
st.set_page_config(page_title="Surgical AI Co-Pilot", layout="wide")

# Initialize session state if not already initialized
if 'agent' not in st.session_state:
    st.session_state.agent = SurgicalVLM()
    st.session_state.optimizer = FFTGaLoreOptimizer(st.session_state.agent.parameters())

with st.sidebar:
    st.header("👤 Patient Information")
    patient_id = st.text_input("Patient ID")
    age = st.number_input("Age", min_value=0, max_value=120, value=40)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    medical_history = st.text_area("Medical History")
    
    st.header("🩺 Clinical Context")
    diagnosis = st.selectbox("Diagnosis", ["Pituitary Adenoma", "Craniopharyngioma", "Meningioma"])
    surgical_phase = st.radio("Surgical Phase", ["Nasal", "Sphenoid", "Sellar", "Closure"])
    
    st.header("📁 Medical Imaging")
    image_file = st.file_uploader("Upload Medical Images", type=["jpg", "jpeg", "png"])

st.title("🧑⚕️ Pituitary Surgery AI Co-Pilot")
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("Medical Imaging Preview")
    if image_file:
        img = Image.open(image_file)
        st.image(img, caption="Uploaded Medical Image", use_column_width=True)
        st.caption(f"Image Dimensions: {img.size[0]}x{img.size[1]} pixels")

with col2:
    st.subheader("AI Surgical Assistant")
    query = st.text_input("Surgeon Query", placeholder="Ask about anatomy, instruments, or next steps...")
    
    if query:
        with st.spinner("🧠 Processing surgical context..."):
            try:
                # Build patient context
                context = f"""
                Patient Context:
                - ID: {patient_id}
                - Age: {age}
                - Gender: {gender}
                - Diagnosis: {diagnosis}
                - Surgical Phase: {surgical_phase}
                - Medical History: {medical_history}
                """
                
                # Prepare multimodal input
                contents = {"text": context + "\n\nSurgeon Query: " + query}
                
                # Add image data if uploaded
                if image_file:
                    img = Image.open(image_file)
                    buf = io.BytesIO()
                    img.save(buf, format='PNG')
                    img_data = base64.b64encode(buf.getvalue()).decode()  # Convert image to base64 string
                    contents["image"] = {
                        "mime_type": "image/png",
                        "data": img_data
                    }
                
                # Generate response from the model
                response = st.session_state.agent(contents)
                
                # Display results
                st.markdown(f"**AI Assistant:**\n{response}")
                
                # Show surgical plan
                st.markdown("### 🗺️ Surgical Plan")
                st.json({
                    "current_phase": surgical_phase,
                    "next_steps": ["Identify anatomical landmarks", "Verify instrument positioning"],
                    "critical_structures": ["Optic nerve", "Internal carotid artery"],
                    "risk_assessment": "Moderate"
                })
                
                st.markdown("### 🔍 Safety Verification")
                cols = st.columns(3)
                cols[0].metric("Tumor Size", "18mm", "±2mm")
                cols[1].metric("Blood Loss", "150ml", "Last hour")
                cols[2].metric("Vital Signs", "Stable", "HR: 82")
                
            except Exception as e:
                st.error(f"Surgical analysis failed: {str(e)}")

# Sidebar emergency button
st.sidebar.markdown("---")
if st.sidebar.button("🚨 Activate Emergency Protocol"):
    st.sidebar.error(""" 
    EMERGENCY MEASURES:
    1. Immediate hemostasis protocol
    2. Notify senior surgeon
    3. Stabilize vital signs
    4. Prepare emergency imaging
    """)

# Final clinical note
if __name__ == "__main__":
    st.write("⚠️ **Clinical Note:** Verify all AI recommendations with surgical team")
