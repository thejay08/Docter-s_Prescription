from __future__ import annotations
import base64
import os
from keys import GEMINI_API_KEY
from typing import List
from datetime import datetime
from langchain.chains import TransformChain
from langchain_core.runnables import chain
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import streamlit as st
import pandas as pd
import shutil
import google.generativeai as genai
from PIL import Image
from io import BytesIO

if not GEMINI_API_KEY:
    st.error("⚠️ Error: Google Gemini API Key is missing! Set it in keys.py")
else:
    genai.configure(api_key=GEMINI_API_KEY)

class MedicationItem(BaseModel):
    name: str
    dosage: str
    frequency: str
    duration: str

class PrescriptionInformations(BaseModel):
    patient_name: str = Field(description="Patient's name")
    patient_age: int = Field(description="Patient's age")
    patient_gender: str = Field(description="Patient's gender")
    doctor_name: str = Field(description="Doctor's name")
    doctor_license: str = Field(description="Doctor's license number")
    prescription_date: datetime = Field(description="Date of the prescription")
    medications: List[MedicationItem] = []
    additional_notes: str = Field(description="Additional notes or instructions")

def load_images(inputs: dict) -> dict:
    image_paths = inputs["image_paths"]
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    images_base64 = [encode_image(image_path) for image_path in image_paths]
    return {"images": images_base64}

load_images_chain = TransformChain(
    input_variables=["image_paths"],
    output_variables=["images"],
    transform=load_images
)

@chain
def image_model(inputs: dict) -> dict:
    try:
        img_base64 = inputs['images'][0]
        img_bytes = base64.b64decode(img_base64)
        img_stream = BytesIO(img_bytes)
        pil_img = Image.open(img_stream)
    except Exception as e:
        return {"error": f"Error decoding image: {str(e)}"}
    
    prompt = """
    Extract the following details from the prescription image:
    - Patient's full name, age, gender
    - Doctor's name, license number
    - Prescription date
    - List of medicines with name, dosage, frequency, and duration
    - Additional notes or instructions
    Return output as structured JSON.
    """
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content([pil_img, prompt])
        return response.text if isinstance(response.text, str) else "{}"
    except Exception as e:
        return {"error": f"Error generating content: {str(e)}"}

def get_prescription_informations(image_paths: List[str]) -> dict:
    parser = JsonOutputParser(pydantic_object=PrescriptionInformations)
    vision_chain = load_images_chain | image_model | parser
    return vision_chain.invoke({'image_paths': image_paths})

def main():
    st.title('Medical Prescription')
    uploaded_file = st.file_uploader("Upload a Prescription image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = uploaded_file.name.split('.')[0].replace(' ', '_')
        output_folder = os.path.join(".", f"Check_{filename}_{timestamp}")
        os.makedirs(output_folder, exist_ok=True)
        check_path = os.path.join(output_folder, uploaded_file.name)
        with open(check_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.expander("Prescription Image", expanded=False):
            st.image(uploaded_file, caption='Uploaded Prescription Image.', use_container_width=True)
        
        with st.spinner('Processing Prescription...'):
            final_result = get_prescription_informations([check_path])
            if isinstance(final_result, PrescriptionInformations):
                final_result = final_result.dict()
            elif not isinstance(final_result, dict):
                st.error(f"Unexpected output: {final_result}")
                return
            
            total_medicines = len(final_result.get('medicines', []))
            st.subheader(f"Total Medicines: {total_medicines}")
            
            data = [(key, final_result[key]) for key in final_result if key != 'medicines']
            df = pd.DataFrame(data, columns=["Field", "Value"])
            st.write(df.to_html(classes='custom-table', index=False, escape=False), unsafe_allow_html=True)
            
            if total_medicines > 0:
                medications_df = pd.DataFrame(final_result['medicines'])
                st.subheader("Medicines")
                st.write(medications_df.to_html(classes='custom-table', index=False, escape=False), unsafe_allow_html=True)
        shutil.rmtree(output_folder)

if __name__ == "__main__":
    main()
