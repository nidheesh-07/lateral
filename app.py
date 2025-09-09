import gradio as gr
import torch
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq
import pandas as pd
import requests

# Placeholder for Hugging Face API token or key if needed for private models
HUGGINGFACE_TOKEN = "YOUR_HUGGINGFACE_TOKEN" # Replace with your token if needed

# --- 1. Load the Speech Recognition Model ---
# Note: The model "ibm-granite/granite-speech-3.3-2b" is not publicly available on Hugging Face.
# This code uses a placeholder model "openai/whisper-tiny".
# You must replace it with your specific model if it becomes available or is a private model.
# Ensure you have the correct access rights and authentication if required.

ASR_MODEL_NAME = "openai/whisper-tiny" # Replace with your model name if different

try:
    asr_pipe = pipeline("automatic-speech-recognition", model=ASR_MODEL_NAME)
    print("ASR pipeline loaded successfully.")
except Exception as e:
    print(f"Error loading ASR model: {e}")
    asr_pipe = None
    print("ASR model could not be loaded. Please ensure the model name is correct and you have internet access.")


# --- 2. Drug Data (Placeholder) ---
# In a real-world scenario, you would use a robust dataset or API.
# This is a simplified in-memory "database" for demonstration.
DRUG_DATA = {
    "acetaminophen": {
        "dosage": "500mg every 4-6 hours for adults",
        "age_restrictions": "Safe for all ages, consult doctor for children.",
        "interactions": ["warfarin (increased bleeding risk)"],
        "alternatives": ["ibuprofen", "aspirin"],
    },
    "ibuprofen": {
        "dosage": "200-400mg every 4-6 hours for adults",
        "age_restrictions": "Not recommended for children under 6 months.",
        "interactions": ["ibuprofen", "aspirin", "blood thinners"],
        "alternatives": ["acetaminophen", "naproxen"],
    },
    "amoxicillin": {
        "dosage": "250-500mg every 8 hours for adults",
        "age_restrictions": "Dosage is weight-based for children.",
        "interactions": ["methotrexate", "birth control pills"],
        "alternatives": ["cephalexin", "doxycycline"],
    },
}

# --- 3. Core Logic for Drug Analysis ---
def get_drug_info(drug_name: str) -> dict:
    """Fetches drug information from a mock database."""
    return DRUG_DATA.get(drug_name.lower(), None)

def check_interactions(drugs_list: list) -> list:
    """Checks for interactions between a list of drugs."""
    interactions_found = []

    # Check for interactions within the provided list of drugs
    for i in range(len(drugs_list)):
        for j in range(i + 1, len(drugs_list)):
            drug1_info = get_drug_info(drugs_list[i])
            drug2_info = get_drug_info(drugs_list[j])

            if drug1_info and drug2_info:
                if drugs_list[j] in [d.lower() for d in drug1_info.get("interactions", [])]:
                    interactions_found.append(f"Interaction warning: {drugs_list[i]} and {drugs_list[j]} should not be taken together.")
                if drugs_list[i] in [d.lower() for d in drug2_info.get("interactions", [])]:
                    interactions_found.append(f"Interaction warning: {drugs_list[j]} and {drugs_list[i]} should not be taken together.")

    return interactions_found

def get_alternatives(drug_name: str, age: int) -> str:
    """Provides safe alternatives based on age."""
    info = get_drug_info(drug_name)
    if info:
        alternatives = info.get("alternatives", [])

        # Filter alternatives based on age (simplified logic)
        safe_alternatives = []
        for alt in alternatives:
            alt_info = get_drug_info(alt)
            if alt_info:
                if "children" in alt_info["age_restrictions"] and age < 12:
                    continue  # Skip if not safe for children
                safe_alternatives.append(alt)

        if safe_alternatives:
            return f"Safe alternatives for {drug_name} at your age: {', '.join(safe_alternatives)}"
        else:
            return f"No safe alternatives found for {drug_name} at your age."
    return f"Drug '{drug_name}' not found in our database."

# --- 4. Main Gradio Function ---
def drug_advisor(audio_file, drug_list_input, age_input):
    """
    Main function to process audio, text, and provide drug advice.
    """
    transcribed_text = ""
    if audio_file is not None and asr_pipe is not None:
        try:
            transcribed_text = asr_pipe(audio_file)["text"]
            # Clean up the transcribed text
            transcribed_text = transcribed_text.lower().strip()
        except Exception as e:
            return f"Error transcribing audio: {e}", "", "", ""

    # Process drug list from either transcription or text input
    if transcribed_text:
        drugs = [d.strip() for d in transcribed_text.split() if d.strip() in DRUG_DATA]
    elif drug_list_input:
        drugs = [d.strip().lower() for d in drug_list_input.split(',')]
    else:
        return "", "", "Please provide drug names either via audio or text input.", ""

    # Initialize outputs
    dosage_info = ""
    interactions_output = ""
    alternatives_output = ""

    # 1. Provide dosage information
    for drug in drugs:
        info = get_drug_info(drug)
        if info:
            dosage_info += f"**{drug.capitalize()}**\nDosage: {info['dosage']}\nAge Restrictions: {info['age_restrictions']}\n\n"
        else:
            dosage_info += f"**{drug.capitalize()}**\nWarning: Drug not found in database.\n\n"

    # 2. Check for interactions
    if len(drugs) > 1:
        interactions = check_interactions(drugs)
        if interactions:
            interactions_output = "\n".join(interactions)
        else:
            interactions_output = "No major interactions found among the specified drugs."
    else:
        interactions_output = "Please enter more than one drug to check for interactions."

    # 3. Provide alternatives
    if drugs and age_input:
        # For simplicity, we'll provide alternatives for the first drug in the list
        alternatives_output = get_alternatives(drugs[0], age_input)

    return transcribed_text, dosage_info, interactions_output, alternatives_output

# --- 5. Gradio Interface ---
# Define the Gradio inputs
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 💊 Gen AI Drug Interaction & Safety Advisor")
    gr.Markdown("Speak the drug names or enter them manually to get information on dosages, interactions, and safe alternatives.")

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Speak Drug Names")
            text_input = gr.Textbox(label="Enter Drug Names (comma-separated)", placeholder="e.g., acetaminophen, ibuprofen, amoxicillin")
            age_input = gr.Slider(minimum=1, maximum=120, value=30, label="Your Age")
            submit_button = gr.Button("Analyze Drugs", variant="primary")

        with gr.Column():
            transcription_output = gr.Textbox(label="Transcription", interactive=False)
            dosage_output = gr.Markdown("### 📝 Dosage Information")
            interactions_output = gr.Markdown("### ⚠️ Drug Interactions")
            alternatives_output = gr.Markdown("### 💡 Safe Alternatives")

    # Connect the components
    submit_button.click(
        fn=drug_advisor,
        inputs=[audio_input, text_input, age_input],
        outputs=[transcription_output, dosage_output, interactions_output, alternatives_output]
    )

if __name__ == "__main__":
    demo.launch()
