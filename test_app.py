import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import pandas as pd
import io
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# 1. Define the IntegratedClassifier class exactly as it was defined in the Colab notebook.
class IntegratedClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(IntegratedClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.classifier(x)

# Set device to CPU for Streamlit deployment flexibility
device = torch.device('cpu')

# 2. Load the tokenizer, fine-tuned BERT, StyleDistance, and ZSL integrated models
@st.cache_resource
def load_models():
    # Load BERT Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Load Fine-tuned BERT Model for embeddings
    bert_model_path = "./data/human_v_machine_bert_finetuned/checkpoint-50469" # path to fine-tuned BERT checkpoint
    bert_model = AutoModelForSequenceClassification.from_pretrained(bert_model_path, num_labels=2)
    bert_model.to(device)
    bert_model.eval()

    # Load style model StyleDistance
    style_model = SentenceTransformer('StyleDistance/styledistance')
    style_model.to(device)

    # Load ZSL Integrated Model
    zsl_model_path = './data/human_v_machine_models/integrated_classifier_zsl_best.pth'
    zsl_input_dim = 1536
    zsl_output_dim = 2 # Binary classification (human/machine)
    zsl_model = IntegratedClassifier(zsl_input_dim, zsl_output_dim).to(device)
    zsl_model.load_state_dict(torch.load(zsl_model_path, map_location=device))
    zsl_model.eval()

    return tokenizer, bert_model, style_model, zsl_model

tokenizer, bert_model, style_model, zsl_model = load_models()

# 3. Helper function to generate BERT embeddings
def get_bert_embedding(texts, model, tokenizer, device, max_length=512):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.bert(**inputs, output_hidden_states=True)
        # For batch processing, we return the CLS embeddings directly
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return cls_embeddings

# 4. Helper function for generating style_embedding
def get_style_embedding(texts, style_model):
    # Encode texts using the style model
    embeddings = style_model.encode(texts, convert_to_tensor=True)
    return embeddings.cpu().numpy()

# Streamlit UI
st.set_page_config(page_title="Human vs. Machine Text Classifier", layout="wide")
st.title("🤖 Human vs. Machine Text Classifier")
#st.markdown("Upload text to get predictions from fine-tuned BERT and ZSL integrated models.")

# Input method selection
input_method = st.radio(
    "Select Input Method:",
    ('Single Text Input', 'Upload CSV File')
)

if input_method == 'Single Text Input':
    user_text = st.text_area("Enter text here:", height=250, placeholder="Paste the text you want to analyze...")

    if st.button("Get Predictions"):
        if user_text:
            with st.spinner("Generating embeddings and predicting..."):
                # Generate BERT embedding
                # Pass a list for single text input to match batch processing functions
                bert_embedding = get_bert_embedding([user_text], bert_model, tokenizer, device)

                # Generate style embedding
                style_embedding = get_style_embedding([user_text], style_model)

                # Concatenate embeddings for integrated models
                integrated_embedding = np.concatenate((bert_embedding, style_embedding), axis=1) # axis=1 for batch
                integrated_embedding_tensor = torch.tensor(integrated_embedding, dtype=torch.float32).to(device)

                # ZSL Model Prediction
                zsl_output = zsl_model(integrated_embedding_tensor)
                zsl_probs = torch.softmax(zsl_output, dim=1)
                zsl_pred_idx = torch.argmax(zsl_probs, dim=1).item()
                zsl_prediction = "Machine-generated" if zsl_pred_idx == 1 else "Human-written"
                zsl_confidence = zsl_probs[0, zsl_pred_idx].item()

            st.subheader("Prediction Results:")
            st.write(f"**Model:** {zsl_prediction} (Confidence: {zsl_confidence:.2f})")

        else:
            st.warning("Please enter some text to get predictions.")

elif input_method == 'Upload CSV File':
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            # Read the CSV file into a Pandas DataFrame
            df_uploaded = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode("utf-8")))

            # Validate columns
            required_columns = ['text', 'labels']
            if not all(col in df_uploaded.columns for col in required_columns):
                st.error(f"Error: The uploaded CSV must contain '{required_columns[0]}' and '{required_columns[1]}' columns.")
            else:
                st.success("CSV file uploaded successfully!")
                st.write("Preview of uploaded data:")
                st.dataframe(df_uploaded.head())

                if st.button("Process CSV and Get Batch Predictions"):
                    if not df_uploaded.empty:
                        all_texts = df_uploaded['text'].tolist()
                        actual_labels = df_uploaded['labels'].tolist()
                        batch_size = 32 # Can be adjusted

                        zsl_predictions = []

                        progress_text = "Operation in progress. Please wait."
                        my_bar = st.progress(0, text=progress_text)

                        for i in tqdm(range(0, len(all_texts), batch_size), desc="Processing batches"):
                            my_bar.progress(int((i / len(all_texts)) * 100))
                            batch_texts = all_texts[i:i+batch_size]

                            # Generate BERT embeddings
                            batch_bert_embeddings = get_bert_embedding(batch_texts, bert_model, tokenizer, device)

                            # Generate StyleDistance embeddings
                            batch_style_embeddings = get_style_embedding(batch_texts, style_model)

                            # Concatenate embeddings
                            batch_integrated_embeddings = np.concatenate((batch_bert_embeddings, batch_style_embeddings), axis=1)
                            batch_integrated_embeddings_tensor = torch.tensor(batch_integrated_embeddings, dtype=torch.float32).to(device)

                            # ZSL Model Batch Prediction
                            with torch.no_grad():
                                zsl_batch_output = zsl_model(batch_integrated_embeddings_tensor)
                                zsl_batch_pred_indices = torch.argmax(zsl_batch_output, dim=1).cpu().numpy()
                                zsl_predictions.extend(zsl_batch_pred_indices)

                        my_bar.progress(100)
                        st.success("Batch processing complete!")

                        # Calculate and display accuracy
                        zsl_accuracy = accuracy_score(actual_labels, zsl_predictions)

                        st.subheader("Batch Prediction Results:")
                        st.write(f"**Model Accuracy:** {zsl_accuracy:.4f}")

                        # Display a preview of actual vs. predicted labels
                        results_df = pd.DataFrame({
                            'Text': all_texts,
                            'Actual Label': actual_labels,
                            'Predicted Label': zsl_predictions
                        })
                        st.write("Preview of Actual vs. Predicted Labels:")
                        st.dataframe(results_df.head(10)) # Show first 10 rows

                    else:
                        st.warning("The uploaded CSV file is empty.")

        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            st.error("Please ensure the file is a valid CSV and correctly formatted.")


st.info("Note: BERT and StyleDistance models are loaded once and cached for performance. Predictions for 'Human-written' are represented by 0 and 'Machine-generated' by 1.")
st.markdown("--- --- --- ---")
st.markdown("Developed as part of the Human vs. Machine Text Classification project.")