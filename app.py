import torch
from transformers import BertForSequenceClassification, AutoTokenizer
from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
from pickle import load
import uvicorn
import threading
with open('mapping.pkl','rb') as file:
    mapping = load(file)
num_classes = len(mapping.keys())
inv_mapping = {v: k for k, v in mapping.items()}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'bert-base-uncased'
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
tokenizer = AutoTokenizer.from_pretrained(model_name)

state_dict = torch.load("trained_model.pt", map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

def predict(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    
    pred_idx = torch.argmax(outputs.logits, dim=1).item()

    label = inv_mapping[pred_idx]
    
    return label


app = FastAPI()

class PredictionRequest(BaseModel):
    text: str

@app.post("/predict")
async def get_prediction(request: PredictionRequest):
    prediction = predict(request.text)
    return {"prediction": prediction}


def gradio_predict(text):
    return predict(text)

gradio_interface = gr.Interface(
    fn=gradio_predict,
    inputs="text",
    outputs="text",
    title="BERT Sequence Classification",
    description="Nhập vào một câu và nhận dự đoán nhãn của model."
)

if __name__ == "__main__":
    def run_fastapi():
        uvicorn.run(app, host="0.0.0.0", port=8000)

    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()

    gradio_interface.launch(share=True,inbrowser=True)
