from pathlib import Path
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse


# -----------------------------------
# FastAPI App
# -----------------------------------

app = FastAPI(
    title="Text Summarizer App",
    description="Text Summarization using T5",
    version="1.0"
)


# -----------------------------------
# Model Path
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(BASE_DIR)

print("Loading tokenizer...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)

print("Loading T5 model...")
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)


# -----------------------------------
# Device
# -----------------------------------

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

model.to(device)
model.eval()


# -----------------------------------
# HTML Templates
# -----------------------------------

templates = Jinja2Templates(
    directory=str(BASE_DIR)
)


# -----------------------------------
# Request Model
# -----------------------------------

class DialogueInput(BaseModel):
    dialogue: str


# -----------------------------------
# Clean Text
# -----------------------------------

def clean_data(text: str) -> str:

    text = re.sub(r"\r\n", " ", text)

    text = re.sub(r"\s+", " ", text)

    text = re.sub(r"<.*?>", " ", text)

    text = text.strip().lower()

    return text


# -----------------------------------
# Summarization
# -----------------------------------

def summarize_dialogue(dialogue: str) -> str:

    dialogue = clean_data(dialogue)

    # T5 summarization task prefix
    dialogue = "summarize: " + dialogue

    # Tokenize
    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )

    # Move tensors to GPU/CPU
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # Generate summary
    with torch.inference_mode():

        targets = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=200,
            min_length=30,
            num_beams=4,
            length_penalty=2.0,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

    # Decode
    summary = tokenizer.decode(
        targets[0],
        skip_special_tokens=True
    )

    return summary

# -----------------------------------
# Summarization API
# -----------------------------------

@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):

    summary = summarize_dialogue(
        dialogue_input.dialogue
    )

    return {
        "summary": summary
    }


# -----------------------------------
# Home Page
# -----------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )