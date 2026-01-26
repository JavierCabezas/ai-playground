# app.py
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from trainer import TrainerWrapper, MODEL_DIR

app = FastAPI()

def bootstrap():
    """
    Ensure model checkpoint exists.
    If not, run training routine automatically.
    """
    Path("models").mkdir(parents=True, exist_ok=True)
    
    if (Path("models") / "model.pth").exists():
        print(f"[bootstrap] Found existing checkpoint at models")
    else:
        print("[bootstrap] No model found - starting training …")
        trainer = TrainerWrapper()
        trainer.train()


# ------------------------------------------------------------------
# 1️⃣ Bootstrap on startup
bootstrap()

# 2️⃣ Load tokenizer / model (now guaranteed to exist)
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model     = AutoModelForQuestionAnswering.from_pretrained(str(MODEL_DIR))

@app.post("/qa")
async def answer(question: str, context: str):
    """
    Very small QA endpoint - returns the best answer span.
    """
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        max_length=384,
    )
    outputs = model(**inputs)
    start_logits = outputs.start_logits
    end_logits   = outputs.end_logits

    # Pick the most probable span (simplified)
    answer_start = int(start_logits.argmax())
    answer_end   = int(end_logits.argmax()) + 1
    answer_tokens = inputs["input_ids"][0][answer_start:answer_end]
    answer_text = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    return {"question": question, "answer": answer_text}

# ------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
