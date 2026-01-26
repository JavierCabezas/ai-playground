# trainer.py
import os
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    Trainer,
    TrainingArguments,
)

MODEL_DIR = Path("models")

# In trainer.py
class TrainerWrapper:
    # ...

    def train(self):
        """
        Fine-tunes a pre-trained language model on a SQuAD-style dataset.
        
        Returns:
            bool: Whether training was successful
        """

        # 1. Load dataset with proper error handling
        try:
            raw_ds = load_dataset("squad_v2", split="train[:5%]")
        except Exception as e:
            print(f"Failed to load SQuAD v2 dataset: {str(e)}")
            return False

        # 2. Preprocessing pipeline with better error handling
        def preprocess(example):
            try:
                return self.base_tokenizer(
                    example["question"],
                    example["context"],
                    truncation=True,
                    padding="max_length",
                    max_length=384,
                )
            except KeyError as e:
                print(f"Missing key in example: {str(e)}")
                return None
            except Exception as e:
                print(f"Preprocessing error: {str(e)}")
                return None

        # 3. Map preprocessing function with batch size control
        batch_size = 32
        tokenized_ds = raw_ds.map(
            preprocess,
            batched=True,
            num_proc=4,  # Adjust for your CPU cores
            remove_columns=["title", "id"],  # Remove unnecessary columns
        )

        # 4. Format dataset with proper type and column mapping
        try:
            tokenized_ds.set_format(
                type="torch",
                columns=['id', 'title', 'context', 'question', 'answers'],
            )
        except Exception as e:
            print(f"Failed to format dataset: {str(e)}")
            return False

        # 5. Training arguments with more informative logging
        training_args = TrainingArguments(
            output_dir=str(MODEL_DIR),
            num_train_epochs=3,
            per_device_train_batch_size=batch_size,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="no",
            overwrite_output_dir=True,
            log_level="warning",  # Adjust for your environment
        )

        # 6. Trainer initialization with proper error handling
        try:
            trainer = Trainer(
                model=self.base_model,
                args=training_args,
                train_dataset=tokenized_ds,
                compute_metrics=lambda pred: {
                    "accuracy": torch.sum(torch.logical_and(pred.label_ids == pred.predictions.argmax(-1), pred.predictions.argmax(-1) != -100)) / len(pred),
                    "loss": trainer.compute_loss(pred)
                }
            )
        except Exception as e:
            print(f"Failed to create trainer: {str(e)}")
            return False

        # 7. Training loop with progress tracking
        try:
            trainer.train()
            print("Training completed successfully.")
            return True
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
            return False
        except Exception as e:
            print(f"Training error: {str(e)}")
            return False

        # 8. Cleanup after training (optional but good practice)
        finally:
            if Path("models/model.pth").exists():
                Path("models/model.pth").unlink()
