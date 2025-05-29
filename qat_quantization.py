import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import logging
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Set Hugging Face cache directory to current directory
cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_cache")
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["HF_HOME"] = cache_dir
os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_dir, "datasets")

# Create cache directory if it doesn't exist
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(os.path.join(cache_dir, "datasets"), exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Phi2Quantizer:
    def __init__(self, model_name="microsoft/phi-2", dataset_name="medmcqa"):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer with explicit cache_dir
        logger.info(f"Loading model {model_name} and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=cache_dir
        )
        
        # Prepare model for quantization
        self._prepare_model_for_quantization()
        
    def _prepare_model_for_quantization(self):
        """Prepare the model for quantization using bitsandbytes"""
        logger.info("Preparing model for quantization...")
        
        # Configure LoRA for efficient fine-tuning
        lora_config = LoraConfig(
            r=16,  # rank
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "dense"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, lora_config)
        
        # Move model to device
        self.model.to(self.device)
        
    def prepare_dataset(self):
        """Prepare the dataset for training"""
        logger.info(f"Loading dataset {self.dataset_name}...")
        dataset = load_dataset(self.dataset_name)
        
        def preprocess_function(examples):
            questions = examples["question"]
            options = [examples["opa"], examples["opb"], examples["opc"], examples["opd"]]
            answers = examples["cop"]
            
            prompts = []
            for q, opts, ans in zip(questions, options, answers):
                prompt = f"Question: {q}\nOptions:\nA. {opts[0]}\nB. {opts[1]}\nC. {opts[2]}\nD. {opts[3]}\nAnswer: {['A','B','C','D'][ans]}"
                prompts.append(prompt)
            
            # Tokenize
            tokenized = self.tokenizer(
                prompts,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Create labels (same as input_ids for causal LM)
            tokenized["labels"] = tokenized["input_ids"].clone()
            
            return tokenized
        
        # Process dataset
        processed_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        
        return processed_dataset
    
    def train(self, num_epochs=1, max_steps=100):
        """Train the model with quantization-aware training"""
        logger.info("Starting QAT training...")
        
        # Prepare dataset
        dataset = self.prepare_dataset()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./phi2_qat_output",
            num_train_epochs=num_epochs,
            max_steps=max_steps,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            save_strategy="steps",
            save_steps=50,
            evaluation_strategy="steps",
            eval_steps=50,
            load_best_model_at_end=True,
            report_to="none"
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            tokenizer=self.tokenizer
        )
        
        # Train
        trainer.train()
        
        # Save the model
        trainer.save_model("./phi2_qat_model")
        logger.info("Training completed and model saved!")
        
    def evaluate(self, dataset=None, limit=200):
        """Evaluate the model on the validation set"""
        logger.info("Evaluating model...")
        
        if dataset is None:
            dataset = self.prepare_dataset()
        
        self.model.eval()
        predictions = []
        references = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataset["validation"]):
                if i >= limit:
                    break
                
                # Prepare input
                inputs = self.tokenizer(
                    batch["question"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                # Generate prediction
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=10,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                # Decode prediction
                pred_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                answer_letter = pred_text.strip().upper().split("Answer:")[-1].strip()[0]
                
                # Map prediction to index
                label_map = {"A": 0, "B": 1, "C": 2, "D": 3}
                if answer_letter in label_map:
                    predictions.append(label_map[answer_letter])
                else:
                    predictions.append(None)
                
                references.append(batch["cop"])
        
        # Calculate accuracy
        filtered = [(p, r) for p, r in zip(predictions, references) if p is not None]
        predictions, references = zip(*filtered) if filtered else ([], [])
        
        accuracy = accuracy_score(references, predictions) if predictions else 0
        return accuracy

def main():
    # Initialize quantizer
    quantizer = Phi2Quantizer()
    
    # Evaluate original model
    original_accuracy = quantizer.evaluate()
    logger.info(f"Original model accuracy: {original_accuracy:.2%}")
    
    # Train with QAT
    quantizer.train(num_epochs=1, max_steps=100)
    
    # Evaluate quantized model
    quantized_accuracy = quantizer.evaluate()
    logger.info(f"Quantized model accuracy: {quantized_accuracy:.2%}")
    
    # Compare accuracies
    delta = quantized_accuracy - original_accuracy
    logger.info(f"Accuracy change: {delta:+.2%}")

if __name__ == "__main__":
    main()
