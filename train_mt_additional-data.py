import os
cache_dir = "/content/huggingface_cache"
os.makedirs(cache_dir, exist_ok=True)

# Set ALL Hugging Face related cache directories
os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_dir, "transformers")
os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_dir, "datasets")
os.environ["HF_HOME"] = os.path.join(cache_dir, "hf_home")
os.environ["HF_ASSETS_CACHE"] = os.path.join(cache_dir, "assets")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(cache_dir, "hub")
os.environ["HF_MODULES_CACHE"] = os.path.join(cache_dir, "modules")

# Create all directories
for dir_path in [os.environ["TRANSFORMERS_CACHE"],
                os.environ["HF_DATASETS_CACHE"],
                os.environ["HF_HOME"],
                os.environ["HF_ASSETS_CACHE"],
                os.environ["HUGGINGFACE_HUB_CACHE"],
                os.environ["HF_MODULES_CACHE"]]:
    os.makedirs(dir_path, exist_ok=True)

# Force datasets to use the new cache
from datasets import config
config.HF_DATASETS_CACHE = os.environ["HF_DATASETS_CACHE"]


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset, Dataset, Audio, DatasetDict, concatenate_datasets
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import evaluate
import pandas as pd
from huggingface_hub import login
import os

os.environ["WANDB_DISABLED"] = "true"
login("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")

model_name = "facebook/nllb-200-3.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

train_dataset = load_dataset("kreasof-ai/be-en-IWSLT2025",split="train+val")
# train_dataset = load_dataset("kreasof-ai/be-en-IWSLT2025",split="train", streaming=True)

test_dataset = load_dataset("kreasof-ai/be-en-IWSLT2025", split="test")

train_dataset = train_dataset.remove_columns(["audio", "speaker_id"])  # Remove audio
test_dataset = test_dataset.remove_columns(["audio", "speaker_id"])

df_train = train_dataset.to_pandas()
df_test = test_dataset.to_pandas()

df_train = df_train.drop_duplicates(subset=["sentence"], keep="first").reset_index(drop=True)
df_test = df_test.drop_duplicates(subset=["sentence"], keep="first").reset_index(drop=True)

df_train = df_train[~df_train["translation"].isna()]

cleaned_train = Dataset.from_pandas(df_train)
cleaned_test = Dataset.from_pandas(df_test)

train_df = pd.read_csv("bt_tatoeba_en_bem_2.csv")
train_dataset_augmented = Dataset.from_pandas(train_df)
train_dataset_augmented = train_dataset_augmented.rename_column("text_en", "translation")
train_dataset_augmented = train_dataset_augmented.rename_column("text_bem", "sentence")
train_dataset_concat = concatenate_datasets([cleaned_train, train_dataset_augmented])

df_train = train_dataset_concat.to_pandas()
# shuffle the row
df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
# cleaned_train = Dataset.from_pandas(df_train.head(10)) # For testing purpose
cleaned_train = Dataset.from_pandas(df_train)

metric_bleu = evaluate.load("sacrebleu")
metric_chrf = evaluate.load("chrf")
# metric_wer = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # wer = 100 * metric_wer.compute(predictions=pred_str, references=label_str)

    bleu = metric_bleu.compute(predictions=pred_str, references=label_str)
    bleu = round(bleu["score"], 2)

    chrf = metric_chrf.compute(predictions=pred_str, references=label_str)
    chrf = round(chrf["score"], 2)

    return {"bleu": bleu, "chrf": chrf}

def preprocess_function(dataset):
    # Tokenize the source text (English) and target text (Hindi)
    try:
      model_inputs = tokenizer(
          dataset["sentence"],  # Source text column
          text_target=dataset["translation"],  # Target text column
          max_length=128,  # Adjust max length based on your needs
          truncation=True,  # Truncate inputs that exceed the max length
          padding="max_length"  # Pad inputs to the max length
      )
      return model_inputs
    except Exception as e:
      print(e)

train_tokenized = cleaned_train.map(preprocess_function, batched=True)
test_tokenized = cleaned_test.map(preprocess_function, batched=True)

# adjust the variable below
hf_repo = "kreasof-ai"
output_dir = "nllb-IWSLT2025-bem-en-dum"

# Fine-tune
# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,               # Directory to save checkpoints
    per_device_train_batch_size=8,      # Adjust batch size based on GPU memory
    gradient_accumulation_steps=1,
    learning_rate=1e-4,                   # Learning rate
    warmup_ratio=0.03,
    num_train_epochs=3,                # Train for 100 epochs
    gradient_checkpointing=True,


    fp16=True,
    hub_model_id=f"{hf_repo}/{output_dir}",  # Change this
    evaluation_strategy="epoch",         # Evaluate at the end of each epoch

    per_device_eval_batch_size=8,
    predict_with_generate=True,          # Generate predictions for validation
    generation_max_length=128,

    save_strategy="epoch",               # Save checkpoints at the end of each epoch
    save_total_limit=2,
    load_best_model_at_end=True,         # Load the best model at the end of training
    metric_for_best_model="chrf",        # Use BLEU as the evaluation metric
    greater_is_better=True,
    logging_dir="./logs",                # Log directory
    logging_steps=100,
    report_to=["tensorboard"],
    push_to_hub=True,                    # Push model to Hugging Face Hub

)

# Initialize the trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

trainer.train()
# trainer.train(resume_from_checkpoint=True)
# trainer.train(resume_from_checkpoint="./nllb-600/checkpoint-1000") # resume from specific checkpoint
# if os.path.exists(output_dir):
#     trainer.train(resume_from_checkpoint=True)
# else:
#     trainer.train()


# Push the final model to Hugging Face Hub
trainer.push_to_hub()