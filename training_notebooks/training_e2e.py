import os
cache_dir = "/workspace/huggingface_cache"
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

from datasets import load_dataset, Dataset, Audio, DatasetDict, config
import pandas as pd
from tqdm import tqdm
from librosa import load, get_duration
from tqdm.notebook import tqdm
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizer, WhisperProcessor, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch
import torch
from torch.utils.data import IterableDataset
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate

from huggingface_hub import notebook_login, login

login("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")

from huggingface_hub import whoami

# Get user info
user_info = whoami()
print(user_info)

train_dataset = load_dataset("kreasof-ai/be-en-IWSLT2025", split="train+val") # Todo: add train+val
# train_dataset1 = load_dataset("kreasof-ai/be-en-IWSLT2025",split="train", streaming=True) # Todo: add train+val

test_dataset = load_dataset("kreasof-ai/be-en-IWSLT2025", split="test")

train_dataset = train_dataset.remove_columns(["sentence", "speaker_id"])

test_dataset = test_dataset.remove_columns(["sentence", "speaker_id"])

# Preprocessing Dataset
train_dataset_cleaned = train_dataset.filter(lambda x: x["translation"] is not None and x["audio"] is not None)
test_dataset_cleaned = test_dataset.filter(lambda x: x["translation"] is not None and x["audio"] is not None)

# Prepare Dataset
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-medium")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-medium", language="en", task="translate")

processor = WhisperProcessor.from_pretrained("openai/whisper-medium", language="en", task="translate")

def prepare_dataset(batch):
    audio = batch["audio"]

    #compute log-Me1 input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["translation"],
                                max_length=448,  # Ensure target length is within model limits
                                truncation=True).input_ids
    return batch


train_converted = train_dataset_cleaned.map(prepare_dataset, remove_columns=train_dataset_cleaned.column_names, num_proc=1)
test_converted = test_dataset_cleaned.map(prepare_dataset, remove_columns=test_dataset_cleaned.column_names, num_proc=1)


# Prepare For Training
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")
model.generation_config.language = "en"
model.generation_config.task = "translate"

# model.generation_config.forced_decoder_ids = None

import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

metric_bleu = evaluate.load("sacrebleu")
metric_chrf = evaluate.load("chrf")
metric_wer = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric_wer.compute(predictions=pred_str, references=label_str)

    # bleu = metric_bleu.compute(predictions=pred_str, references=label_str)
    # bleu = round(bleu["score"], 2)

    # chrf = metric_chrf.compute(predictions=pred_str, references=label_str)
    # chrf = round(chrf["score"], 2)

    return {"wer": wer}

# adjust the variable below
# TODO: adjust the variable below
hf_repo = "kreasof-ai"
output_dir = "whisper-medium-bem2en-end2end"

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,  # Change this to your desired output directory
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,  # Increase by 2x for every 2x decrease in batch size
    learning_rate=1e-4,
    warmup_ratio=0.03,
    num_train_epochs=3,
    gradient_checkpointing=False,

    fp16=True,
    hub_model_id=f"{hf_repo}/{output_dir}",
    # Changed to evaluate at the end of each epoch
    evaluation_strategy="epoch",

    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,

    # Save best model based on WER metric at the end of each epoch
    save_strategy="epoch",
    save_total_limit=2,  # Keep only the last 2 checkpoints
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,  # Lower WER is better, for CHRF greater is better

    # More frequent logging for better monitoring
    logging_steps=50,  # Adjust if needed
    report_to=["tensorboard"],

    push_to_hub=True,  # Upload model to Hugging Face Hub if needed
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_converted,
    eval_dataset=test_converted,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

trainer.train()
trainer.push_to_hub(commit_message="Finish training whisper medium for end-to-end speech-translation systems") # TODO: add commit message