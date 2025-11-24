# Fine-tuned BERT Model for NER on Music Dataset

## Overview
This notebook demonstrates how to fine-tune a pre-trained BERT model for Named Entity Recognition (NER) on a custom music dataset. The goal is to identify music-related entities such as Artists and Works of Art (WoA) from user queries.

---

## Main Steps

1. **Data Loading and Preprocessing** - Load the music NER dataset and clean labels
2. **Entity Extraction with spaCy** - Process text and create spaCy documents with entity spans
3. **BIO Format Parsing** - Convert data from BIO format and extract entity information
4. **Train/Test Split** - Split the dataset into training and testing sets
5. **Dataset Preparation** - Create HuggingFace Dataset objects with proper features
6. **Tokenization and Label Alignment** - Tokenize text and align NER labels with subword tokens
7. **Model Setup** - Initialize BERT model for token classification
8. **Training** - Fine-tune the model on the training data
9. **Evaluation** - Evaluate model performance on test data
10. **Model Persistence** - Save and load the trained model
11. **Inference** - Test the model on new examples

---

## Detailed Code Explanation

### Cell 1: Notebook Title
```markdown
# Fine-tuned BERT Model for NER on Music Dataset
```
**Purpose**: Documentation cell that describes the notebook's objective.

---

### Cell 2: Import Utility Functions
```python
%run -i "../util/lang_utils.ipynb"
```
**Explanation**:
- `%run -i` is a Jupyter magic command that executes another notebook
- `-i` flag runs the notebook in the current namespace, making its variables/functions available
- This loads utility functions from `lang_utils.ipynb`, likely including the `small_model` spaCy model used later

---

### Cell 3: Import Libraries
```python
from datasets import (
    load_dataset, Dataset, Features, Value,
    ClassLabel, Sequence, DatasetDict)
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from transformers import DataCollatorForTokenClassification
from transformers import (
    AutoModelForTokenClassification, TrainingArguments, Trainer)
import numpy as np
from sklearn.model_selection import train_test_split
from evaluate import load
```

**Explanation line by line**:
- **Lines 1-3**: Import HuggingFace `datasets` library components:
  - `load_dataset`: Load datasets from HuggingFace Hub or local files
  - `Dataset`: Core dataset class for storing data
  - `Features`: Define schema/structure for datasets
  - `Value`: Represent single values in the schema
  - `ClassLabel`: Represent categorical labels (NER tags)
  - `Sequence`: Represent sequences of values (tokens, tags)
  - `DatasetDict`: Dictionary containing train/test splits

- **Line 4**: Import pandas for data manipulation and DataFrame operations

- **Line 5**: Import HuggingFace transformers components:
  - `AutoTokenizer`: Automatically load the correct tokenizer for a model
  - `AutoModel`: Load pre-trained transformer models

- **Line 6**: Import `DataCollatorForTokenClassification`:
  - Handles batching and padding for token classification tasks
  - Ensures proper alignment of labels with tokenized inputs

- **Lines 7-8**: Import training components:
  - `AutoModelForTokenClassification`: BERT model with a token classification head
  - `TrainingArguments`: Configure training hyperparameters
  - `Trainer`: High-level API for training and evaluation

- **Line 9**: Import NumPy for numerical operations and array manipulation

- **Line 10**: Import `train_test_split` from scikit-learn to split data into train/test sets

- **Line 11**: Import `load` from `evaluate` library to load evaluation metrics (seqeval for NER)

---

### Cell 4: Load and Clean Data
```python
music_ner_df = pd.read_csv("../data/music_ner.csv")
def change_label(input_label):
    input_label = input_label.replace("_deduced", "")
    return input_label
music_ner_df["label"] = music_ner_df["label"].apply(change_label)
music_ner_df["text"] = music_ner_df["text"].apply(lambda x: x.replace("|", ", "))
music_ner_df.head()
```

**Explanation line by line**:
- **Line 1**: Load the music NER dataset from CSV into a pandas DataFrame
  - Contains columns: id, text, start_offset, end_offset, label

- **Lines 2-4**: Define a function to clean label names:
  - Removes the "_deduced" suffix from label names
  - Example: "Artist_known_deduced" → "Artist_known"

- **Line 5**: Apply the label cleaning function to all labels in the dataset

- **Line 6**: Clean the text column:
  - Replace pipe characters `|` with commas followed by space `, `
  - Improves text readability and formatting

- **Line 7**: Display the first 5 rows to verify the data structure

**Output shows**:
- Each row represents one entity annotation
- Same `id` can appear multiple times (multiple entities in one text)
- `start_offset` and `end_offset` mark entity boundaries
- Labels include: Artist_known, Artist_or_WoA, WoA, Artist

---

### Cell 5: Create spaCy Documents with Entity Spans
```python
# Data preprocessing
ids = list(set(music_ner_df["id"].values))
docs = {}
for id in ids:
    entity_rows = music_ner_df[music_ner_df["id"]==id]
    text = entity_rows.head(1)["text"].values[0]
    doc = small_model(text)
    ents = []
    for _, row in entity_rows.iterrows():
        start = row["start_offset"]
        end = row["end_offset"]
        label = row["label"]
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is not None:
            ents.append(span)
    doc.ents = ents
    docs[doc.text] = doc
```

**Explanation line by line**:
- **Line 2**: Extract unique document IDs from the dataset
  - `set()` removes duplicates, `list()` converts back to list

- **Line 3**: Initialize empty dictionary to store spaCy documents (text → doc mapping)

- **Line 4**: Iterate through each unique document ID

- **Line 5**: Filter DataFrame to get all entity rows for current document ID
  - One document can have multiple entity annotations

- **Line 6**: Extract the text content from the first row
  - All rows with same ID have identical text, so we only need one

- **Line 7**: Process the text using spaCy's `small_model` to create a Doc object
  - Tokenizes and performs linguistic analysis

- **Line 8**: Initialize empty list to store entity spans

- **Line 9**: Iterate through each entity annotation row for this document

- **Lines 10-12**: Extract entity information:
  - `start`: Character offset where entity begins
  - `end`: Character offset where entity ends
  - `label`: Entity type (Artist, WoA, etc.)

- **Line 13**: Create a spaCy Span object using character offsets:
  - `char_span()`: Creates span from character positions
  - `alignment_mode="contract"`: If span doesn't align with token boundaries, shrink it to fit
  - Returns `None` if alignment fails completely

- **Lines 14-15**: Add valid spans to the entity list
  - Filters out `None` values from failed alignments

- **Line 16**: Assign the collected entities to the spaCy document's `.ents` attribute

- **Line 17**: Store the document in the dictionary using its text as the key

**Purpose**: Creates a lookup dictionary to retrieve processed spaCy documents by their text content.

---

### Cell 6: Parse BIO Format Data
```python
# Parses data into tokens; maps NER tags to integers, reconstructs sentences,
# Extracts the predicted entity spans
data_file = "../data/music_ner_bio.bio"
tag_mapping = {"O": 0, "B-Artist": 1, "I-Artist": 2, "B-WoA": 3, "I-WoA": 4}
with open(data_file, "r") as f:
    data = f.read()
tokens = [] # word lists per sentence
ner_tags = [] # integer NER tag lists per sentence
spans = [] # extracted entity spans per sentence
sentences = data.split("\n\n")
for sentence in sentences:
    words = [] # words in this sentence
    tags = [] # integer NER tags in this sentence
    this_sentence_spans = [] # NER spans from model output
    word_tag_pairs = sentence.split("\n") # word-tag pairs, separated by <TAB>
    for pair in word_tag_pairs:
        (word, tag) = pair.split("\t")
        words.append(word)
        tags.append(tag_mapping[tag])
    sentences_text = " ".join(words)
    try:
        doc = docs[sentences_text]
    except:
        pass
    ent_dict = {}
    for ent in doc.ents:
        this_sentence_spans.append(f"{ent.label_}: {ent.text}")
    tokens.append(words)
    ner_tags.append(tags)
    spans.append(this_sentence_spans)
```

**Explanation line by line**:
- **Lines 1-2**: Comments describing what this cell does

- **Line 3**: Path to BIO format file
  - BIO format: Begin-Inside-Outside tagging scheme for NER
  - B- = Beginning of entity, I- = Inside entity, O = Outside (not an entity)

- **Line 4**: Define mapping from BIO tags to integer labels:
  - 0 = O (not an entity)
  - 1 = B-Artist (beginning of Artist entity)
  - 2 = I-Artist (continuation of Artist entity)
  - 3 = B-WoA (beginning of Work of Art entity)
  - 4 = I-WoA (continuation of Work of Art entity)

- **Lines 5-6**: Open and read the entire BIO file content

- **Lines 7-9**: Initialize three lists to store processed data:
  - `tokens`: List of word lists (one per sentence)
  - `ner_tags`: List of integer tag lists (parallel to tokens)
  - `spans`: List of entity span lists (for reference/evaluation)

- **Line 10**: Split file content into individual sentences
  - Sentences are separated by double newlines (`\n\n`)

- **Line 11**: Iterate through each sentence

- **Lines 12-14**: Initialize temporary lists for the current sentence:
  - `words`: Tokens in this sentence
  - `tags`: Integer NER tags for tokens
  - `this_sentence_spans`: Human-readable entity spans

- **Line 15**: Split sentence into word-tag pairs
  - Each line has format: `word\ttag`

- **Line 16**: Iterate through each word-tag pair

- **Line 17**: Split the pair by tab character to extract word and tag

- **Line 18**: Add the word to the words list

- **Line 19**: Convert the BIO tag to integer using the mapping and add to tags list

- **Line 20**: Reconstruct the original sentence by joining words with spaces

- **Lines 21-23**: Try to retrieve the corresponding spaCy document:
  - Uses the docs dictionary created in the previous cell
  - `except pass`: Silently skip if document not found (defensive programming)

- **Line 24**: Initialize empty dictionary (appears unused in the code)

- **Lines 25-26**: Extract entity information from the spaCy document:
  - Iterate through all entities in the document
  - Format as "Label: entity_text" and add to spans list

- **Lines 27-29**: Append the processed sentence data to the main lists:
  - Add words to tokens list
  - Add integer tags to ner_tags list
  - Add entity spans to spans list

**Purpose**: Converts BIO format data into structured lists suitable for training while preserving entity information.

---

### Cell 7: Split Data into Train and Test Sets
```python
# Split data into train and test sets
indices = range(0, len(spans))
train, test = train_test_split(indices, test_size=0.1)
train_tokens = []
train_ner_tags = []
train_spans = []
test_tokens = []
test_ner_tags = []
test_spans = []
for i, (toke, ner_tag, span) in enumerate(zip(tokens, ner_tags, spans)):
    if i in train:
        train_tokens.append(toke)
        train_ner_tags.append(ner_tag)
        train_spans.append(span)
    else:
        test_tokens.append(toke)
        test_ner_tags.append(ner_tag)
        test_spans.append(span)
print(len(train_tokens), len(test_tokens))
```

**Explanation line by line**:
- **Line 2**: Create a range of indices from 0 to the total number of sentences
  - These indices will be split into train/test sets

- **Line 3**: Use scikit-learn's `train_test_split` to randomly split indices:
  - `test_size=0.1`: Allocate 10% for testing, 90% for training
  - Returns two lists: train indices and test indices

- **Lines 4-9**: Initialize six empty lists:
  - Three for training data: tokens, NER tags, and spans
  - Three for testing data: tokens, NER tags, and spans

- **Line 10**: Iterate through all data with enumeration:
  - `zip(tokens, ner_tags, spans)`: Combines the three lists element-wise
  - `enumerate()`: Provides index `i` along with the data

- **Lines 11-14**: If current index is in the training set:
  - Append the data (tokens, tags, spans) to the training lists

- **Lines 15-18**: Otherwise (index is in test set):
  - Append the data to the testing lists

- **Line 19**: Print the sizes of training and testing sets

**Output**: `539 60` indicates 539 training examples and 60 test examples (approximately 90-10 split).

---

### Cell 8: Create DataFrames
```python
training_df = pd.DataFrame({"tokens":train_tokens,
                            "ner_tags":train_ner_tags,
                            "spans":train_spans})
testing_df = pd.DataFrame({"tokens":test_tokens,
                           "ner_tags":test_ner_tags,
                           "spans":test_spans})
training_df["text"] = training_df["tokens"].apply(lambda x: " ".join(x))
testing_df["text"] = testing_df["tokens"].apply(lambda x: " ".join(x))
training_df.dropna(inplace=True)
testing_df.dropna(inplace=True)
training_df.head()
```

**Explanation line by line**:
- **Lines 1-3**: Create a pandas DataFrame for training data with three columns:
  - `tokens`: List of words for each sentence
  - `ner_tags`: List of integer NER tags
  - `spans`: List of entity span strings

- **Lines 4-6**: Create a pandas DataFrame for testing data with the same structure

- **Line 7**: Add a `text` column to training DataFrame:
  - `apply(lambda x: " ".join(x))`: Joins token lists into single strings
  - Converts `['music', 'similar', 'to']` → `'music similar to'`

- **Line 8**: Add the same `text` column to testing DataFrame

- **Line 9**: Remove any rows with missing values from training data:
  - `dropna()`: Drops rows containing NaN/None
  - `inplace=True`: Modifies the DataFrame directly instead of returning a copy

- **Line 10**: Remove rows with missing values from testing data

- **Line 11**: Display the first 5 rows of training DataFrame

**Output shows**:
- Each row is a complete sentence with its annotations
- `tokens`: List of words
- `ner_tags`: Parallel list of integer labels (0-4)
- `spans`: Human-readable entity strings for reference
- `text`: Reconstructed sentence as a single string

---

### Cell 9: Create HuggingFace Dataset Objects
```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
features = Features({
    "tokens": Sequence(feature=Value(dtype="string", id=None),
                       length=-1, id=None),
    "ner_tags": Sequence(feature=ClassLabel(names=["O", "B-Artist", "I-Artist", "B-WoA", "I-WoA"]),
                         length=-1, id=None),
    "spans": Sequence(feature=Value(dtype="string", id=None),
                      length=-1, id=None),
    "text": Value(dtype="string", id=None)
})
training_dataset = Dataset.from_pandas(training_df, features=features)
testing_dataset = Dataset.from_pandas(testing_df, features=features)
dataset = DatasetDict({
    "train": training_dataset,
    "test": testing_dataset
})
print(dataset["train"].features)
label_names = dataset["train"].features["ner_tags"].feature.names
print(dataset)
```

**Explanation line by line**:
- **Line 1**: Load the BERT tokenizer:
  - `AutoTokenizer.from_pretrained()`: Automatically downloads the correct tokenizer
  - `"bert-base-uncased"`: Pre-trained BERT model (uncased = lowercased)
  - Downloads vocab file and tokenizer config from HuggingFace Hub

- **Lines 2-10**: Define the schema (Features) for the dataset:

  - **Lines 3-4**: `tokens` field:
    - `Sequence()`: Indicates this is a variable-length list
    - `Value(dtype="string")`: Each element is a string
    - `length=-1`: Variable length (not fixed)

  - **Lines 5-6**: `ner_tags` field:
    - `Sequence()`: Variable-length list
    - `ClassLabel()`: Each element is a categorical label
    - `names=[...]`: Defines the mapping from integers to label names
    - 0→"O", 1→"B-Artist", 2→"I-Artist", 3→"B-WoA", 4→"I-WoA"

  - **Lines 7-8**: `spans` field:
    - `Sequence()`: List of strings
    - Contains human-readable entity annotations

  - **Line 9**: `text` field:
    - `Value(dtype="string")`: Single string value per example

- **Line 11**: Convert training DataFrame to HuggingFace Dataset:
  - `Dataset.from_pandas()`: Creates Dataset from pandas DataFrame
  - `features=features`: Applies the defined schema

- **Line 12**: Convert testing DataFrame to HuggingFace Dataset

- **Lines 13-16**: Create a DatasetDict to organize train/test splits:
  - Dictionary with "train" and "test" keys
  - Enables easy access: `dataset["train"]` and `dataset["test"]`

- **Line 17**: Print the features of the training dataset to verify schema

- **Line 18**: Extract label names for later use in evaluation:
  - Navigates to the ner_tags feature's ClassLabel names
  - Returns: `["O", "B-Artist", "I-Artist", "B-WoA", "I-WoA"]`

- **Line 19**: Print the complete DatasetDict summary

**Output shows**:
- Training dataset has 539 examples
- Testing dataset has 60 examples
- Four features per example: tokens, ner_tags, spans, text

---

### Cell 10: Tokenize and Align Labels
```python
def tokenize_adjust_labels(all_samples_per_split):
    tokenized_samples = tokenizer.batch_encode_plus(
        all_samples_per_split["text"])
    total_adjusted_labels = []
    for k in range(0, len(tokenized_samples["input_ids"])):
        prev_wid = -1
        word_ids_list = tokenized_samples.word_ids(batch_index=k)
        existing_label_ids = all_samples_per_split["ner_tags"][k]
        i = -1
        adjusted_label_ids = []
        for wid in word_ids_list:
            if wid is None:
                adjusted_label_ids.append(-100)
            elif wid != prev_wid:
                i += 1
                adjusted_label_ids.append(existing_label_ids[i])
                prev_wid = wid
            else:
                label_name = label_names[existing_label_ids[i]]
                adjusted_label_ids.append(existing_label_ids[i])
        total_adjusted_labels.append(adjusted_label_ids)
    tokenized_samples["labels"] = total_adjusted_labels
    return tokenized_samples
```

**Explanation line by line**:

This function is crucial for BERT fine-tuning because BERT uses subword tokenization, which can split words into multiple tokens. We need to align the original word-level NER tags with the new subword tokens.

- **Line 1**: Define function that takes a batch of samples

- **Lines 2-3**: Tokenize all text samples in the batch:
  - `batch_encode_plus()`: Tokenizes multiple texts at once
  - Converts text to input_ids, attention_mask, etc.
  - Example: "radioheads" might become ["radio", "##heads"]

- **Line 4**: Initialize list to store adjusted labels for all samples

- **Line 5**: Iterate through each sample in the batch:
  - `len(tokenized_samples["input_ids"])`: Number of samples

- **Line 6**: Track the previous word ID to detect when we move to a new word
  - Initialize to -1 (no previous word yet)

- **Line 7**: Get the word IDs for the current sample:
  - `word_ids()`: Maps each token back to its original word position
  - Example: ["[CLS]", "radio", "##heads", "[SEP]"] → [None, 0, 0, None]
  - Special tokens ([CLS], [SEP]) have word_id = None

- **Line 8**: Get the original NER tags for this sample (word-level labels)

- **Line 9**: Initialize counter for the original word position
  - Starts at -1 because we'll increment before using

- **Line 10**: Initialize list for adjusted labels (token-level)

- **Line 11**: Iterate through each word ID in the tokenized sequence

- **Lines 12-13**: If word_id is None (special tokens):
  - Assign label -100
  - -100 is PyTorch's ignore index (won't contribute to loss)

- **Lines 14-17**: If we've moved to a new word (wid != prev_wid):
  - Increment the word counter `i`
  - Assign the original word's label to this token
  - Update `prev_wid` to track the new word

- **Lines 18-20**: Else (continuation of the same word, e.g., "##heads"):
  - Get the label name (appears unused, could be optimization bug)
  - Assign the same label as the first subword token
  - This ensures all subwords of a word get the same label

- **Line 21**: Add the adjusted labels for this sample to the total list

- **Line 22**: Add the labels to the tokenized samples dictionary:
  - Key "labels" is what the Trainer expects for supervised learning

- **Line 23**: Return the tokenized samples with aligned labels

**Purpose**: This function is essential for handling BERT's WordPiece tokenization, ensuring that:
1. Special tokens are ignored during training (label -100)
2. Each subword token gets the appropriate NER label
3. Labels align correctly with tokenized input

---

### Cell 11: Apply Tokenization to Dataset
```python
tokenized_dataset = dataset.map(
    tokenize_adjust_labels,batched=True)
```

**Explanation**:
- **Line 1-2**: Apply the tokenization function to the entire dataset:
  - `dataset.map()`: Applies a function to all examples in the dataset
  - `tokenize_adjust_labels`: The function defined in the previous cell
  - `batched=True`: Process multiple examples at once for efficiency
  - Applies to both "train" and "test" splits automatically

**Output**: Shows progress bars for mapping operation on both splits (539 and 60 examples)

**Result**: `tokenized_dataset` now contains:
- Original fields: tokens, ner_tags, spans, text
- New fields: input_ids, attention_mask, labels (aligned with tokens)

---

### Cell 12: Create Data Collator
```python
data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer)
```

**Explanation**:
- **Lines 1-2**: Create a data collator for token classification tasks:
  - `DataCollatorForTokenClassification`: Specialized collator for NER/token classification
  - `tokenizer=tokenizer`: Uses the BERT tokenizer for padding

**What it does**:
- Takes a batch of examples and prepares them for the model
- Pads sequences to the same length (required for batching)
- Pads labels with -100 (ignored in loss calculation)
- Creates attention masks to indicate real tokens vs. padding
- Returns a dictionary with input_ids, attention_mask, and labels tensors

---

### Cell 13: Define Evaluation Metrics
```python
metric = load("seqeval")
def compute_metrics(data):
    predictions, labels = data
    predictions = np.argmax(predictions, axis=2)

    data = zip(predictions, labels)
    data = [
        [(p, l) for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in data
    ]

    true_predictions = [
        [label_names[p] for (p, l) in data_point]
        for data_point in data
    ]
    true_labels = [
        [label_names[l] for (p, l) in data_point]
        for data_point in data
    ]

    results = metric.compute(predictions=true_predictions,
                             references=true_labels)
    flat_results = {
        "overall_precision": results["overall_precision"],
        "overall_recall": results["overall_recall"],
        "overall_f1": results["overall_f1"],
        "overall_accuracy": results["overall_accuracy"],
    }
    for k in results.keys():
        if (k not in flat_results.keys()):
            flat_results[k + "_f1"] = results[k]["f1"]
    return flat_results
```

**Explanation line by line**:

- **Line 1**: Load the seqeval metric:
  - `seqeval`: Standard evaluation metric for sequence labeling tasks
  - Computes precision, recall, F1 for each entity type
  - Handles BIO format correctly (considers full entity spans)

- **Line 2**: Define function to compute evaluation metrics:
  - Called by Trainer during evaluation
  - Takes predictions and labels as input

- **Line 3**: Unpack the input data tuple

- **Line 4**: Convert logits to predicted labels:
  - `predictions`: Shape (batch_size, seq_length, num_labels) - raw logits
  - `np.argmax(axis=2)`: Get the label with highest score for each token
  - Result: Shape (batch_size, seq_length) - predicted label IDs

- **Line 6**: Zip predictions and labels together for parallel iteration

- **Lines 7-9**: Filter out padding tokens:
  - List comprehension iterates through each example
  - Inner list comprehension pairs predictions with labels
  - `if l != -100`: Excludes padding tokens (labeled -100)
  - Result: List of (prediction, label) tuples per example

- **Lines 11-14**: Convert predicted label IDs to label names:
  - Iterate through each data point
  - For each (p, l) pair, convert prediction `p` to label name
  - Example: 1 → "B-Artist"
  - Result: List of lists of predicted label strings

- **Lines 15-18**: Convert true label IDs to label names:
  - Same process for ground truth labels
  - Example: 3 → "B-WoA"

- **Lines 20-21**: Compute metrics using seqeval:
  - `predictions=true_predictions`: Predicted label sequences
  - `references=true_labels`: Ground truth label sequences
  - Returns dictionary with precision, recall, F1 per entity type

- **Lines 22-27**: Extract overall metrics:
  - Create new dictionary with overall (micro-averaged) metrics
  - Precision: What percentage of predicted entities are correct
  - Recall: What percentage of true entities were found
  - F1: Harmonic mean of precision and recall
  - Accuracy: Token-level accuracy

- **Lines 28-30**: Add per-entity F1 scores:
  - Iterate through all keys in results
  - If key not already in flat_results, it's an entity-specific metric
  - Extract F1 score for that entity type
  - Example: results["Artist"]["f1"] → flat_results["Artist_f1"]

- **Line 31**: Return the flattened metrics dictionary

**Purpose**: Provides comprehensive evaluation during training, tracking both overall performance and per-entity-type performance.

---

### Cell 14: Train the Model
```python
model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-uncased", num_labels=len(label_names))
training_args = TrainingArguments(
    output_dir="./fine_tune_bert_output",
    eval_strategy="steps",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=7,
    weight_decay=0.01,
    logging_steps=1000,
    run_name="ep_10_tokenized_l1",
    save_strategy="no",
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()
```

**Explanation line by line**:

- **Lines 1-2**: Load pre-trained BERT model for token classification:
  - `AutoModelForTokenClassification`: BERT with a classification head
  - `from_pretrained("bert-base-uncased")`: Loads pre-trained BERT weights
  - `num_labels=len(label_names)`: 5 labels (O, B-Artist, I-Artist, B-WoA, I-WoA)
  - The classification head is randomly initialized (will be trained)

- **Lines 3-14**: Configure training arguments:

  - **Line 4**: `output_dir`: Directory to save checkpoints and logs

  - **Line 5**: `eval_strategy="steps"`: Evaluate at regular step intervals

  - **Line 6**: `learning_rate=2e-5`: Learning rate (0.00002)
    - Standard value for fine-tuning BERT

  - **Line 7**: `per_device_train_batch_size=16`: 16 examples per batch during training

  - **Line 8**: `per_device_eval_batch_size=16`: 16 examples per batch during evaluation

  - **Line 9**: `num_train_epochs=7`: Train for 7 complete passes through the data

  - **Line 10**: `weight_decay=0.01`: L2 regularization factor
    - Helps prevent overfitting

  - **Line 11**: `logging_steps=1000`: Log metrics every 1000 steps

  - **Line 12**: `run_name`: Name for this training run (for logging/tracking)

  - **Line 13**: `save_strategy="no"`: Don't save checkpoints during training
    - Saves disk space; we'll save the final model manually

- **Lines 15-23**: Create Trainer object:

  - **Line 16**: `model=model`: The BERT model to train

  - **Line 17**: `args=training_args`: Training configuration

  - **Line 18**: `train_dataset`: Tokenized training data (539 examples)

  - **Line 19**: `eval_dataset`: Tokenized test data (60 examples)

  - **Line 20**: `data_collator`: Handles batching and padding

  - **Line 21**: `tokenizer=tokenizer`: BERT tokenizer (for decoding if needed)

  - **Line 22**: `compute_metrics`: Function to compute evaluation metrics

- **Line 24**: Start training:
  - Runs for 7 epochs
  - Automatically handles forward pass, loss calculation, backpropagation
  - Periodically evaluates on test set
  - Returns training statistics

**Output**:
- Warning: "Some weights... are newly initialized" - Expected, the classification head is new
- Warning about deprecated `tokenizer` parameter - Can be ignored
- Training completes in 377 seconds (about 6 minutes)
- Final training loss: 0.256 (model learned well)
- 238 total training steps (539 samples × 7 epochs / 16 batch size)

---

### Cell 15: Evaluate the Model
```python
trainer.evaluate()
```

**Explanation**:
- Evaluates the trained model on the test dataset
- Computes all metrics defined in `compute_metrics` function
- Returns a dictionary with evaluation results

**Output Analysis**:
- `eval_loss: 0.201`: Low test loss indicates good generalization
- `eval_overall_precision: 0.544`: 54.4% of predicted entities are correct
- `eval_overall_recall: 0.612`: 61.2% of true entities were found
- `eval_overall_f1: 0.576`: Overall F1 score of 57.6%
- `eval_overall_accuracy: 0.924`: 92.4% token-level accuracy
- `eval_Artist_f1: 0.574`: F1 for Artist entities
- `eval_WoA_f1: 0.582`: F1 for Work of Art entities
- `eval_runtime: 2.77s`: Evaluation took 2.77 seconds
- `eval_samples_per_second: 21.7`: Processed ~22 samples/second

**Interpretation**:
- Model performs reasonably well for a small training dataset
- Slightly better recall than precision (tends to over-predict entities)
- High token-level accuracy but moderate entity-level F1
- Similar performance on both entity types (Artist and WoA)

---

### Cell 16: Save the Model
```python
trainer.save_model("../models/bert_fine_tuned")
```

**Explanation**:
- Saves the trained model to disk
- `"../models/bert_fine_tuned"`: Directory where model will be saved
- Saves:
  - Model weights (pytorch_model.bin or model.safetensors)
  - Model configuration (config.json)
  - Tokenizer files (vocab.txt, tokenizer_config.json, etc.)
- Can be loaded later with `from_pretrained()`

---

### Cell 17: Load the Saved Model
```python
model = AutoModelForTokenClassification.from_pretrained("../models/bert_fine_tuned")
tokenizer = AutoTokenizer.from_pretrained("../models/bert_fine_tuned")
```

**Explanation**:
- **Line 1**: Load the fine-tuned model from the saved directory
  - Automatically detects and loads the correct architecture
  - Loads the trained weights from the classification head

- **Line 2**: Load the tokenizer from the saved directory
  - Ensures we use the exact same tokenization as during training

**Purpose**: Demonstrates how to reload the model for inference or further training.

---

### Cell 18: Test the Model on New Text
```python
text = "music similar to morphie robocobra quartet | featuring elements like saxophone prominent bass"
from transformers import pipeline
pipeline = pipeline(
    task="token-classification", model=model, tokenizer=tokenizer,
    aggregation_strategy="simple")
result = pipeline(text)
for entity in result:
    print(entity)
```

**Explanation line by line**:

- **Line 1**: Define a test sentence:
  - Contains potential entities: "morphie robocobra quartet" (likely an artist/band)
  - Real-world style query about music recommendations

- **Line 2**: Import the pipeline utility from transformers

- **Lines 3-5**: Create a token classification pipeline:
  - `task="token-classification"`: Specify NER/token classification task
  - `model=model`: Use our fine-tuned model
  - `tokenizer=tokenizer`: Use the corresponding tokenizer
  - `aggregation_strategy="simple"`: Merge subword tokens into words
    - Combines "roboco" + "##bra" into "robocobra"
    - Averages confidence scores for merged tokens

- **Line 6**: Run inference on the text:
  - Tokenizes, runs model, post-processes predictions
  - Returns list of detected entities with scores

- **Lines 7-8**: Print each detected entity:
  - Shows entity group (label), confidence score, text, and position

**Output Analysis**:
- **Entity 1**:
  - Label: `LABEL_0` (corresponds to "O" - not an entity)
  - Text: "music similar to"
  - Score: 0.999 (very confident)

- **Entity 2**:
  - Label: `LABEL_1` (corresponds to "B-Artist")
  - Text: "morphie roboco"
  - Score: 0.855 (fairly confident)
  - Model correctly identifies beginning of artist name

- **Entity 3**:
  - Label: `LABEL_2` (corresponds to "I-Artist")
  - Text: "##bra quartet"
  - Score: 0.719 (moderate confidence)
  - Continuation of artist name (should be "bra" not "##bra")

- **Entity 4**:
  - Label: `LABEL_0` (O - not an entity)
  - Text: "| featuring elements like saxophone prominent bass"
  - Score: 0.999 (very confident)

**Interpretation**:
- Model successfully identified "morphie robocobra quartet" as an Artist
- The aggregation could be improved (subword "##bra" shouldn't show "##")
- Model correctly ignored non-entity portions of the text
- Confidence scores indicate reasonable certainty in predictions

---

## Summary

This notebook demonstrates a complete pipeline for fine-tuning BERT for Named Entity Recognition:

1. **Data Preparation**: Loaded and cleaned music NER data, converted labels to BIO format
2. **Tokenization Challenge**: Handled BERT's WordPiece tokenization by aligning subword tokens with word-level labels
3. **Model Architecture**: Used pre-trained BERT with a randomly initialized classification head (5 classes)
4. **Training**: Fine-tuned for 7 epochs with standard hyperparameters (lr=2e-5, batch_size=16)
5. **Evaluation**: Achieved 57.6% F1 score on entity detection, 92.4% token-level accuracy
6. **Deployment**: Saved and loaded the model for inference using HuggingFace pipelines

**Key Concepts**:
- **Subword Tokenization**: BERT splits words into pieces; requires careful label alignment
- **Label -100**: PyTorch's ignore index for padding and special tokens
- **BIO Tagging**: B=Begin, I=Inside, O=Outside for sequential entity labeling
- **Transfer Learning**: Pre-trained BERT provides strong language understanding; fine-tuning adapts it to NER
- **Seqeval Metric**: Proper evaluation that considers complete entity spans, not just individual tokens

**Potential Improvements**:
- Increase training data (599 examples is relatively small)
- Experiment with learning rate schedules
- Try larger models (bert-large, RoBERTa, etc.)
- Use data augmentation techniques
- Tune the aggregation strategy for better entity span extraction
