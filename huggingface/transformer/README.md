# 🤗 `transformers` Cheat Sheet

> Quick reference for major classes, methods, and usage patterns in Hugging Face’s `transformers` package.

---

## Overview

- **Purpose**: Provides pretrained models and tools for NLP, vision, and audio tasks.
- **Core Features**:
  - Access to 1000+ pretrained transformer models.
  - Tokenization, model inference, training, and fine-tuning.
  - Pipelines for end-to-end tasks.
  - Integration with PyTorch, TensorFlow, and JAX.

---

## Key Classes & Their Methods

### `AutoModel`, `AutoModelForSequenceClassification`, etc.
| Method | Description |
|--------|-------------|
| `from_pretrained(model_name)` | Load a pretrained model. |
| `save_pretrained(path)` | Save model to directory. |
| `forward(input_ids, attention_mask)` | Perform forward pass. |

### `AutoTokenizer`
| Method | Description |
|--------|-------------|
| `from_pretrained(model_name)` | Load a pretrained tokenizer. |
| `encode(text)` | Encode text into token ids. |
| `decode(token_ids)` | Decode token ids into text. |
| `save_pretrained(path)` | Save tokenizer to directory. |

### `Trainer`
| Method | Description |
|--------|-------------|
| `train()` | Fine-tune model on training dataset. |
| `evaluate()` | Evaluate model on validation set. |
| `predict(test_dataset)` | Run inference on test set. |
| `save_model(path)` | Save model checkpoints. |

### `TrainingArguments`
| Argument | Description |
|----------|-------------|
| `output_dir` | Directory to save checkpoints. |
| `per_device_train_batch_size` | Training batch size per device. |
| `learning_rate` | Initial learning rate. |
| `num_train_epochs` | Number of training epochs. |

### `pipeline`
| Method | Description |
|--------|-------------|
| `pipeline(task, model, tokenizer)` | Quickly create a ready-to-use model for a task. |

---

## Example Code Snippets

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize input and run model
inputs = tokenizer("Hello world!", return_tensors="pt")
outputs = model(**inputs)
```

```python
# Using a pipeline
classifier = pipeline("sentiment-analysis")
result = classifier("Hugging Face is awesome!")
```

---

## 🔄 Common Workflows

1. **Text Classification (Pipeline API)**
   ```python
   classifier = pipeline("sentiment-analysis")
   result = classifier("Sample text")
   ```

2. **Fine-tuning with Trainer**
   ```python
   from transformers import Trainer, TrainingArguments
   args = TrainingArguments(output_dir="./results", per_device_train_batch_size=8)
   trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds)
   trainer.train()
   ```

3. **Save and Load Model**
   ```python
   model.save_pretrained("my_model")
   model = AutoModelForSequenceClassification.from_pretrained("my_model")
   ```

---

## CLI Tools

```bash
transformers-cli login
transformers-cli env
```

---

## 💡 Pro Tips

- Use `Auto*` classes to avoid specifying model architecture.
- `pipeline()` is perfect for quick demos and prototypes.
- Hugging Face Hub contains model cards with usage tips.
- Combine with `datasets` and `accelerate` for full workflows.

---

## 📚 Resources

- [Official Documentation](https://huggingface.co/docs/transformers)
- [Model Hub](https://huggingface.co/models)
- [API Reference](https://huggingface.co/docs/transformers/main/en/model_doc/auto)
- [Example Notebooks](https://github.com/huggingface/notebooks)

