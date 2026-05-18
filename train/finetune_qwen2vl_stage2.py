"""
Fine-tune Qwen2-VL-2B-Instruct with LoRA for AI-generated image detection.

Uses HuggingFace TRL + PEFT for memory-efficient LoRA fine-tuning.
Works on a single RTX 3090 Ti (24GB VRAM).

Usage:
    python train/finetune_qwen2vl_stage2.py --dataset_name 72b_balanced \
        --train_data dataset/finetune_72b/train_balanced.json \
        --val_data dataset/finetune_72b/val_balanced.json

    # Custom run name
    python train/finetune_qwen2vl_stage2.py --run_name my_experiment \
        --train_data dataset/finetune_72b/train_balanced.json
"""

import argparse
import json
import logging
import os
import re
import sys
import torch
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path

from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info


def build_auto_run_name(args):
    """Build a run name from the core training hyperparameters."""
    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    run_name = f"{date_str}_qwen2vl-2b_{args.dataset_name}_r{args.lora_r}_ep{args.epochs}"
    return run_name


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2-VL with LoRA")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="custom",
                        help="Dataset name for run naming (e.g., '72b_balanced', 'sample', 'aigen')")
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="output/finetune",
                        help="Base output directory")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Run name (auto-generated if not set)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--inp_h", type=int, default=342)
    parser.add_argument("--inp_w", type=int, default=512)
    parser.add_argument("--vision_only", action="store_true",
                        help="Fine-tune only vision encoder (freeze LLM). "
                             "Allows using sample solution LLM binary without re-quantizing.")
    parser.add_argument("--augment", action="store_true", default=True,
                        help="Apply data augmentation (flip, color jitter, JPEG compression) to training images")
    parser.add_argument("--no_augment", action="store_false", dest="augment",
                        help="Disable data augmentation")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="Label smoothing factor (0.0=off, 0.1=default)")
    parser.add_argument("--contrastive", action="store_true", default=False,
                        help="Add supervised contrastive loss (pushes real/fake embeddings apart)")
    parser.add_argument("--contrastive_weight", type=float, default=0.1,
                        help="Weight for contrastive loss (total = CE + weight * contrastive)")
    parser.add_argument("--contrastive_temp", type=float, default=0.07,
                        help="Temperature for contrastive loss similarity")
    parser.add_argument("--stage1_aux_cls", action="store_true", default=False,
                        help="Add a training-only auxiliary classifier on stage-1 hidden states")
    parser.add_argument("--stage1_aux_cls_weight", type=float, default=0.1,
                        help="Weight for stage-1 auxiliary classification loss")
    parser.add_argument("--criterion_aux_cls", action="store_true", default=False,
                        help="Add a training-only per-criterion auxiliary classifier (8 outputs, BCE)")
    parser.add_argument("--criterion_aux_cls_weight", type=float, default=0.05,
                        help="Weight for per-criterion auxiliary classification loss")
    parser.add_argument("--resume_lora", type=str, default=None,
                        help="Path to previously trained LoRA adapter (for sequential stage1→stage2 training)")
    parser.add_argument("--attn_impl", type=str, default="eager",
                        choices=["eager", "sdpa", "flash_attention_2"],
                        help="Attention implementation. Use sdpa on H100/H200 for 2-3x speedup.")
    parser.add_argument("--use_dora", action="store_true", default=False,
                        help="Experimental: use DoRA decomposition")
    parser.add_argument("--use_rslora", action="store_true", default=False,
                        help="Experimental: use rank-stabilized LoRA scaling")
    parser.add_argument("--focal_loss", action="store_true", default=False,
                        help="Experimental: use focal loss instead of cross-entropy")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Focal loss gamma (higher = more focus on hard tokens)")
    args = parser.parse_args()

    # Auto-generate run name if not set
    if args.run_name is None:
        args.run_name = build_auto_run_name(args)

    return args


def setup_logging(run_dir, run_name):
    """Setup logging to both console and file."""
    log_file = run_dir / f"{run_name}.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)
    file_fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    return logger, log_file


class EpochLossLoggerCallback(TrainerCallback):
    """Persist per-epoch training and evaluation losses to a run-local file."""

    def __init__(self, run_dir):
        self.output_path = Path(run_dir) / "epoch_losses.json"
        self._records = {}

    @staticmethod
    def _epoch_key(epoch):
        epoch_value = float(epoch)
        epoch_label = int(epoch_value) if epoch_value.is_integer() else round(epoch_value, 4)
        return epoch_value, epoch_label

    def _update_record(self, epoch, logs, global_step=None):
        epoch_key, epoch_label = self._epoch_key(epoch)
        record = self._records.get(epoch_key, {"epoch": epoch_label})

        if "loss" in logs:
            record["train_loss"] = float(logs["loss"])
        if "eval_loss" in logs:
            record["eval_loss"] = float(logs["eval_loss"])
        if "learning_rate" in logs:
            record["learning_rate"] = float(logs["learning_rate"])
        if "grad_norm" in logs:
            record["grad_norm"] = float(logs["grad_norm"])
        if global_step is not None:
            record["global_step"] = int(global_step)

        record["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self._records[epoch_key] = record

    def _write(self):
        records = [self._records[key] for key in sorted(self._records)]
        with open(self.output_path, "w") as f:
            json.dump(records, f, indent=2)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not getattr(state, "is_world_process_zero", True):
            return
        if not logs:
            return

        epoch = logs.get("epoch", state.epoch)
        if epoch is None:
            return

        tracked_keys = {"loss", "eval_loss", "learning_rate", "grad_norm"}
        if not tracked_keys.intersection(logs):
            return

        self._update_record(epoch, logs, global_step=state.global_step)
        self._write()

    def backfill_from_log_history(self, log_history):
        for logs in log_history:
            epoch = logs.get("epoch")
            if epoch is None:
                continue
            tracked_keys = {"loss", "eval_loss", "learning_rate", "grad_norm"}
            if not tracked_keys.intersection(logs):
                continue
            self._update_record(epoch, logs, global_step=logs.get("step"))
        if self._records:
            self._write()


class ImageAugmentation:
    """Augmentations applied before VLM processor. Only for training."""

    def __init__(self):
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            ], p=0.2),
            transforms.RandomApply([
                # Simulate JPEG compression artifacts
                lambda img: img,  # placeholder — PIL JPEG compression below
            ], p=0.3),
        ])

    def __call__(self, image_path):
        from PIL import Image
        import io
        import random

        img = Image.open(image_path).convert("RGB")
        img = self.transform(img)

        # Random JPEG compression (simulates real-world quality)
        if random.random() < 0.3:
            quality = random.randint(50, 95)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            img = Image.open(buffer).convert("RGB")

        return img


STAGE1_REAL_PATTERNS = [
    re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for pattern in [
        r"no evidence to suggest .*?fake",
        r"no significant indicators .*?(digital manipulation|artificial creation|artificial generation)",
        r"image appears to be real",
        r"image is real",
        r"highly likely that the image is real",
        r"likely that the image is real",
        r"most likely real",
        r"appears to be a real photograph",
        r"real photograph",
        r"real image",
        r"support the authenticity of the image",
        r"support the authenticity",
    ]
]

STAGE1_FAKE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for pattern in [
        r"image is \*\*fake\*\*",
        r"image is fake",
        r"image is determined to be fake",
        r"image is most likely fake",
        r"image is likely fake",
        r"image is clearly fake",
        r"image is definitively fake",
        r"it is clear that (?:this )?image is .*?fake",
        r"most likely \*\*fake\*\*",
        r"most likely fake",
        r"likely fake",
        r"clearly fake",
        r"definitively fake",
        r"not real",
        r"ai-generated",
        r"ai generated",
        r"digitally created",
        r"created digitally",
        r"digital creation",
        r"computer-generated",
        r"artificially generated",
    ]
]

STAGE2_OVERALL_PATTERN = re.compile(r'"overall_likelihood"\s*:\s*"([^"]+)"', re.IGNORECASE)


def parse_explicit_is_fake(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in (0, 1):
        return value
    if isinstance(value, float) and value in (0.0, 1.0):
        return int(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "fake", "ai-generated", "ai generated"}:
            return 1
        if normalized in {"0", "false", "real"}:
            return 0
    return None


def normalize_text(text):
    return " ".join(text.lower().split())


def infer_stage1_class_label(assistant_text):
    normalized = normalize_text(assistant_text)
    start = normalized.rfind("final conclusion")
    if start == -1:
        start = normalized.rfind("conclusion")
    tail = normalized[start:] if start != -1 else normalized[-1200:]

    matches = []
    for pattern in STAGE1_REAL_PATTERNS:
        matches.extend((match.start(), 0) for match in pattern.finditer(tail))
    for pattern in STAGE1_FAKE_PATTERNS:
        matches.extend((match.start(), 1) for match in pattern.finditer(tail))

    if not matches:
        return -100
    matches.sort(key=lambda item: item[0])
    return matches[-1][1]


def infer_stage2_class_label(assistant_text):
    match = STAGE2_OVERALL_PATTERN.search(assistant_text)
    if match is None:
        return -100

    verdict = match.group(1).strip().lower()
    if verdict == "real":
        return 0
    if verdict in {"ai-generated", "ai generated"}:
        return 1
    return -100


def infer_sample_label(item, assistant_text, has_image):
    explicit_label = parse_explicit_is_fake(item.get("is_fake"))
    if explicit_label is not None:
        return explicit_label, "explicit"

    if has_image:
        inferred = infer_stage1_class_label(assistant_text)
    else:
        inferred = infer_stage2_class_label(assistant_text)

    if inferred != -100:
        return inferred, "inferred"
    return -100, "missing"


def mean_pool_hidden_states(hidden_states, labels=None, attention_mask=None):
    if labels is not None:
        mask = labels != -100
    elif attention_mask is not None:
        mask = attention_mask.bool()
    else:
        return hidden_states.mean(dim=1)

    mask = mask.unsqueeze(-1).to(hidden_states.dtype)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return (hidden_states * mask).sum(dim=1) / denom


class AIGCDetectionDataset(Dataset):
    """Dataset for AIGC detection fine-tuning."""

    def __init__(self, data_path, processor, max_length=2048, inp_h=342, inp_w=512,
                 augment=False):
        with open(data_path) as f:
            self.data = json.load(f)
        self.processor = processor
        self.max_length = max_length
        self.inp_h = inp_h
        self.inp_w = inp_w
        self.augment = augment
        self.augmentation = ImageAugmentation() if augment else None
        self.label_stats = self._compute_label_stats()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        messages = item["messages"]

        user_msg = messages[0]
        assistant_msg = messages[1]

        image_path = None
        user_text = ""
        for content in user_msg["content"]:
            if content["type"] == "image":
                image_path = content["image"]
            elif content["type"] == "text":
                user_text = content["text"]

        assistant_text = ""
        for content in assistant_msg["content"]:
            if content["type"] == "text":
                assistant_text = content["text"]

        has_image = image_path is not None
        binary_label, _ = infer_sample_label(item, assistant_text, has_image)
        stage1_class_label = binary_label if has_image else -100

        # Apply augmentation to image (training only)
        augmented_image = None
        if has_image and self.augment and self.augmentation:
            try:
                augmented_image = self.augmentation(image_path)
            except Exception:
                augmented_image = None

        if has_image:
            # Stage 1: image + text prompt
            if augmented_image is not None:
                # Use augmented PIL image directly
                chat_messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": augmented_image,
                                "resized_height": self.inp_h,
                                "resized_width": self.inp_w,
                            },
                            {"type": "text", "text": user_text},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": assistant_text}],
                    },
                ]
            else:
                # Use original image path
                chat_messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image_path,
                                "resized_height": self.inp_h,
                                "resized_width": self.inp_w,
                            },
                            {"type": "text", "text": user_text},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": assistant_text}],
                    },
                ]
        else:
            # Stage 2: text-only (no image)
            chat_messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_text}],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": assistant_text}],
                },
            ]

        text = self.processor.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=False
        )

        if has_image:
            image_inputs, _ = process_vision_info(chat_messages)
            inputs = self.processor(
                text=text,
                images=image_inputs,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
            )
        else:
            inputs = self.processor(
                text=text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
            )

        input_ids = inputs["input_ids"].squeeze(0)
        labels = input_ids.clone()

        text_before_assistant = self.processor.apply_chat_template(
            chat_messages[:1], tokenize=False, add_generation_prompt=True
        )
        if has_image:
            tokens_before = self.processor(
                text=text_before_assistant,
                images=image_inputs,
                return_tensors="pt",
            )["input_ids"].shape[1]
        else:
            tokens_before = self.processor(
                text=text_before_assistant,
                return_tensors="pt",
            )["input_ids"].shape[1]

        labels[:tokens_before] = -100
        labels[input_ids == self.processor.tokenizer.pad_token_id] = -100

        result = {
            "input_ids": input_ids,
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels,
            "is_fake": torch.tensor(binary_label, dtype=torch.long),  # 0=real, 1=fake, -100=unknown
            "stage1_class_label": torch.tensor(stage1_class_label, dtype=torch.long),
        }

        # Per-criterion labels (8 binary scores, -100 for missing)
        criterion_labels = item.get("criterion_labels")
        if has_image and isinstance(criterion_labels, list) and len(criterion_labels) == 8:
            result["criterion_labels"] = torch.tensor(criterion_labels, dtype=torch.long)
        else:
            result["criterion_labels"] = torch.full((8,), -100, dtype=torch.long)

        for key in ["pixel_values", "image_grid_thw", "mm_token_type_ids"]:
            if key in inputs and inputs[key] is not None:
                result[key] = inputs[key].squeeze(0)
        return result

    def _compute_label_stats(self):
        stats = {
            "stage1_total": 0,
            "stage1_labeled": 0,
            "stage1_real": 0,
            "stage1_fake": 0,
            "stage1_explicit": 0,
            "stage1_inferred": 0,
            "overall_labeled": 0,
            "criterion_labeled": 0,  # Stage 1 entries with at least one valid criterion label
        }
        for item in self.data:
            messages = item["messages"]
            user_msg = messages[0]
            assistant_msg = messages[1]
            has_image = any(content["type"] == "image" for content in user_msg["content"])
            assistant_text = ""
            for content in assistant_msg["content"]:
                if content["type"] == "text":
                    assistant_text = content["text"]
                    break

            binary_label, label_source = infer_sample_label(item, assistant_text, has_image)
            if binary_label != -100:
                stats["overall_labeled"] += 1

            if not has_image:
                continue

            stats["stage1_total"] += 1
            criterion_labels = item.get("criterion_labels")
            if isinstance(criterion_labels, list) and len(criterion_labels) == 8:
                if any(v != -100 for v in criterion_labels):
                    stats["criterion_labeled"] += 1

            if binary_label == -100:
                continue

            stats["stage1_labeled"] += 1
            if label_source == "explicit":
                stats["stage1_explicit"] += 1
            elif label_source == "inferred":
                stats["stage1_inferred"] += 1
            if binary_label == 0:
                stats["stage1_real"] += 1
            else:
                stats["stage1_fake"] += 1
        return stats


def collate_fn(batch):
    """Custom collate function to handle variable-size inputs."""
    result = {}
    all_keys = set()
    for item in batch:
        all_keys.update(item.keys())

    for key in all_keys:
        values = [item[key] for item in batch if key in item and item[key] is not None]
        if not values:
            continue
        if key in ("pixel_values", "image_grid_thw"):
            result[key] = torch.cat(values, dim=0) if len(values[0].shape) > 1 else torch.stack(values)
        else:
            result[key] = torch.stack(values)
    return result


def compute_focal_loss(logits, labels, gamma=2.0, ignore_index=-100):
    """Focal loss for autoregressive LM.

    Focuses training on hard-to-predict tokens by down-weighting easy ones.
    FL(p) = -(1-p)^gamma * log(p)
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    ce = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction='none',
        ignore_index=ignore_index,
    )
    pt = torch.exp(-ce)
    focal_weight = (1.0 - pt) ** gamma
    return (focal_weight * ce).sum() / (shift_labels != ignore_index).sum().clamp(min=1)


class AIGCTrainer(Trainer):
    """Custom Trainer that adds optional auxiliary losses to standard causal LM loss.

    Contrastive loss (InfoNCE/NT-Xent) operates on the mean-pooled hidden states
    from the last layer. It pushes embeddings of same-class samples together
    and different-class samples apart.

    The stage-1 auxiliary classifier predicts real/fake from stage-1 pooled
    hidden states only. It is training-only and saved separately from the LoRA
    adapters so the inference model format stays unchanged.
    """

    AUX_HEAD_FILENAME = "stage1_aux_classifier.pt"
    CRITERION_HEAD_FILENAME = "criterion_aux_classifier.pt"

    def __init__(
        self,
        contrastive_weight=0.1,
        contrastive_temperature=0.07,
        use_stage1_aux_cls=False,
        stage1_aux_cls_weight=0.1,
        use_criterion_aux_cls=False,
        criterion_aux_cls_weight=0.05,
        use_focal_loss=False,
        focal_gamma=2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.contrastive_weight = contrastive_weight
        self.contrastive_temperature = contrastive_temperature
        self.use_stage1_aux_cls = use_stage1_aux_cls
        self.stage1_aux_cls_weight = stage1_aux_cls_weight
        self.use_criterion_aux_cls = use_criterion_aux_cls
        self.criterion_aux_cls_weight = criterion_aux_cls_weight
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.stage1_classifier = None
        self.criterion_classifier = None

        hidden_size = None
        if self.use_stage1_aux_cls or self.use_criterion_aux_cls:
            hidden_size = getattr(self.model.config, "hidden_size", None)
            if hidden_size is None and hasattr(self.model.config, "text_config"):
                hidden_size = getattr(self.model.config.text_config, "hidden_size", None)
            if hidden_size is None:
                raise ValueError("Could not determine hidden size for auxiliary classifiers")

        if self.use_stage1_aux_cls:
            self.stage1_classifier = torch.nn.Linear(hidden_size, 2)

        if self.use_criterion_aux_cls:
            # 8 outputs for 8 per-criterion binary scores (independent BCE)
            self.criterion_classifier = torch.nn.Linear(hidden_size, 8)

    def _move_model_to_device(self, model, device):
        super()._move_model_to_device(model, device)
        if getattr(self, "stage1_classifier", None) is not None:
            self.stage1_classifier.to(device)
        if getattr(self, "criterion_classifier", None) is not None:
            self.criterion_classifier.to(device)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        decay_parameters = self.get_decay_parameter_names(self.model)
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if p.requires_grad and n in decay_parameters
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if p.requires_grad and n not in decay_parameters
                ],
                "weight_decay": 0.0,
            },
        ]
        if self.stage1_classifier is not None:
            optimizer_grouped_parameters.append(
                {
                    "params": [p for p in self.stage1_classifier.parameters() if p.requires_grad],
                    "weight_decay": self.args.weight_decay,
                }
            )
        if self.criterion_classifier is not None:
            optimizer_grouped_parameters.append(
                {
                    "params": [p for p in self.criterion_classifier.parameters() if p.requires_grad],
                    "weight_decay": self.args.weight_decay,
                }
            )

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, self.model)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        is_fake = inputs.pop("is_fake", None)
        stage1_class_label = inputs.pop("stage1_class_label", None)
        criterion_labels = inputs.pop("criterion_labels", None)
        labels = inputs.get("labels")
        need_hidden_states = (
            self.contrastive_weight > 0
            or self.use_stage1_aux_cls
            or self.use_criterion_aux_cls
        )

        outputs = model(**inputs, output_hidden_states=need_hidden_states)

        if self.use_focal_loss and labels is not None:
            ce_loss = compute_focal_loss(outputs.logits, labels, gamma=self.focal_gamma)
        elif labels is not None and self.label_smoother is not None:
            ce_loss = self.label_smoother(outputs, labels, shift_labels=True)
        else:
            ce_loss = outputs.loss

        contrastive_loss = torch.tensor(0.0, device=ce_loss.device)
        stage1_aux_loss = torch.tensor(0.0, device=ce_loss.device)
        criterion_aux_loss = torch.tensor(0.0, device=ce_loss.device)
        pooled = None

        if need_hidden_states and outputs.hidden_states is not None:
            last_hidden = outputs.hidden_states[-1]
            pooled = mean_pool_hidden_states(
                last_hidden,
                labels=labels,
                attention_mask=inputs.get("attention_mask"),
            )

        if self.use_stage1_aux_cls and self.stage1_classifier is not None and pooled is not None and stage1_class_label is not None:
            valid_stage1 = stage1_class_label != -100
            if valid_stage1.any():
                if self.stage1_classifier.weight.device != pooled.device:
                    self.stage1_classifier.to(pooled.device)
                cls_logits = self.stage1_classifier(
                    pooled[valid_stage1].to(self.stage1_classifier.weight.dtype)
                )
                stage1_aux_loss = F.cross_entropy(cls_logits, stage1_class_label[valid_stage1])

        if (
            self.use_criterion_aux_cls
            and self.criterion_classifier is not None
            and pooled is not None
            and criterion_labels is not None
        ):
            # criterion_labels shape: (batch, 8) with values in {0, 1, -100}
            # Only keep samples where all 8 labels are valid (no -100)
            valid_per_sample = (criterion_labels != -100).all(dim=-1)
            if valid_per_sample.any():
                if self.criterion_classifier.weight.device != pooled.device:
                    self.criterion_classifier.to(pooled.device)
                cri_logits = self.criterion_classifier(
                    pooled[valid_per_sample].to(self.criterion_classifier.weight.dtype)
                )
                cri_targets = criterion_labels[valid_per_sample].to(cri_logits.dtype)
                criterion_aux_loss = F.binary_cross_entropy_with_logits(
                    cri_logits, cri_targets
                )

        if self.contrastive_weight > 0 and pooled is not None and is_fake is not None:
            valid_fake = is_fake != -100
            if valid_fake.sum() > 1 and is_fake[valid_fake].unique().numel() > 1:
                contrastive_pooled = torch.nn.functional.normalize(pooled[valid_fake], dim=-1)
                contrastive_labels = is_fake[valid_fake]
                sim_matrix = torch.matmul(contrastive_pooled, contrastive_pooled.T) / self.contrastive_temperature
                same_label = (contrastive_labels.unsqueeze(0) == contrastive_labels.unsqueeze(1)).float()
                mask = torch.eye(len(contrastive_labels), device=ce_loss.device).bool()
                same_label = same_label.masked_fill(mask, 0)
                sim_matrix = sim_matrix.masked_fill(mask, float("-inf"))
                if same_label.sum() > 0:
                    exp_sim = torch.exp(sim_matrix)
                    denom = exp_sim.masked_fill(mask, 0).sum(dim=1, keepdim=True)
                    log_prob = sim_matrix - torch.log(denom.clamp(min=1e-8))
                    positive_mask = same_label > 0
                    if positive_mask.any():
                        contrastive_loss = -(log_prob * positive_mask.float()).sum() / positive_mask.float().sum()

        total_loss = ce_loss
        total_loss = total_loss + self.contrastive_weight * contrastive_loss
        total_loss = total_loss + self.stage1_aux_cls_weight * stage1_aux_loss
        total_loss = total_loss + self.criterion_aux_cls_weight * criterion_aux_loss

        if return_outputs:
            return total_loss, outputs
        return total_loss

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir=output_dir, state_dict=state_dict)
        if self.stage1_classifier is not None:
            torch.save(
                self.stage1_classifier.state_dict(),
                os.path.join(output_dir, self.AUX_HEAD_FILENAME),
            )
        if self.criterion_classifier is not None:
            torch.save(
                self.criterion_classifier.state_dict(),
                os.path.join(output_dir, self.CRITERION_HEAD_FILENAME),
            )

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        if self.stage1_classifier is not None:
            classifier_path = os.path.join(resume_from_checkpoint, self.AUX_HEAD_FILENAME)
            if os.path.exists(classifier_path):
                state_dict = torch.load(classifier_path, map_location="cpu")
                self.stage1_classifier.load_state_dict(state_dict)
        if self.criterion_classifier is not None:
            criterion_path = os.path.join(resume_from_checkpoint, self.CRITERION_HEAD_FILENAME)
            if os.path.exists(criterion_path):
                state_dict = torch.load(criterion_path, map_location="cpu")
                self.criterion_classifier.load_state_dict(state_dict)


def main():
    args = parse_args()

    # Setup run directory
    run_dir = Path(args.output_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger, log_file = setup_logging(run_dir, args.run_name)

    logger.info(f"{'='*60}")
    logger.info(f"Run: {args.run_name}")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'='*60}")
    logger.info(f"Config:")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Dataset: {args.dataset_name}")
    logger.info(f"  Train data: {args.train_data}")
    logger.info(f"  Val data: {args.val_data}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size} x {args.gradient_accumulation} grad accum")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    logger.info(f"  Max length: {args.max_length}")
    logger.info(f"  Image size: {args.inp_h}x{args.inp_w}")
    logger.info(f"  Output dir: {run_dir}")
    logger.info(f"  Log file: {log_file}")
    logger.info(f"")

    # Save config
    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    logger.info(f"Loading model: {args.model}")
    processor = AutoProcessor.from_pretrained(args.model)

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # Detect distributed training
    is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if is_distributed:
        logger.info(f"Distributed training: WORLD_SIZE={os.environ.get('WORLD_SIZE')}, LOCAL_RANK={local_rank}")
        device_map = {"": local_rank}
    else:
        device_map = "auto"

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation=args.attn_impl,
    )

    # Configure LoRA
    if args.vision_only:
        logger.info("Vision-only mode: fine-tuning only visual encoder (LLM frozen)")
        logger.info("  -> Only need to re-quantize VEG (Example1A + 2A)")
        logger.info("  -> LLM binary from sample solution can be reused")

        # Collect exact module names for vision encoder Linear layers only
        import re
        actual_targets = []
        for name, module in model.named_modules():
            if "visual" in name and isinstance(module, torch.nn.Linear):
                actual_targets.append(name)

        logger.info(f"  Found {len(actual_targets)} vision Linear layers")
        logger.info(f"  Examples: {actual_targets[:5]}")

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=actual_targets,
            use_dora=args.use_dora,
            use_rslora=args.use_rslora,
        )
    else:
        # Target LLM decoder layers (default)
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            use_dora=args.use_dora,
            use_rslora=args.use_rslora,
        )

    if args.resume_lora and os.path.exists(args.resume_lora):
        from peft import PeftModel
        logger.info(f"Resuming LoRA from: {args.resume_lora}")
        model = PeftModel.from_pretrained(model, args.resume_lora, is_trainable=True)
        logger.info("  LoRA adapter loaded. Continuing training from this checkpoint.")
    else:
        model = get_peft_model(model, lora_config)

    # PEFT adapters plus checkpointing can drop the autograd graph unless
    # the input embedding activations require grad. Keep this enabled for
    # both fresh LoRA and resumed adapter training.
    model.enable_input_require_grads()
    # Additionally, for resumed LoRA + gradient checkpointing on text-only
    # data (stage 2), the embedding layer itself must produce grad-carrying
    # tensors.  Register a second hook on the raw embed_tokens to be safe.
    if hasattr(model, "get_input_embeddings"):
        _embed = model.get_input_embeddings()
        if _embed is not None:
            _embed.register_forward_hook(
                lambda _mod, _in, out: out.requires_grad_(True)
            )
    logger.info("Enabled input require grads for PEFT training")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
    if trainable_params == 0:
        raise RuntimeError("No trainable parameters found after LoRA setup. Check --resume_lora and adapter loading.")

    # Load datasets
    logger.info(f"Loading training data: {args.train_data}")
    logger.info(f"  Augmentation: {'ON' if args.augment else 'OFF'}")
    train_dataset = AIGCDetectionDataset(
        args.train_data, processor, args.max_length, args.inp_h, args.inp_w,
        augment=args.augment  # Augmentation for training only
    )
    logger.info(f"  Train samples: {len(train_dataset)}")
    logger.info(
        f"  Stage-1 aux labels: {train_dataset.label_stats['stage1_labeled']}/"
        f"{train_dataset.label_stats['stage1_total']} labeled "
        f"(real={train_dataset.label_stats['stage1_real']}, fake={train_dataset.label_stats['stage1_fake']}, "
        f"explicit={train_dataset.label_stats['stage1_explicit']}, inferred={train_dataset.label_stats['stage1_inferred']})"
    )
    logger.info(
        f"  Criterion aux labels: {train_dataset.label_stats.get('criterion_labeled', 0)}/"
        f"{train_dataset.label_stats['stage1_total']} stage-1 samples have 8-criterion labels"
    )
    if args.stage1_aux_cls and train_dataset.label_stats["stage1_labeled"] == 0:
        logger.warning("  Stage-1 aux cls is enabled, but no labeled stage-1 samples were found. Auxiliary loss will stay inactive.")
    if args.criterion_aux_cls and train_dataset.label_stats.get("criterion_labeled", 0) == 0:
        logger.warning("  Criterion aux cls is enabled, but no samples have per-criterion labels. Run extract_criterion_labels.py first.")

    val_dataset = None
    if args.val_data and os.path.exists(args.val_data):
        val_dataset = AIGCDetectionDataset(
            args.val_data, processor, args.max_length, args.inp_h, args.inp_w,
            augment=False  # No augmentation for validation
        )
        logger.info(f"  Val samples: {len(val_dataset)}")
        logger.info(
            f"  Val stage-1 aux labels: {val_dataset.label_stats['stage1_labeled']}/"
            f"{val_dataset.label_stats['stage1_total']} labeled "
            f"(real={val_dataset.label_stats['stage1_real']}, fake={val_dataset.label_stats['stage1_fake']}, "
            f"explicit={val_dataset.label_stats['stage1_explicit']}, inferred={val_dataset.label_stats['stage1_inferred']})"
        )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        run_name=args.run_name,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_strategy="epoch",
        logging_dir=str(run_dir / "logs"),
        # Save best model based on eval loss (not just every epoch)
        save_strategy="epoch",
        eval_strategy="epoch" if val_dataset else "no",
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,  # Keep only top 3 checkpoints to save disk
        # Early stopping: stop if eval loss doesn't improve for N evals
        # (handled by EarlyStoppingCallback below)
        # Precision
        bf16=True,
        label_smoothing_factor=args.label_smoothing,
        # Memory
        gradient_checkpointing=not is_distributed,
        gradient_checkpointing_kwargs={"use_reentrant": False} if not is_distributed else None,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        # Reproducibility
        seed=42,
        data_seed=42,
        report_to="none",
    )

    # Early stopping callback
    epoch_loss_logger = EpochLossLoggerCallback(run_dir)
    callbacks = [epoch_loss_logger]
    logger.info(f"  Epoch loss log: {epoch_loss_logger.output_path}")
    if val_dataset:
        from transformers import EarlyStoppingCallback
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=2,  # Stop if no improvement for 2 epochs
            early_stopping_threshold=0.01,  # Minimum improvement to count
        ))
        logger.info(f"  Early stopping: patience=2, threshold=0.01")

    logger.info(f"  Label smoothing: {args.label_smoothing}")
    logger.info(f"  Contrastive loss: {'ON' if args.contrastive else 'OFF'} (weight={args.contrastive_weight})")
    logger.info(f"  Stage-1 aux cls: {'ON' if args.stage1_aux_cls else 'OFF'} (weight={args.stage1_aux_cls_weight})")
    logger.info(f"  Criterion aux cls: {'ON' if args.criterion_aux_cls else 'OFF'} (weight={args.criterion_aux_cls_weight})")
    logger.info(f"  DoRA: {'ON' if args.use_dora else 'OFF'}")
    logger.info(f"  rsLoRA: {'ON' if args.use_rslora else 'OFF'}")
    logger.info(f"  Focal loss: {'ON (gamma=' + str(args.focal_gamma) + ')' if args.focal_loss else 'OFF'}")
    logger.info(f"  Save best model: {'ON' if val_dataset else 'OFF'}")
    logger.info(f"  Seed: 42")

    trainer = AIGCTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        callbacks=callbacks,
        contrastive_weight=args.contrastive_weight if args.contrastive else 0.0,
        contrastive_temperature=args.contrastive_temp,
        use_stage1_aux_cls=args.stage1_aux_cls,
        stage1_aux_cls_weight=args.stage1_aux_cls_weight if args.stage1_aux_cls else 0.0,
        use_criterion_aux_cls=args.criterion_aux_cls,
        criterion_aux_cls_weight=args.criterion_aux_cls_weight if args.criterion_aux_cls else 0.0,
        use_focal_loss=args.focal_loss,
        focal_gamma=args.focal_gamma,
    )


    logger.info(f"\nStarting training...")
    trainer.train()
    epoch_loss_logger.backfill_from_log_history(trainer.state.log_history)

    # Save final model (best model if early stopping, last model otherwise)
    final_dir = run_dir / "final"
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    if trainer.stage1_classifier is not None:
        torch.save(
            trainer.stage1_classifier.state_dict(),
            final_dir / trainer.AUX_HEAD_FILENAME,
        )
    if trainer.criterion_classifier is not None:
        torch.save(
            trainer.criterion_classifier.state_dict(),
            final_dir / trainer.CRITERION_HEAD_FILENAME,
        )

    # Save training summary
    train_result = {
        "run_name": args.run_name,
        "dataset_name": args.dataset_name,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset) if val_dataset else 0,
        "epochs": args.epochs,
        "lr": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "stage1_aux_cls": args.stage1_aux_cls,
        "stage1_aux_cls_weight": args.stage1_aux_cls_weight if args.stage1_aux_cls else 0.0,
        "stage1_aux_train_labeled": train_dataset.label_stats["stage1_labeled"],
        "stage1_aux_train_real": train_dataset.label_stats["stage1_real"],
        "stage1_aux_train_fake": train_dataset.label_stats["stage1_fake"],
        "stage1_aux_train_explicit": train_dataset.label_stats["stage1_explicit"],
        "stage1_aux_train_inferred": train_dataset.label_stats["stage1_inferred"],
        "criterion_aux_cls": args.criterion_aux_cls,
        "criterion_aux_cls_weight": args.criterion_aux_cls_weight if args.criterion_aux_cls else 0.0,
        "criterion_aux_train_labeled": train_dataset.label_stats.get("criterion_labeled", 0),
        "epoch_loss_log": str(epoch_loss_logger.output_path),
        "final_train_loss": trainer.state.log_history[-1].get("train_loss", None),
        "final_eval_loss": next(
            (h["eval_loss"] for h in reversed(trainer.state.log_history) if "eval_loss" in h),
            None
        ),
        "train_runtime_sec": trainer.state.log_history[-1].get("train_runtime", None),
        "model_path": str(final_dir),
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(train_result, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"Training complete!")
    logger.info(f"  Final train loss: {train_result['final_train_loss']}")
    logger.info(f"  Final eval loss: {train_result['final_eval_loss']}")
    logger.info(f"  Model saved to: {final_dir}")
    logger.info(f"  Log file: {log_file}")
    logger.info(f"  Summary: {run_dir / 'summary.json'}")
    logger.info(f"  Epoch losses: {epoch_loss_logger.output_path}")
    logger.info(f"{'='*60}")
    logger.info(f"\nTest with:")
    logger.info(f"  python test_qwen2vl.py --model {final_dir} --dataset_dir dataset/sample_dataset --output_dir output/results_{args.run_name}")


if __name__ == "__main__":
    main()