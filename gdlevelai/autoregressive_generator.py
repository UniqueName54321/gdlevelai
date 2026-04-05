from __future__ import annotations

import base64
import json
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset

from .device_support import resolve_device


OBJ_RE = re.compile(r"<k>k4</k>\s*<s>(.*?)</s>", re.DOTALL)
NAME_RE = re.compile(r"<k>k2</k>\s*<s>(.*?)</s>", re.DOTALL)
DESC_RE = re.compile(r"<k>k3</k>\s*<s>(.*?)</s>", re.DOTALL)
SONG_RE = re.compile(r"<k>k8</k>\s*<i>(-?\d+)</i>", re.DOTALL)
CUSTOM_SONG_RE = re.compile(r"<k>k45</k>\s*<i>(-?\d+)</i>", re.DOTALL)
WORD_RE = re.compile(r"[A-Za-z0-9']+")

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
OBJ_START = "<obj>"
OBJ_END = "<eobj>"
META_START = "<meta>"
META_END = "<emeta>"
NAME_START = "<name>"
NAME_END = "<ename>"
DESC_START = "<desc>"
DESC_END = "<edesc>"
LAYOUT_START = "<layout>"
LAYOUT_END = "<elayout>"
ARTIFACT_SCHEMA_VERSION = 2


def _log(message: str) -> None:
    print(f"[autoregressive_generator] {message}", flush=True)


@dataclass
class AutoregressiveConfig:
    seq_len: int = 192
    embed_dim: int = 192
    hidden_dim: int = 320
    epochs: int = 8
    batch_size: int = 32
    lr: float = 2e-3
    device: str = "auto"
    max_objects_per_file: int = 0
    max_vocab_size: int = 50000
    min_token_freq: int = 1
    position_quant: int = 10
    rotation_quant: int = 15
    sample_stride: int = 2
    max_steps_per_epoch: int = 2500
    log_every_steps: int = 100
    num_threads: int = 0
    torch_compile: bool = True
    name_max_words: int = 8
    desc_max_words: int = 20
    name_min_words: int = 1
    desc_min_words: int = 1
    max_song_id: int = 1000
    max_custom_song_id: int = 100000000
    artifacts_subdir: str = "artifacts"
    checkpoints_subdir: str = "checkpoints"
    samples_subdir: str = "samples"
    save_preprocessed_artifacts: bool = True
    save_checkpoint_every_epochs: int = 1
    save_samples_every_epochs: int = 1
    samples_per_epoch: int = 1
    sample_preview_max_new_tokens: int = 800
    resume_checkpoint: str = ""


def _extract_object_blob(gmd_text: str) -> str | None:
    match = OBJ_RE.search(gmd_text)
    if not match:
        return None
    return match.group(1).strip()


def _extract_level_name(gmd_text: str) -> str:
    match = NAME_RE.search(gmd_text)
    if not match:
        return "Untitled"
    name = match.group(1).strip()
    return name or "Untitled"


def _decode_level_description(raw_description: str) -> str:
    raw = raw_description.strip()
    if not raw:
        return ""

    if re.fullmatch(r"[A-Za-z0-9+/=_-]{8,}", raw):
        candidates = (
            lambda s: base64.b64decode(s, validate=False),
            lambda s: base64.urlsafe_b64decode(s),
        )
        for decode_fn in candidates:
            try:
                padding = "=" * ((4 - (len(raw) % 4)) % 4)
                decoded = decode_fn(raw + padding)
                text = decoded.decode("utf-8", errors="ignore").strip()
                if text and any(ch.isalpha() for ch in text):
                    printable = sum(ch.isprintable() for ch in text) / max(1, len(text))
                    if printable >= 0.9:
                        return text
            except Exception:
                continue
        return ""

    return raw


def _extract_level_description(gmd_text: str) -> str:
    match = DESC_RE.search(gmd_text)
    if not match:
        return ""
    return _decode_level_description(match.group(1))


def _extract_song_choice(
    gmd_text: str,
    default_song_id: int,
    max_song_id: int,
    max_custom_song_id: int,
) -> tuple[int, bool]:
    custom_match = CUSTOM_SONG_RE.search(gmd_text)
    if custom_match:
        custom_id = max(
            1,
            min(max_custom_song_id, _safe_int(custom_match.group(1), 1)),
        )
        return custom_id, True

    match = SONG_RE.search(gmd_text)
    if not match:
        return max(0, min(max_song_id, default_song_id)), False
    return max(0, min(max_song_id, _safe_int(match.group(1), default_song_id))), False


def _tokenize_words(text: str, max_words: int, prefix: str) -> list[str]:
    words = WORD_RE.findall(text.lower())
    if max_words > 0:
        words = words[:max_words]
    return [f"{prefix}:{word}" for word in words]


def _metadata_to_tokens(
    level_name: str,
    level_description: str,
    song_id: int,
    is_custom_song: bool,
    cfg: AutoregressiveConfig,
) -> list[str]:
    safe_name = level_name.strip() or "Untitled"
    safe_description = level_description.strip() or "No description"
    tokens: list[str] = [
        META_START,
        "songsrc:custom" if is_custom_song else "songsrc:official",
        (
            f"csong:{max(1, min(cfg.max_custom_song_id, song_id))}"
            if is_custom_song
            else f"song:{max(0, min(cfg.max_song_id, song_id))}"
        ),
        NAME_START,
    ]
    tokens.extend(_tokenize_words(safe_name, cfg.name_max_words, "n"))
    tokens.append(NAME_END)
    tokens.append(DESC_START)
    tokens.extend(_tokenize_words(safe_description, cfg.desc_max_words, "d"))
    tokens.append(DESC_END)
    tokens.append(META_END)
    return tokens


def _quantize_int(value: int, step: int) -> int:
    if step <= 1:
        return value
    return int(round(value / step) * step)


def _parse_object_map(raw_obj: str) -> dict[str, str]:
    parts = raw_obj.split(",")
    obj: dict[str, str] = {}
    for i in range(0, len(parts) - 1, 2):
        key = parts[i].strip()
        val = parts[i + 1].strip()
        if key:
            obj[key] = val
    return obj


def _safe_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _extract_raw_objects(blob: str) -> list[str]:
    return [part.strip() for part in blob.split(";") if part.strip()]


def _object_to_field_tokens(raw_obj: str, cfg: AutoregressiveConfig) -> list[str]:
    tokens: list[str] = []
    obj = _parse_object_map(raw_obj)
    object_id = _safe_int(obj.get("1", "1"), 1)
    x = _quantize_int(_safe_int(obj.get("2", "0"), 0), cfg.position_quant)
    y = _quantize_int(_safe_int(obj.get("3", "0"), 0), cfg.position_quant)
    rot = _quantize_int(_safe_int(obj.get("6", "0"), 0), cfg.rotation_quant)

    tokens.extend(
        [
            OBJ_START,
            f"id:{object_id}",
            f"x:{x}",
            f"y:{y}",
            f"r:{rot}",
            OBJ_END,
        ]
    )
    return tokens


def _build_vocab(
    token_sequences: list[list[str]], max_vocab_size: int, min_token_freq: int
) -> tuple[dict[str, int], list[str], Counter[str]]:
    freq = Counter(tok for seq in token_sequences for tok in seq)

    metadata_tokens = {
        META_START,
        META_END,
        NAME_START,
        NAME_END,
        DESC_START,
        DESC_END,
        LAYOUT_START,
        LAYOUT_END,
    }
    must_keep = {
        tok
        for tok in freq
        if tok in metadata_tokens
        or tok.startswith("song:")
        or tok.startswith("songsrc:")
        or tok.startswith("csong:")
        or tok.startswith("n:")
        or tok.startswith("d:")
    }

    kept: list[str] = sorted(must_keep)
    capacity = max(0, max_vocab_size - 4 - len(kept))
    for tok, count in freq.most_common():
        if tok in must_keep:
            continue
        if count < min_token_freq:
            continue
        if capacity > 0 and len(kept) - len(must_keep) >= capacity:
            break
        kept.append(tok)

    itos = [PAD, BOS, EOS, UNK] + kept
    stoi = {tok: i for i, tok in enumerate(itos)}
    return stoi, itos, freq


class TokenDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, encoded: torch.Tensor, seq_len: int, stride: int) -> None:
        self.encoded = encoded
        self.seq_len = seq_len
        self.stride = max(1, stride)
        self._windows = max(0, self.encoded.numel() - self.seq_len - 1)

    def __len__(self) -> int:
        if self._windows <= 0:
            return 0
        return ((self._windows - 1) // self.stride) + 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        x = self.encoded[start : start + self.seq_len]
        y = self.encoded[start + 1 : start + self.seq_len + 1]
        return x, y


def _make_random_batch(
    encoded: torch.Tensor,
    seq_len: int,
    batch_size: int,
    *,
    stride: int,
    device: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = encoded.numel() - seq_len - 1
    if max_start <= 0:
        raise RuntimeError("Encoded token stream is too short for seq_len")

    starts = torch.randint(
        low=0,
        high=(max_start // max(1, stride)) + 1,
        size=(batch_size,),
    ) * max(1, stride)
    offsets = torch.arange(seq_len)

    idx = starts.unsqueeze(1) + offsets.unsqueeze(0)
    x = encoded[idx]
    y = encoded[idx + 1]
    return x.to(device), y.to(device)


class GRULM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self, x: torch.Tensor, h: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        e = self.emb(x)
        o, h = self.gru(e, h)
        logits = self.head(o)
        return logits, h


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def _state_dict_with_prefixed_keys(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {f"_orig_mod.{key}": value for key, value in state_dict.items()}


def _load_state_dict_flexible(
    model: nn.Module, state_dict: dict[str, torch.Tensor]
) -> None:
    variants = [
        state_dict,
        _normalize_state_dict_keys(state_dict),
        _state_dict_with_prefixed_keys(state_dict),
    ]
    last_exc: Exception | None = None
    for candidate in variants:
        try:
            model.load_state_dict(candidate)
            return
        except Exception as exc:  # pragma: no cover - fallback path
            last_exc = exc
    raise RuntimeError(f"Failed to load checkpoint state_dict: {last_exc}")


def _checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: AutoregressiveConfig,
    stoi: dict[str, int],
    itos: list[str],
    epoch: int,
    global_step: int,
) -> dict[str, object]:
    return {
        "model_type": "autoregressive",
        "representation": "object_token",
        "state_dict": _unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "cfg": cfg.__dict__,
        "stoi": stoi,
        "itos": itos,
        "epoch": epoch,
        "global_step": global_step,
    }


def _save_preprocessed_artifacts(
    artifacts_dir: Path,
    cfg: AutoregressiveConfig,
    stoi: dict[str, int],
    itos: list[str],
    token_sequences: list[list[str]],
    encoded: torch.Tensor,
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    offsets: list[int] = []
    cursor = 0
    for seq in token_sequences:
        offsets.append(cursor)
        cursor += len(seq) + 2

    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "representation": "object_token",
        "sequence_count": len(token_sequences),
        "encoded_token_count": int(encoded.numel()),
        "created_unix": int(time.time()),
        "cfg": cfg.__dict__,
    }
    (artifacts_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    (artifacts_dir / "itos.json").write_text(json.dumps(itos), encoding="utf-8")
    (artifacts_dir / "stoi.json").write_text(json.dumps(stoi), encoding="utf-8")
    torch.save(
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "encoded": encoded,
            "sequence_offsets": offsets,
            "sequence_count": len(token_sequences),
        },
        artifacts_dir / "processed_sequences.pt",
    )
    (artifacts_dir / "metadata_schema.json").write_text(
        json.dumps(
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "tokens": {
                    "meta": [META_START, META_END],
                    "name": [NAME_START, NAME_END],
                    "description": [DESC_START, DESC_END],
                    "layout": [LAYOUT_START, LAYOUT_END],
                    "song_source_prefix": "songsrc:",
                    "song_prefix": "song:",
                    "custom_song_prefix": "csong:",
                    "name_prefix": "n:",
                    "description_prefix": "d:",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def train_autoregressive(
    dataset_dir: Path, out_model_path: Path, cfg: AutoregressiveConfig
) -> None:
    _log(f"Training token autoregressive baseline from {dataset_dir}")
    read_started = time.monotonic()
    token_sequences: list[list[str]] = []
    clipped_files = 0
    clipped_object_count = 0

    all_files = sorted(dataset_dir.glob("*.gmd"))
    _log(f"Found {len(all_files)} .gmd files")
    for i, path in enumerate(all_files, start=1):
        text = path.read_text(encoding="utf-8", errors="ignore")
        blob = _extract_object_blob(text)
        if blob is None:
            continue

        raw_objects = _extract_raw_objects(blob)
        if not raw_objects:
            continue

        level_name = _extract_level_name(text)
        level_description = _extract_level_description(text)
        song_id, is_custom_song = _extract_song_choice(
            text,
            default_song_id=1,
            max_song_id=cfg.max_song_id,
            max_custom_song_id=cfg.max_custom_song_id,
        )

        if cfg.max_objects_per_file > 0 and len(raw_objects) > cfg.max_objects_per_file:
            clipped_files += 1
            clipped_object_count += len(raw_objects) - cfg.max_objects_per_file
            raw_objects = raw_objects[: cfg.max_objects_per_file]

        objects: list[str] = []
        for raw_obj in raw_objects:
            objects.extend(_object_to_field_tokens(raw_obj, cfg))

        metadata_tokens = _metadata_to_tokens(
            level_name,
            level_description,
            song_id,
            is_custom_song,
            cfg,
        )
        token_sequences.append(
            metadata_tokens + [LAYOUT_START] + objects + [LAYOUT_END]
        )
        if i % 50 == 0:
            _log(f"Parsed {i}/{len(all_files)} files")

    if not token_sequences:
        raise RuntimeError("No usable .gmd files with object strings found")

    total_tokens = sum(len(s) for s in token_sequences)
    _log(
        f"Prepared {len(token_sequences)} object sequences "
        f"({total_tokens} field-tokens) in {time.monotonic() - read_started:.2f}s"
    )

    if cfg.max_objects_per_file > 0:
        _log(
            f"Per-file token cap enabled: {cfg.max_objects_per_file} "
            f"(clipped_files={clipped_files}, clipped_objects={clipped_object_count})"
        )
    else:
        _log("Per-file object cap disabled (using full object sequences)")

    stoi, itos, freq = _build_vocab(
        token_sequences, cfg.max_vocab_size, cfg.min_token_freq
    )
    unk_id = stoi[UNK]
    bos_id = stoi[BOS]
    eos_id = stoi[EOS]

    encoded_ids: list[int] = []
    unk_hits = 0
    for seq in token_sequences:
        encoded_ids.append(bos_id)
        for tok in seq:
            idx = stoi.get(tok, unk_id)
            if idx == unk_id:
                unk_hits += 1
            encoded_ids.append(idx)
        encoded_ids.append(eos_id)

    encoded = torch.tensor(encoded_ids, dtype=torch.long)
    ds = TokenDataset(encoded, cfg.seq_len, cfg.sample_stride)
    if len(ds) == 0:
        raise RuntimeError("Dataset too small for configured seq_len")

    if cfg.num_threads > 0:
        torch.set_num_threads(cfg.num_threads)
        _log(f"Set torch CPU threads to {cfg.num_threads}")

    runtime = resolve_device(cfg.device)
    device = runtime.device

    model_root = out_model_path.parent
    artifacts_dir = model_root / cfg.artifacts_subdir
    checkpoints_dir = model_root / cfg.checkpoints_subdir
    samples_dir = model_root / cfg.samples_subdir

    if cfg.save_preprocessed_artifacts:
        _save_preprocessed_artifacts(
            artifacts_dir,
            cfg,
            stoi,
            itos,
            token_sequences,
            encoded,
        )
        _log(f"Saved preprocessed artifacts to {artifacts_dir}")

    model = GRULM(len(itos), cfg.embed_dim, cfg.hidden_dim).to(device)
    if (
        cfg.torch_compile
        and hasattr(torch, "compile")
        and runtime.backend
        in (
            "cpu",
            "cuda",
            "rocm",
        )
    ):
        try:
            _log("Compiling model with torch.compile for faster training")
            model = torch.compile(model)
        except Exception as exc:
            _log(f"torch.compile unavailable/failed, continuing uncompiled: {exc}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    start_epoch = 0
    global_step = 0

    if cfg.resume_checkpoint:
        resume_path = Path(cfg.resume_checkpoint)
        ckpt = torch.load(resume_path, map_location="cpu")
        _load_state_dict_flexible(model, ckpt["state_dict"])
        if "optimizer_state_dict" in ckpt:
            opt.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0))
        global_step = int(ckpt.get("global_step", 0))
        _log(
            f"Resumed from checkpoint {resume_path} "
            f"(epoch={start_epoch}, global_step={global_step})"
        )

    steps_per_epoch = len(ds) // max(1, cfg.batch_size)
    if steps_per_epoch <= 0:
        steps_per_epoch = 1
    if cfg.max_steps_per_epoch > 0:
        steps_per_epoch = min(steps_per_epoch, cfg.max_steps_per_epoch)

    _log(
        "Start training "
        f"(samples={len(ds)}, vocab={len(itos)}, unique_field_tokens={len(freq)}, "
        f"epochs={cfg.epochs}, seq_len={cfg.seq_len}, stride={cfg.sample_stride}, "
        f"batch_size={cfg.batch_size}, steps_per_epoch={steps_per_epoch}, "
        f"unk_hits={unk_hits}, device={device}, backend={runtime.backend})"
    )

    model.train()
    encoded_device = encoded
    for epoch in range(start_epoch, cfg.epochs):
        epoch_start = time.monotonic()
        epoch_loss = 0.0
        steps = 0

        for _ in range(steps_per_epoch):
            x, y = _make_random_batch(
                encoded_device,
                cfg.seq_len,
                cfg.batch_size,
                stride=cfg.sample_stride,
                device=device,
            )

            logits, _ = model(x)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.reshape(-1)
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            epoch_loss += float(loss.item())
            steps += 1
            global_step += 1

            if steps % max(1, cfg.log_every_steps) == 0:
                elapsed = time.monotonic() - epoch_start
                it_s = steps / max(1e-6, elapsed)
                tok_s = (steps * cfg.batch_size * cfg.seq_len) / max(1e-6, elapsed)
                _log(
                    f"Epoch {epoch + 1}/{cfg.epochs} step {steps}/{steps_per_epoch} "
                    f"loss={loss.item():.5f} speed={it_s:.2f} it/s ({tok_s:.0f} tok/s)"
                )

        elapsed = time.monotonic() - epoch_start
        epoch_tok_s = (steps * cfg.batch_size * cfg.seq_len) / max(1e-6, elapsed)
        _log(
            f"Epoch {epoch + 1}/{cfg.epochs} done "
            f"loss={epoch_loss / max(1, steps):.5f} steps={steps} elapsed={elapsed:.2f}s "
            f"throughput={epoch_tok_s:.0f} tok/s"
        )

        if cfg.save_checkpoint_every_epochs > 0 and (
            (epoch + 1) % cfg.save_checkpoint_every_epochs == 0
        ):
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoints_dir / f"epoch_{epoch + 1:04d}.pt"
            torch.save(
                _checkpoint_payload(
                    model,
                    opt,
                    cfg,
                    stoi,
                    itos,
                    epoch + 1,
                    global_step,
                ),
                checkpoint_path,
            )
            _log(f"Saved checkpoint to {checkpoint_path}")

        if cfg.save_samples_every_epochs > 0 and (
            (epoch + 1) % cfg.save_samples_every_epochs == 0
        ):
            samples_dir.mkdir(parents=True, exist_ok=True)
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            preview_state_path = checkpoints_dir / "_preview_state.pt"
            torch.save(
                _checkpoint_payload(
                    model,
                    opt,
                    cfg,
                    stoi,
                    itos,
                    epoch + 1,
                    global_step,
                ),
                preview_state_path,
            )
            preview_records: list[dict[str, object]] = []
            for sample_idx in range(max(1, cfg.samples_per_epoch)):
                sample_path = (
                    samples_dir
                    / f"epoch_{epoch + 1:04d}_sample_{sample_idx + 1:02d}.gmd"
                )
                sample_stats = sample_autoregressive(
                    model_path=preview_state_path,
                    out_path=sample_path,
                    level_name=None,
                    level_description=None,
                    song_id=None,
                    device_override=cfg.device,
                    seed=(epoch + 1) * 1000 + sample_idx,
                    max_new_tokens=cfg.sample_preview_max_new_tokens,
                    sample_log_every_tokens=0,
                )
                sample_stats["sample_path"] = str(sample_path)
                preview_records.append(sample_stats)

            preview_json_path = samples_dir / f"epoch_{epoch + 1:04d}_summary.json"
            preview_json_path.write_text(
                json.dumps(
                    {
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "samples": preview_records,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            _log(f"Saved sample previews to {preview_json_path}")
            if preview_state_path.exists():
                preview_state_path.unlink()

    out_model_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict_model = _unwrap_model(model)
    torch.save(
        {
            "model_type": "autoregressive",
            "representation": "object_token",
            "state_dict": state_dict_model.state_dict(),
            "cfg": cfg.__dict__,
            "stoi": stoi,
            "itos": itos,
        },
        out_model_path,
    )
    _log(f"Saved autoregressive model to {out_model_path}")


def _build_gmd_blob(
    level_name: str,
    level_description: str,
    objects_blob: str,
    song_id: int,
    is_custom_song: bool,
) -> str:
    desc = base64.b64encode(level_description.encode("utf-8")).decode("ascii")
    creator = "gdlevelai"
    song_tag = "\t<k>k8</k><i>0</i>"
    if is_custom_song:
        song_tag += f"\t<k>k45</k><i>{song_id}</i>"
    else:
        song_tag = f"\t<k>k8</k><i>{song_id}</i>"
    return (
        "<d>"
        "\t<k>k1</k><i>0</i>"
        f"\t<k>k2</k><s>{level_name}</s>"
        f"\t<k>k3</k><s>{desc}</s>"
        f"\t<k>k4</k><s>{objects_blob}</s>"
        f"\t<k>k5</k><s>{creator}</s>"
        f"{song_tag}"
        "\t<k>k9</k><i>0</i>"
        "</d>"
    )


def _sample_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int | None,
    banned_ids: set[int] | None = None,
) -> torch.Tensor:
    logits = logits / max(1e-6, temperature)
    if banned_ids:
        for idx in banned_ids:
            if 0 <= idx < logits.shape[-1]:
                logits[..., idx] = -1e9
    if top_k is not None and top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
        cutoff = values[..., -1, None]
        logits = torch.where(logits < cutoff, torch.full_like(logits, -1e9), logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _normalize_state_dict_keys(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(key.startswith("_orig_mod.") for key in keys):
        return {
            key.removeprefix("_orig_mod."): value for key, value in state_dict.items()
        }
    return state_dict


def _words_to_text(words: list[str], fallback: str) -> str:
    if not words:
        return fallback
    text = " ".join(words).strip()
    return text if text else fallback


def _parse_generated_metadata(
    field_tokens: list[str],
    cfg: AutoregressiveConfig,
) -> tuple[str, str, int, bool]:
    song_id = 1
    is_custom_song = False
    in_name = False
    in_desc = False
    name_words: list[str] = []
    desc_words: list[str] = []

    for tok in field_tokens:
        if tok == "songsrc:custom":
            is_custom_song = True
            continue
        if tok == "songsrc:official":
            is_custom_song = False
            continue
        if tok.startswith("song:"):
            song_id = max(0, min(cfg.max_song_id, _safe_int(tok[5:], 1)))
            is_custom_song = False
            continue
        if tok.startswith("csong:"):
            song_id = max(1, min(cfg.max_custom_song_id, _safe_int(tok[6:], 1)))
            is_custom_song = True
            continue
        if tok == NAME_START:
            in_name = True
            in_desc = False
            continue
        if tok == NAME_END:
            in_name = False
            continue
        if tok == DESC_START:
            in_desc = True
            in_name = False
            continue
        if tok == DESC_END:
            in_desc = False
            continue

        if in_name and tok.startswith("n:"):
            name_words.append(tok[2:])
        elif in_desc and tok.startswith("d:"):
            desc_words.append(tok[2:])

    level_name = _words_to_text(name_words[: cfg.name_max_words], "Untitled")
    level_description = _words_to_text(
        desc_words[: cfg.desc_max_words],
        "Generated by gdlevelai autoregressive baseline",
    )
    return level_name, level_description, song_id, is_custom_song


def _token_ids_with_prefix(stoi: dict[str, int], prefix: str) -> list[int]:
    ids = [idx for tok, idx in stoi.items() if tok.startswith(prefix)]
    ids.sort()
    return ids


def sample_autoregressive(
    model_path: Path,
    out_path: Path,
    level_name: str | None,
    *,
    device_override: str | None = None,
    seed: int | None = None,
    max_new_tokens: int = 4000,
    temperature: float = 1.0,
    top_k: int = 64,
    min_tokens_before_eos: int = 10,
    sample_log_every_tokens: int = 250,
    ban_special_tokens: bool = True,
    song_id: int | None = None,
    custom_song_id: int | None = None,
    level_description: str | None = None,
) -> dict[str, object]:
    _log(
        f"Sampling autoregressive model from {model_path} "
        f"(max_new_tokens={max_new_tokens}, temperature={temperature}, top_k={top_k})"
    )
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        _log(f"Using deterministic seed={seed}")

    checkpoint = torch.load(model_path, map_location="cpu")
    if checkpoint.get("model_type") != "autoregressive":
        raise RuntimeError("Model is not autoregressive; use --model-type diffusion")

    if checkpoint.get("representation") not in (None, "object_token"):
        raise RuntimeError("Unsupported autoregressive representation in checkpoint")

    cfg = AutoregressiveConfig(**checkpoint["cfg"])
    itos: list[str] = checkpoint["itos"]
    stoi: dict[str, int] = checkpoint["stoi"]

    runtime = resolve_device(device_override or cfg.device)
    device = runtime.device

    model = GRULM(len(itos), cfg.embed_dim, cfg.hidden_dim).to(device)
    _load_state_dict_flexible(model, checkpoint["state_dict"])
    model.eval()

    pad_id = stoi.get(PAD, -1)
    bos_id = stoi.get(BOS, 1)
    eos_id = stoi.get(EOS, 2)
    unk_id = stoi.get(UNK, 3)
    banned = {pad_id, bos_id} if ban_special_tokens else None

    meta_start_id = stoi.get(META_START)
    meta_end_id = stoi.get(META_END)
    name_start_id = stoi.get(NAME_START)
    name_end_id = stoi.get(NAME_END)
    desc_start_id = stoi.get(DESC_START)
    desc_end_id = stoi.get(DESC_END)
    layout_start_id = stoi.get(LAYOUT_START)
    layout_end_id = stoi.get(LAYOUT_END)
    song_source_token_ids = _token_ids_with_prefix(stoi, "songsrc:")
    song_token_ids = _token_ids_with_prefix(stoi, "song:")
    custom_song_token_ids = _token_ids_with_prefix(stoi, "csong:")
    name_token_ids = _token_ids_with_prefix(stoi, "n:")
    desc_token_ids = _token_ids_with_prefix(stoi, "d:")

    metadata_scaffold_available = (
        meta_start_id is not None
        and meta_end_id is not None
        and name_start_id is not None
        and name_end_id is not None
        and desc_start_id is not None
        and desc_end_id is not None
    )
    metadata_guidance = metadata_scaffold_available and bool(
        song_source_token_ids and (song_token_ids or custom_song_token_ids)
    )

    generated_ids: list[int] = []
    x = torch.tensor([[bos_id]], dtype=torch.long, device=device)
    h = None
    sample_start = time.monotonic()

    model_name = "Untitled"
    model_description = "Generated by gdlevelai autoregressive baseline"
    model_song_id = 1
    model_is_custom_song = False
    stop_reason = "max_new_tokens"

    with torch.no_grad():

        def emit_next(
            target_ids: list[int],
            *,
            forced_id: int | None = None,
            allowed_ids: list[int] | None = None,
            banned_extra: set[int] | None = None,
        ) -> int:
            nonlocal x, h
            logits, h = model(x, h)
            step_logits = logits[:, -1, :]

            if forced_id is not None:
                idx = int(forced_id)
            else:
                dynamic_banned: set[int] | None = None
                if banned or banned_extra:
                    dynamic_banned = set()
                    if banned:
                        dynamic_banned.update(banned)
                    if banned_extra:
                        dynamic_banned.update(banned_extra)

                if allowed_ids:
                    masked_logits = torch.full_like(step_logits, -1e9)
                    allowed = torch.tensor(allowed_ids, dtype=torch.long, device=device)
                    masked_logits[:, allowed] = step_logits[:, allowed]
                    next_idx = _sample_token(masked_logits, temperature, top_k)
                else:
                    next_idx = _sample_token(
                        step_logits,
                        temperature,
                        top_k,
                        banned_ids=dynamic_banned,
                    )
                idx = int(next_idx.item())

            target_ids.append(idx)
            x = torch.tensor([[idx]], dtype=torch.long, device=device)
            return idx

        metadata_candidate_ids: list[int] = []
        if metadata_guidance:
            emit_next(metadata_candidate_ids, forced_id=meta_start_id)
            emit_next(metadata_candidate_ids, allowed_ids=song_source_token_ids)

            source_token = None
            if metadata_candidate_ids:
                source_idx = metadata_candidate_ids[-1]
                if 0 <= source_idx < len(itos):
                    source_token = itos[source_idx]
            if source_token == "songsrc:custom" and custom_song_token_ids:
                emit_next(metadata_candidate_ids, allowed_ids=custom_song_token_ids)
            elif song_token_ids:
                emit_next(metadata_candidate_ids, allowed_ids=song_token_ids)
            elif custom_song_token_ids:
                emit_next(metadata_candidate_ids, allowed_ids=custom_song_token_ids)

            emit_next(metadata_candidate_ids, forced_id=name_start_id)
            name_allowed = list(name_token_ids)
            name_emitted = 0
            for _ in range(max(1, cfg.name_max_words)):
                allow_end = name_emitted >= max(0, cfg.name_min_words)
                allowed = list(name_allowed)
                if allow_end:
                    allowed.append(name_end_id)
                idx = emit_next(metadata_candidate_ids, allowed_ids=allowed)
                if idx == name_end_id:
                    break
                name_emitted += 1
            if metadata_candidate_ids[-1] != name_end_id:
                emit_next(metadata_candidate_ids, forced_id=name_end_id)
            if name_emitted == 0:
                _log("Model emitted empty name metadata; using fallback if needed")

            emit_next(metadata_candidate_ids, forced_id=desc_start_id)
            desc_allowed = list(desc_token_ids)
            desc_emitted = 0
            for _ in range(max(1, cfg.desc_max_words)):
                allow_end = desc_emitted >= max(0, cfg.desc_min_words)
                allowed = list(desc_allowed)
                if allow_end:
                    allowed.append(desc_end_id)
                idx = emit_next(metadata_candidate_ids, allowed_ids=allowed)
                if idx == desc_end_id:
                    break
                desc_emitted += 1
            if metadata_candidate_ids[-1] != desc_end_id:
                emit_next(metadata_candidate_ids, forced_id=desc_end_id)
            if desc_emitted == 0:
                _log(
                    "Model emitted empty description metadata; using fallback if needed"
                )

            emit_next(metadata_candidate_ids, forced_id=meta_end_id)

            metadata_candidate_tokens = [
                itos[i] for i in metadata_candidate_ids if 0 <= i < len(itos)
            ]
            (
                model_name,
                model_description,
                model_song_id,
                model_is_custom_song,
            ) = _parse_generated_metadata(metadata_candidate_tokens, cfg)

        if custom_song_id is not None:
            resolved_is_custom_song = True
            resolved_song_id = max(1, min(cfg.max_custom_song_id, int(custom_song_id)))
        elif song_id is not None:
            resolved_is_custom_song = False
            resolved_song_id = max(0, min(cfg.max_song_id, int(song_id)))
        else:
            resolved_is_custom_song = model_is_custom_song
            resolved_song_id = model_song_id

        resolved_level_name = level_name if level_name else model_name
        resolved_level_description = (
            level_description if level_description is not None else model_description
        )

        generated_ids.clear()
        x = torch.tensor([[bos_id]], dtype=torch.long, device=device)
        h = None

        if metadata_scaffold_available:
            metadata_prompt_tokens = _metadata_to_tokens(
                resolved_level_name,
                resolved_level_description,
                resolved_song_id,
                resolved_is_custom_song,
                cfg,
            )
            for tok in metadata_prompt_tokens:
                emit_next(generated_ids, forced_id=stoi.get(tok, unk_id))

        if layout_start_id is not None:
            emit_next(generated_ids, forced_id=layout_start_id)

        remaining_tokens = max(0, max_new_tokens - len(generated_ids))
        for _ in range(remaining_tokens):
            idx = emit_next(generated_ids)

            if layout_end_id is not None and idx == layout_end_id:
                _log(f"Reached layout end at token {len(generated_ids)}")
                stop_reason = "layout_end"
                break
            if idx == eos_id and len(generated_ids) > max(0, min_tokens_before_eos):
                _log(f"Reached EOS at token {len(generated_ids)}")
                stop_reason = "eos"
                break

            if (
                sample_log_every_tokens > 0
                and len(generated_ids) % sample_log_every_tokens == 0
            ):
                elapsed = time.monotonic() - sample_start
                tps = len(generated_ids) / max(1e-6, elapsed)
                _log(
                    f"Sampling progress {len(generated_ids)}/{max_new_tokens} tokens ({tps:.1f} tok/s)"
                )

    field_tokens = [itos[i] for i in generated_ids if i < len(itos)]

    objects: list[str] = []
    current: dict[str, int] | None = None
    for tok in field_tokens:
        if tok == OBJ_START:
            current = {"id": 1, "x": 0, "y": 0, "r": 0}
            continue
        if tok == OBJ_END:
            if current is not None:
                objects.append(
                    f"1,{current['id']},2,{current['x']},3,{current['y']},6,{current['r']}"
                )
            current = None
            continue

        if current is None:
            continue

        if tok.startswith("id:"):
            current["id"] = max(1, _safe_int(tok[3:], 1))
        elif tok.startswith("x:"):
            current["x"] = max(0, _safe_int(tok[2:], 0))
        elif tok.startswith("y:"):
            current["y"] = max(0, _safe_int(tok[2:], 0))
        elif tok.startswith("r:"):
            current["r"] = _safe_int(tok[2:], 0)

    if not objects:
        objects = ["1,1,2,30,3,30,6,0"]

    blob = ";".join(objects) + ";"
    gmd_text = _build_gmd_blob(
        resolved_level_name,
        resolved_level_description,
        blob,
        resolved_song_id,
        resolved_is_custom_song,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(gmd_text, encoding="utf-8")
    _log(
        f"Generated autoregressive level at {out_path} "
        f"(objects={len(objects)}, "
        f"{'custom_song_id' if resolved_is_custom_song else 'song_id'}={resolved_song_id}, "
        f"name='{resolved_level_name}', backend={runtime.backend})"
    )
    return {
        "level_name": resolved_level_name,
        "level_description": resolved_level_description,
        "song_id": resolved_song_id,
        "is_custom_song": resolved_is_custom_song,
        "object_count": len(objects),
        "stop_reason": stop_reason,
        "tokens_generated": len(generated_ids),
        "backend": runtime.backend,
    }
