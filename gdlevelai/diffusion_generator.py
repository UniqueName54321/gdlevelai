from __future__ import annotations

import base64
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .device_support import resolve_device


OBJ_RE = re.compile(r"<k>k4</k>\s*<s>(.*?)</s>", re.DOTALL)


def _log(message: str) -> None:
    print(f"[diffusion_generator] {message}", flush=True)


@dataclass
class DiffusionConfig:
    max_objects: int = 256
    feature_dim: int = 4
    timesteps: int = 500
    beta_start: float = 1e-4
    beta_end: float = 0.02
    hidden_dim: int = 512
    epochs: int = 10
    batch_size: int = 16
    lr: float = 2e-4
    device: str = "auto"
    id_scale: float = 2000.0
    x_scale: float = 50000.0
    y_scale: float = 50000.0
    rotation_scale: float = 360.0
    log_every_steps: int = 20
    dataloader_shuffle: bool = True
    dataloader_drop_last: bool = True
    dataloader_workers: int = 0
    num_threads: int = 0


def _parse_object_string(gmd_text: str) -> list[list[float]]:
    match = OBJ_RE.search(gmd_text)
    if not match:
        return []
    raw = match.group(1)
    objects = []
    for obj in raw.split(";"):
        if not obj:
            continue
        parts = obj.split(",")
        obj_map: dict[str, str] = {}
        for i in range(0, len(parts) - 1, 2):
            obj_map[parts[i]] = parts[i + 1]

        try:
            object_id = float(obj_map.get("1", "0"))
            x = float(obj_map.get("2", "0"))
            y = float(obj_map.get("3", "0"))
            rotation = float(obj_map.get("6", "0"))
        except ValueError:
            continue

        objects.append([object_id, x, y, rotation])
    return objects


def _vectorize_objects(
    objects: list[list[float]], cfg: DiffusionConfig
) -> torch.Tensor:
    vec = torch.zeros(cfg.max_objects, cfg.feature_dim)
    id_scale = max(1e-6, cfg.id_scale)
    x_scale = max(1e-6, cfg.x_scale)
    y_scale = max(1e-6, cfg.y_scale)
    rotation_scale = max(1e-6, cfg.rotation_scale)
    for i, obj in enumerate(objects[: cfg.max_objects]):
        vec[i, 0] = obj[0] / id_scale
        vec[i, 1] = obj[1] / x_scale
        vec[i, 2] = obj[2] / y_scale
        vec[i, 3] = obj[3] / rotation_scale
    return vec.view(-1)


class GMDDataset(Dataset[torch.Tensor]):
    def __init__(self, paths: Iterable[Path], cfg: DiffusionConfig) -> None:
        self.items: list[torch.Tensor] = []
        parsed_count = 0
        for path in paths:
            text = path.read_text(encoding="utf-8", errors="ignore")
            objects = _parse_object_string(text)
            if objects:
                self.items.append(_vectorize_objects(objects, cfg))
                parsed_count += 1
        _log(f"Loaded dataset entries: {parsed_count}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.items[index]


class NoisePredictor(nn.Module):
    def __init__(self, vector_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vector_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, vector_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = t.float().unsqueeze(1)
        return self.net(torch.cat([x, t], dim=1))


def _build_schedule(
    timesteps: int,
    device: object,
    beta_start: float,
    beta_end: float,
) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps, device=device)


def train_diffusion(
    dataset_dir: Path, out_model_path: Path, cfg: DiffusionConfig
) -> None:
    _log(f"Training started with dataset_dir={dataset_dir}")
    paths = sorted(dataset_dir.glob("*.gmd"))
    _log(f"Found {len(paths)} .gmd files for training")
    ds = GMDDataset(paths, cfg)
    if len(ds) == 0:
        raise RuntimeError("No usable .gmd files found in dataset directory")

    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=cfg.dataloader_shuffle,
        drop_last=cfg.dataloader_drop_last,
        num_workers=max(0, cfg.dataloader_workers),
    )
    if cfg.num_threads > 0:
        torch.set_num_threads(cfg.num_threads)
        _log(f"Set torch CPU threads to {cfg.num_threads}")

    runtime = resolve_device(cfg.device)
    device = runtime.device

    vec_dim = cfg.max_objects * cfg.feature_dim
    model = NoisePredictor(vec_dim, cfg.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    betas = _build_schedule(cfg.timesteps, device, cfg.beta_start, cfg.beta_end)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    model.train()
    _log(f"Diffusion training backend={runtime.backend}, device={device}")
    for epoch in range(cfg.epochs):
        _log(f"Epoch {epoch + 1}/{cfg.epochs} started")
        step = 0
        epoch_loss_sum = 0.0
        for clean in dl:
            step += 1
            clean = clean.to(device)
            t = torch.randint(0, cfg.timesteps, (clean.size(0),)).to(device)

            noise = torch.randn_like(clean)
            alpha_bar_t = alpha_bars[t].unsqueeze(1)
            noisy = (
                torch.sqrt(alpha_bar_t) * clean + torch.sqrt(1.0 - alpha_bar_t) * noise
            )

            pred = model(noisy, t / float(cfg.timesteps))
            loss = nn.functional.mse_loss(pred, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss_sum += float(loss.item())
            if cfg.log_every_steps > 0 and step % cfg.log_every_steps == 0:
                _log(
                    f"Epoch {epoch + 1}/{cfg.epochs} step {step}: "
                    f"loss={loss.item():.6f}"
                )

        mean_loss = epoch_loss_sum / max(1, step)
        _log(
            f"Epoch {epoch + 1}/{cfg.epochs} completed "
            f"(steps={step}, mean_loss={mean_loss:.6f})"
        )

    out_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "cfg": cfg.__dict__,
        },
        out_model_path,
    )
    _log(f"Training finished. Model saved to {out_model_path}")


def _decode_vector_to_objects(vec: torch.Tensor, cfg: DiffusionConfig) -> str:
    arr = vec.view(cfg.max_objects, cfg.feature_dim).cpu()
    objects: list[str] = []
    id_scale = max(1.0, cfg.id_scale)
    x_scale = max(1.0, cfg.x_scale)
    y_scale = max(1.0, cfg.y_scale)
    rotation_scale = max(1.0, cfg.rotation_scale)
    for row in arr:
        object_id = int(max(1, min(int(id_scale), round(float(row[0]) * id_scale))))
        x = int(max(0, min(int(x_scale), round(float(row[1]) * x_scale))))
        y = int(max(0, min(int(y_scale), round(float(row[2]) * y_scale))))
        rot = int(
            max(
                -int(rotation_scale),
                min(int(rotation_scale), round(float(row[3]) * rotation_scale)),
            )
        )

        if x < 5 and y < 5:
            continue
        objects.append(f"1,{object_id},2,{x},3,{y},6,{rot}")

    if not objects:
        objects.append("1,1,2,30,3,30,6,0")
    return ";".join(objects) + ";"


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


def sample_level(
    model_path: Path,
    out_path: Path,
    level_name: str | None,
    seed: int | None = None,
    device_override: str | None = None,
    timesteps_override: int | None = None,
    sample_log_every_steps: int = 0,
    song_id: int | None = None,
    custom_song_id: int | None = None,
    level_description: str | None = None,
) -> tuple[str, str, int, bool]:
    _log(f"Sampling started using model_path={model_path}")
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        _log(f"Using deterministic seed={seed}")

    checkpoint = torch.load(model_path, map_location="cpu")
    cfg = DiffusionConfig(**checkpoint["cfg"])
    runtime = resolve_device(device_override or cfg.device)
    device = runtime.device

    vec_dim = cfg.max_objects * cfg.feature_dim
    model = NoisePredictor(vec_dim, cfg.hidden_dim).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    sample_timesteps = (
        timesteps_override if timesteps_override is not None else cfg.timesteps
    )
    if sample_timesteps <= 0:
        raise RuntimeError("timesteps must be > 0")

    betas = _build_schedule(sample_timesteps, device, cfg.beta_start, cfg.beta_end)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    _log(f"Diffusion sampling backend={runtime.backend}, device={device}")
    x = torch.randn(1, vec_dim).to(device)
    with torch.no_grad():
        progress_every = (
            sample_log_every_steps
            if sample_log_every_steps > 0
            else max(1, sample_timesteps // 10)
        )
        for t in reversed(range(sample_timesteps)):
            t_tensor = torch.tensor([t / float(sample_timesteps)]).to(device)
            pred_noise = model(x, t_tensor)
            alpha = alphas[t]
            alpha_bar = alpha_bars[t]
            beta = betas[t]

            x = (1 / torch.sqrt(alpha)) * (
                x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * pred_noise
            )
            if t > 0:
                x = x + torch.sqrt(beta) * torch.randn_like(x)
            if t % progress_every == 0:
                _log(f"Sampling progress: t={t}/{sample_timesteps}")

    if custom_song_id is not None:
        resolved_song_id = max(1, int(custom_song_id))
        resolved_is_custom_song = True
    else:
        resolved_song_id = max(0, int(song_id)) if song_id is not None else 1
        resolved_is_custom_song = False
    resolved_level_name = level_name if level_name else "Diffusion Generated"
    resolved_level_description = (
        level_description
        if level_description is not None
        else "Generated by gdlevelai diffusion"
    )
    objects_blob = _decode_vector_to_objects(x[0], cfg)
    gmd_text = _build_gmd_blob(
        resolved_level_name,
        resolved_level_description,
        objects_blob,
        resolved_song_id,
        resolved_is_custom_song,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(gmd_text, encoding="utf-8")
    _log(
        f"Sampling finished. Wrote generated level to {out_path} "
        f"({'custom_song_id' if resolved_is_custom_song else 'song_id'}={resolved_song_id})"
    )
    return (
        resolved_level_name,
        resolved_level_description,
        resolved_song_id,
        resolved_is_custom_song,
    )
