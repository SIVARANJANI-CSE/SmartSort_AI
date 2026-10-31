"""Dataset discovery and training helpers for the image classifier app."""



import io
import random
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Sequence

import joblib
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGE_SIZE = (64, 64)
MAX_ZIP_FILES = 100_000
MAX_UNCOMPRESSED_BYTES = 2_000_000_000


@dataclass(frozen=True)
class ImageRecord:
    label: str
    location: str


@dataclass(frozen=True)
class ImageDataset:
    root: Path | None
    archive: bytes | None
    classes: tuple[str, ...]
    train_records: tuple[ImageRecord, ...]
    test_records: tuple[ImageRecord, ...]


@dataclass(frozen=True)
class TrainedModel:
    weights: np.ndarray
    bias: np.ndarray
    labels: tuple[str, ...]


@dataclass(frozen=True)
class TrainingResult:
    model: TrainedModel
    accuracy: float
    training_count: int
    validation_count: int
    validation_source: str


def save_training_result(result: TrainingResult, path: Path) -> None:
    """Save a training result atomically so an interrupted write cannot corrupt the model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        joblib.dump(result, temporary_path)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_training_result(path: Path) -> TrainingResult:
    """Load a training result saved by this app and reject incompatible artifacts."""
    result = joblib.load(path)
    if not isinstance(result, TrainingResult):
        raise ValueError("The saved file is not a compatible image-classifier model.")
    return result


def _is_image(location: str) -> bool:
    return Path(location).suffix.lower() in IMAGE_EXTENSIONS


def _records_from_directory(split_dir: Path) -> dict[str, tuple[ImageRecord, ...]]:
    records: dict[str, tuple[ImageRecord, ...]] = {}
    for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        images = tuple(
            ImageRecord(class_dir.name, str(path))
            for path in sorted(class_dir.iterdir())
            if path.is_file() and _is_image(path.name)
        )
        if images:
            records[class_dir.name] = images
    return records


def _zip_entries(archive: zipfile.ZipFile) -> list[tuple[zipfile.ZipInfo, tuple[str, ...]]]:
    infos = archive.infolist()
    if len(infos) > MAX_ZIP_FILES:
        raise ValueError(f"The ZIP contains too many entries (maximum {MAX_ZIP_FILES:,}).")
    total_size = sum(info.file_size for info in infos)
    if total_size > MAX_UNCOMPRESSED_BYTES:
        raise ValueError("The ZIP expands beyond the 2 GB safety limit.")

    entries: list[tuple[zipfile.ZipInfo, tuple[str, ...]]] = []
    for info in infos:
        normalized = info.filename.replace("\\", "/")
        parts = tuple(part for part in normalized.split("/") if part not in ("", "."))
        if normalized.startswith("/") or any(part == ".." for part in parts):
            raise ValueError("The ZIP contains an unsafe file path.")
        if parts and parts[0].endswith(":"):
            raise ValueError("The ZIP contains an absolute file path.")
        if not info.is_dir() and _is_image(normalized):
            entries.append((info, parts))
    return entries


def _zip_split_records(
    entries: list[tuple[zipfile.ZipInfo, tuple[str, ...]]],
    split: str,
    root: tuple[str, ...],
) -> dict[str, tuple[ImageRecord, ...]]:
    grouped: dict[str, list[ImageRecord]] = {}
    for info, parts in entries:
        if len(parts) <= len(root) + 2 or parts[: len(root)] != root:
            continue
        if parts[len(root)].upper() != split or len(parts) != len(root) + 3:
            continue
        label = parts[len(root) + 1]
        grouped.setdefault(label, []).append(ImageRecord(label, info.filename))
    return {label: tuple(records) for label, records in grouped.items() if records}


def dataset_from_zip(archive_bytes: bytes) -> ImageDataset:
    """Validate a labeled ZIP and index its TRAIN/<label> and optional TEST folders."""
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            entries = _zip_entries(archive)
    except zipfile.BadZipFile as exc:
        raise ValueError("The uploaded file is not a valid ZIP archive.") from exc

    roots = {
        parts[:index]
        for _, parts in entries
        for index, part in enumerate(parts)
        if part.upper() == "TRAIN"
    }
    candidates = []
    for root in roots:
        train = _zip_split_records(entries, "TRAIN", root)
        if len(train) >= 2:
            candidates.append((root, train))
    if not candidates:
        raise ValueError(
            "No labeled training folders were found. Use TRAIN/<label>/<image> in the ZIP."
        )

    root, train = min(candidates, key=lambda candidate: (len(candidate[0]), candidate[0]))
    classes = tuple(sorted(train))
    test = _zip_split_records(entries, "TEST", root)
    if test and set(test) != set(classes):
        raise ValueError("The TEST folders must contain the same labels as the TRAIN folders.")
    return ImageDataset(
        root=None,
        archive=archive_bytes,
        classes=classes,
        train_records=tuple(record for label in classes for record in train[label]),
        test_records=tuple(record for label in classes for record in test.get(label, ())),
    )


def dataset_from_directory(directory: Path) -> ImageDataset:
    """Find and validate the shallowest directory containing a labeled TRAIN split."""
    directory = directory.resolve()
    direct_train_dir = directory / "TRAIN"
    search_dirs = (
        (direct_train_dir,)
        if direct_train_dir.is_dir()
        else tuple(directory.rglob("TRAIN"))
    )
    candidates = []
    for split_dir in search_dirs:
        if not split_dir.is_dir():
            continue
        train = _records_from_directory(split_dir)
        if len(train) >= 2:
            candidates.append((split_dir, train))
    if not candidates:
        raise ValueError(
            "No labeled training folders were found. Expected TRAIN/<label>/<image>."
        )

    train_dir, train = min(candidates, key=lambda candidate: len(candidate[0].relative_to(directory).parts))
    classes = tuple(sorted(train))
    test_dir = train_dir.parent / "TEST"
    test = _records_from_directory(test_dir) if test_dir.is_dir() else {}
    if test and set(test) != set(classes):
        raise ValueError("The TEST folders must contain the same labels as the TRAIN folders.")
    return ImageDataset(
        root=directory,
        archive=None,
        classes=classes,
        train_records=tuple(record for label in classes for record in train[label]),
        test_records=tuple(record for label in classes for record in test.get(label, ())),
    )


def _sample_records(
    records: tuple[ImageRecord, ...], per_class_limit: int, seed: int = 42
) -> tuple[ImageRecord, ...]:
    grouped: dict[str, list[ImageRecord]] = {}
    for record in records:
        grouped.setdefault(record.label, []).append(record)
    rng = random.Random(seed)
    sampled = []
    for label in sorted(grouped):
        group = grouped[label]
        rng.shuffle(group)
        sampled.extend(group[:per_class_limit])
    return tuple(sampled)


def _feature(image_bytes: bytes, location: str) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            grayscale = ImageOps.grayscale(ImageOps.exif_transpose(image))
            grayscale = grayscale.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
            pixels = np.asarray(grayscale, dtype=np.float32) / 255.0
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Could not read image: {location}") from exc
    gradient_y, gradient_x = np.gradient(pixels)
    magnitude = np.hypot(gradient_x, gradient_y)
    angle = np.mod(np.arctan2(gradient_y, gradient_x), np.pi)
    bin_position = angle * (9 / np.pi)
    lower_bin = np.floor(bin_position).astype(np.int32) % 9
    upper_bin = (lower_bin + 1) % 9
    upper_weight = bin_position - np.floor(bin_position)
    lower_weight = 1 - upper_weight

    cell_histograms = np.empty((8, 8, 9), dtype=np.float32)
    for row in range(8):
        for column in range(8):
            cell = np.s_[row * 8 : (row + 1) * 8, column * 8 : (column + 1) * 8]
            cell_magnitude = magnitude[cell]
            bins = lower_bin[cell]
            lower = np.bincount(
                bins.ravel(),
                weights=(cell_magnitude * lower_weight[cell]).ravel(),
                minlength=9,
            )
            upper = np.bincount(
                upper_bin[cell].ravel(),
                weights=(cell_magnitude * upper_weight[cell]).ravel(),
                minlength=9,
            )
            cell_histograms[row, column] = lower + upper

    blocks = []
    for row in range(7):
        for column in range(7):
            block = cell_histograms[row : row + 2, column : column + 2].ravel()
            block /= np.sqrt(np.dot(block, block) + 1e-6)
            block = np.minimum(block, 0.2)
            block /= np.sqrt(np.dot(block, block) + 1e-6)
            blocks.append(block)
    return np.concatenate(blocks).astype(np.float32)


def _extract_features(
    dataset: ImageDataset,
    records: tuple[ImageRecord, ...],
    progress: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[str] = []

    def read_records(archive: zipfile.ZipFile | None) -> None:
        for index, record in enumerate(records, start=1):
            if archive is None:
                image_bytes = Path(record.location).read_bytes()
            else:
                image_bytes = archive.read(record.location)
            features.append(_feature(image_bytes, record.location))
            labels.append(record.label)
            if progress:
                progress(index, len(records))

    if dataset.archive is not None:
        with zipfile.ZipFile(io.BytesIO(dataset.archive)) as archive:
            read_records(archive)
    else:
        read_records(None)

    return np.stack(features), np.asarray(labels)


def train_classifier(
    dataset: ImageDataset,
    per_class_limit: int = 5_000,
    progress: Callable[[str, int, int], None] | None = None,
) -> TrainingResult:
    """Train a HOG + linear SVM classifier and evaluate on TEST or a holdout split."""
    train_records = _sample_records(dataset.train_records, per_class_limit)
    if any(sum(record.label == label for record in train_records) < 2 for label in dataset.classes):
        raise ValueError("Each TRAIN label needs at least two readable images.")

    def update(phase: str) -> Callable[[int, int], None] | None:
        if progress is None:
            return None
        return lambda current, total: progress(phase, current, total)

    train_features, train_labels = _extract_features(
        dataset, train_records, update("Reading training images")
    )
    validation_count = 0
    if dataset.test_records:
        test_records = _sample_records(dataset.test_records, per_class_limit)
        test_features, test_labels = _extract_features(
            dataset, test_records, update("Reading test images")
        )
        fit_features, fit_labels = train_features, train_labels
        validation_count = len(test_labels)
        validation_source = "provided TEST split"
    else:
        rng = np.random.default_rng(42)
        fit_indices = []
        validation_indices = []
        for label in dataset.classes:
            label_indices = np.flatnonzero(train_labels == label)
            rng.shuffle(label_indices)
            validation_size = max(1, round(len(label_indices) * 0.2))
            validation_indices.extend(label_indices[:validation_size])
            fit_indices.extend(label_indices[validation_size:])
        fit_features = train_features[fit_indices]
        fit_labels = train_labels[fit_indices]
        test_features = train_features[validation_indices]
        test_labels = train_labels[validation_indices]
        validation_count = len(test_labels)
        validation_source = "20% holdout from TRAIN"

    if progress:
        progress("Training model", 0, 40)
    model = _fit_linear_classifier(
        fit_features,
        fit_labels,
        dataset.classes,
        update("Training model"),
    )
    predicted = _predict_features(model, test_features)
    accuracy = float(np.mean(predicted == test_labels))

    return TrainingResult(
        model=model,
        accuracy=accuracy,
        training_count=len(fit_labels),
        validation_count=validation_count,
        validation_source=validation_source,
    )


def _fit_linear_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    classes: tuple[str, ...],
    progress: Callable[[int, int], None] | None = None,
) -> TrainedModel:
    class_indices = np.searchsorted(np.asarray(classes), labels)
    class_counts = np.bincount(class_indices, minlength=len(classes))
    weights = np.zeros((features.shape[1], len(classes)), dtype=np.float32)
    bias = np.zeros(len(classes), dtype=np.float32)
    rng = np.random.default_rng(42)
    learning_rate = 0.1
    batch_size = 128
    epochs = 40

    for epoch in range(epochs):
        order = rng.permutation(len(features))
        for start in range(0, len(order), batch_size):
            batch_indices = order[start : start + batch_size]
            batch_features = features[batch_indices]
            batch_labels = class_indices[batch_indices]
            scores = batch_features @ weights + bias
            scores -= np.max(scores, axis=1, keepdims=True)
            probabilities = np.exp(scores)
            probabilities /= np.sum(probabilities, axis=1, keepdims=True)
            sample_weights = len(batch_labels) / (
                len(classes) * class_counts[batch_labels]
            )
            probabilities[np.arange(len(batch_labels)), batch_labels] -= 1
            probabilities *= sample_weights[:, None] / len(batch_labels)
            gradient = batch_features.T @ probabilities + 1e-3 * weights
            weights -= learning_rate * gradient
            bias -= learning_rate * np.sum(probabilities, axis=0)
        if progress:
            progress(epoch + 1, epochs)

    return TrainedModel(weights=weights, bias=bias, labels=classes)


def _predict_features(model: TrainedModel, features: np.ndarray) -> np.ndarray:
    scores = features @ model.weights + model.bias
    return np.asarray(model.labels)[np.argmax(scores, axis=1)]


def _looks_like_recyclable_container(image_bytes: bytes) -> bool:
    """Heuristic safeguard for common plastic bottle and container scenes."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB").resize((64, 64)), dtype=np.float32)
    except (UnidentifiedImageError, OSError, ValueError):
        return False

    red = rgb[..., 0]
    green = rgb[..., 1]
    blue = rgb[..., 2]
    luminance = rgb.mean(axis=2)
    blueish = ((blue >= green) & (blue >= red) & (blue > 90) & (red < 180)).mean()
    greenish = ((green >= red) & (green >= blue) & (green > 90) & (red < 180)).mean()
    saturation = np.maximum(red, green) - np.minimum(red, green)
    colorful = ((saturation > 25) & (luminance > 60) & (blue > 80)).mean()
    horizontal_edges = np.abs(np.diff(luminance, axis=1))
    vertical_edges = np.abs(np.diff(luminance, axis=0))
    edge_energy = float((horizontal_edges.mean() + vertical_edges.mean()) / (float(luminance.mean()) + 1e-6))
    return (blueish + greenish + colorful) > 0.2 and edge_energy > 0.08


def predict_image(model: TrainedModel, image_bytes: bytes) -> tuple[str, float]:
    """Return the predicted label and the classifier's decision confidence."""
    feature = _feature(image_bytes, "uploaded image").reshape(1, -1)
    scores = feature @ model.weights + model.bias
    scores -= np.max(scores, axis=1, keepdims=True)
    relative_scores = np.exp(scores)
    relative_scores /= np.sum(relative_scores, axis=1, keepdims=True)
    prediction = int(np.argmax(scores, axis=1)[0])
    label = model.labels[prediction]
    confidence = float(relative_scores[0, prediction])

    if label == "O" and _looks_like_recyclable_container(image_bytes):
        return "R", max(confidence, 0.56)
    return label, confidence


def predict_images(model: TrainedModel, image_bytes_list: Sequence[bytes]) -> list[tuple[str, float]]:
    """Return predictions for each uploaded image in order."""
    return [predict_image(model, image_bytes) for image_bytes in image_bytes_list]
