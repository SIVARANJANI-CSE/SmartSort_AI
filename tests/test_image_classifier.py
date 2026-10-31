import io
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from image_classifier import (
    dataset_from_directory,
    dataset_from_zip,
    load_training_result,
    predict_image,
    predict_images,
    save_training_result,
    train_classifier,
)


def image_bytes(brightness: int) -> bytes:
    output = io.BytesIO()
    Image.fromarray(np.full((24, 24), brightness, dtype=np.uint8)).save(output, format="PNG")
    return output.getvalue()


class DatasetTests(unittest.TestCase):
    def test_discovers_nested_directory_and_trains(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "DATASET"
            for split in ("TRAIN", "TEST"):
                for label, brightness in (("O", 20), ("R", 230)):
                    class_dir = root / split / label
                    class_dir.mkdir(parents=True)
                    count = 4 if split == "TRAIN" else 2
                    for index in range(count):
                        (class_dir / f"{index}.png").write_bytes(image_bytes(brightness + index))

            dataset = dataset_from_directory(Path(temporary_directory))
            result = train_classifier(dataset, per_class_limit=10)
            predicted, _ = predict_image(result.model, image_bytes(25))

            model_path = Path(temporary_directory) / "models" / "image_classifier.joblib"
            save_training_result(result, model_path)
            restored_result = load_training_result(model_path)

            self.assertEqual(dataset.classes, ("O", "R"))
            self.assertEqual(result.training_count, 8)
            self.assertEqual(result.validation_count, 4)
            self.assertEqual(predicted, "O")
            self.assertEqual(restored_result.model.labels, result.model.labels)
            self.assertEqual(restored_result.accuracy, result.accuracy)

    def test_reads_labeled_zip(self) -> None:
        archive_bytes = io.BytesIO()
        with zipfile.ZipFile(archive_bytes, "w") as archive:
            for split in ("TRAIN", "TEST"):
                for label, brightness in (("O", 20), ("R", 230)):
                    count = 2 if split == "TRAIN" else 1
                    for index in range(count):
                        archive.writestr(
                            f"dataset/{split}/{label}/{index}.png",
                            image_bytes(brightness + index),
                        )

        dataset = dataset_from_zip(archive_bytes.getvalue())

        self.assertEqual(dataset.classes, ("O", "R"))
        self.assertEqual(len(dataset.train_records), 4)
        self.assertEqual(len(dataset.test_records), 2)

    def test_predicts_multiple_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "DATASET"
            for label, brightness in (("O", 20), ("R", 230)):
                class_dir = root / "TRAIN" / label
                class_dir.mkdir(parents=True)
                for index in range(4):
                    (class_dir / f"{index}.png").write_bytes(image_bytes(brightness + index))

            dataset = dataset_from_directory(Path(temporary_directory))
            result = train_classifier(dataset, per_class_limit=10)
            predictions = predict_images(
                result.model,
                [image_bytes(25), image_bytes(235), image_bytes(25), image_bytes(235)],
            )

            self.assertEqual(len(predictions), 4)
            self.assertTrue(all(label in {"O", "R"} for label, _ in predictions))
            self.assertTrue(all(0.0 <= score <= 1.0 for _, score in predictions))

    def test_reclassifies_bottle_like_images_as_recyclable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "DATASET"
            for split in ("TRAIN", "TEST"):
                for label, brightness in (("O", 20), ("R", 230)):
                    class_dir = root / split / label
                    class_dir.mkdir(parents=True)
                    for index in range(4):
                        (class_dir / f"{index}.png").write_bytes(image_bytes(brightness + index))

            dataset = dataset_from_directory(Path(temporary_directory))
            result = train_classifier(dataset, per_class_limit=10)

            bottle = np.zeros((64, 64, 3), dtype=np.uint8)
            bottle[:, :, 0] = 30
            bottle[:, :, 1] = 120
            bottle[:, :, 2] = 200
            bottle[:, ::4] = (180, 200, 220)
            bottle_image = io.BytesIO()
            Image.fromarray(bottle).save(bottle_image, format="PNG")

            predicted, score = predict_image(result.model, bottle_image.getvalue())
            self.assertEqual(predicted, "R")
            self.assertGreater(score, 0.5)

    def test_rejects_zip_path_traversal(self) -> None:
        archive_bytes = io.BytesIO()
        with zipfile.ZipFile(archive_bytes, "w") as archive:
            archive.writestr("../TRAIN/O/image.png", image_bytes(20))

        with self.assertRaisesRegex(ValueError, "unsafe file path"):
            dataset_from_zip(archive_bytes.getvalue())


if __name__ == "__main__":
    unittest.main()
