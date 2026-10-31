from pathlib import Path
from image_classifier import dataset_from_directory, train_classifier, predict_image


d = dataset_from_directory(Path('DATASET'))
r = train_classifier(d, per_class_limit=5000)
matrix = {'O': {'O': 0, 'R': 0}, 'R': {'O': 0, 'R': 0}}
for label in d.classes:
    base = Path('DATASET') / 'TEST' / label
    for f in sorted(base.glob('*.*')):
        pred, _ = predict_image(r.model, f.read_bytes())
        matrix[label][pred] = matrix[label].get(pred, 0) + 1

print('confusion_matrix', matrix)
print('overall_accuracy', round((matrix['O']['O'] + matrix['R']['R']) / sum(sum(v.values()) for v in matrix.values()), 4))
for label in d.classes:
    sample = []
    for f in sorted((Path('DATASET') / 'TEST' / label).glob('*.*'))[:10]:
        sample.append((f.name, predict_image(r.model, f.read_bytes())[0]))
    print(label, sample)
