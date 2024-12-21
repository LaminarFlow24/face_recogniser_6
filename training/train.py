import os
import argparse
import joblib
import numpy as np
from PIL import Image
from torchvision import transforms, datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import metrics
from face_recognition import preprocessing, FaceFeaturesExtractor, FaceRecogniser

MODEL_DIR_PATH = 'model'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for training Face Recognition model. Supports anomaly detection using class centers.'
    )
    parser.add_argument('-d', '--dataset-path', help='Path to folder with subfolders of images.', required=True)
    parser.add_argument('--grid-search', action='store_true', help='Enable grid search for Logistic Regression.')
    return parser.parse_args()


def dataset_to_embeddings(dataset, features_extractor):
    transform = transforms.Compose([
        preprocessing.ExifOrientationNormalize(),
        transforms.Resize(1024)
    ])

    embeddings = []
    labels = []
    for img_path, label in dataset.samples:
        print(f"Processing: {img_path}")
        _, embedding = features_extractor(transform(Image.open(img_path).convert('RGB')))
        if embedding is None:
            print(f"Could not find face in {img_path}")
            continue
        if embedding.shape[0] > 1:
            print(f"Multiple faces detected in {img_path}, selecting the most confident.")
            embedding = embedding[0, :]
        embeddings.append(embedding.flatten())
        labels.append(label)

    return np.stack(embeddings), labels


def compute_class_centers(embeddings, labels, class_to_idx):
    """
    Compute mean embeddings (class centers) for each class.

    :param embeddings: Embeddings array (N, D).
    :param labels: List of labels corresponding to embeddings.
    :param class_to_idx: Dictionary mapping class names to indices.
    :return: Dictionary mapping class indices to mean embeddings.
    """
    class_centers = {}
    for class_name, class_idx in class_to_idx.items():
        class_embeddings = np.array([embeddings[i] for i in range(len(labels)) if labels[i] == class_idx])
        class_centers[class_idx] = np.mean(class_embeddings, axis=0)
    return class_centers


def load_data(args, features_extractor):
    dataset = datasets.ImageFolder(args.dataset_path)
    embeddings, labels = dataset_to_embeddings(dataset, features_extractor)
    return embeddings, labels, dataset.class_to_idx


def train(args, embeddings_train, labels_train):
    softmax = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=10000)
    if args.grid_search:
        clf = GridSearchCV(
            estimator=softmax,
            param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
            cv=3
        )
    else:
        clf = softmax
    clf.fit(embeddings_train, labels_train)

    return clf.best_estimator_ if args.grid_search else clf


def main():
    args = parse_args()

    # Initialize face feature extractor
    features_extractor = FaceFeaturesExtractor()

    # Load dataset and compute embeddings
    embeddings, labels, class_to_idx = load_data(args, features_extractor)

    # Perform train-test split
    embeddings_train, embeddings_test, labels_train, labels_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Train the model
    clf = train(args, embeddings_train, labels_train)

    # Compute class centers for anomaly detection
    class_centers = compute_class_centers(embeddings_train, labels_train, class_to_idx)

    # Evaluate the model
    labels_pred = clf.predict(embeddings_test)
    print("Test Set Classification Report:")
    print(metrics.classification_report(labels_test, labels_pred))

    # Save model and class centers
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    if not os.path.isdir(MODEL_DIR_PATH):
        os.mkdir(MODEL_DIR_PATH)
    model_path = os.path.join(MODEL_DIR_PATH, 'face_recogniser.pkl')
    joblib.dump(FaceRecogniser(features_extractor, clf, idx_to_class, class_centers), model_path)
    print(f"Model and class centers saved to {model_path}")


if __name__ == '__main__':
    main()
