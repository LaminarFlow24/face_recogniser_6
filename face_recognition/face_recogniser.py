from collections import namedtuple
import numpy as np

Prediction = namedtuple('Prediction', 'label confidence')
Face = namedtuple('Face', 'top_prediction bb all_predictions')
BoundingBox = namedtuple('BoundingBox', 'left top right bottom')

def top_prediction(idx_to_class, probs):
    top_label = probs.argmax()
    return Prediction(label=idx_to_class[top_label], confidence=probs[top_label])

def to_predictions(idx_to_class, probs):
    return [Prediction(label=idx_to_class[i], confidence=prob) for i, prob in enumerate(probs)]

class FaceRecogniser:
    def __init__(self, feature_extractor, classifier, idx_to_class, class_centers=None, unknown_threshold=0.4): ################################################################################
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.idx_to_class = idx_to_class
        self.class_centers = class_centers
        self.unknown_threshold = unknown_threshold

    def _is_anomalous(self, embedding):
        if not self.class_centers:
            return False  # Skip anomaly detection if no class centers are provided.

        distances = {
            label: np.linalg.norm(embedding - center)
            for label, center in self.class_centers.items()
        }

        closest_class = min(distances, key=distances.get)
        closest_distance = distances[closest_class]

        return closest_distance > self.unknown_threshold, closest_class, closest_distance

    def recognise_faces(self, img):
        bbs, embeddings = self.feature_extractor(img)
        if bbs is None:
            return []

        predictions = self.classifier.predict_proba(embeddings)

        result_faces = []
        for bb, probs, embedding in zip(bbs, predictions, embeddings):
            is_anomalous, _, distance = self._is_anomalous(embedding)

            if is_anomalous:
                top_pred = Prediction(label="Unknown", confidence=1 - distance / self.unknown_threshold)
            else:
                top_pred = top_prediction(self.idx_to_class, probs)

            result_faces.append(
                Face(
                    top_prediction=top_pred,
                    bb=BoundingBox(left=bb[0], top=bb[1], right=bb[2], bottom=bb[3]),
                    all_predictions=to_predictions(self.idx_to_class, probs)
                )
            )

        return result_faces

    def __call__(self, img):
        return self.recognise_faces(img)
