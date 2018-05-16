from sift import SIFT


class SimilarFeatures:
    def __init__(self):
        self.sift = SIFT()
        self.sift.get_features()
        self.vq = self.sift.pooling(graph=False)
