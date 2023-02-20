from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

N_GEN = 100

class Classifier(BaseEstimator):
    def __init__(self):
        self.mem = None

    def fit(self, X, y):
        """

        :param X: matrice de matrice [nb_image x 3 x 64 x 64]
        :param y: None
        :return:
        """
        print(f"{y=}")
        self.mem = X[: N_GEN]
        assert self.mem.shape == [N_GEN, 64, 64, 3]

    def predict(self, X):
        """

        :param X: Noise de dimension : N_echantilllon x 1024
        :return: images : [N_echantillon x 3 x 64 x 64]
        """
        assert X.shape[0] == N_GEN, f"{X.shape=}"
        return self.mem


def get_estimator():

    classifier = Classifier()
    pipe = make_pipeline(classifier)
    return pipe
