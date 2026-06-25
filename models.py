from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import (
    LogisticRegression, SGDClassifier, RidgeClassifier,
    Perceptron, PassiveAggressiveClassifier,
)
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, HistGradientBoostingClassifier,
    AdaBoostClassifier, BaggingClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis,
)

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False


def _calibrate(estimator, cv: int = 3):
    return CalibratedClassifierCV(estimator, cv=cv)


def get_models(random_state: int = 42) -> dict:
    rs = random_state
    cw = "balanced"   # kompensacja nierównowagi klas (~17% Oscarów)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=rs, class_weight=cw, solver="lbfgs",
        ),
        "SGD Classifier": _calibrate(
            SGDClassifier(loss="log_loss", max_iter=1000, random_state=rs,
                          class_weight=cw, tol=1e-3), cv=3
        ),
        "Ridge Classifier": _calibrate(
            RidgeClassifier(class_weight=cw), cv=3
        ),
        "Perceptron": _calibrate(
            Perceptron(max_iter=1000, random_state=rs, class_weight=cw), cv=3
        ),
        "Passive Aggressive": _calibrate(
            PassiveAggressiveClassifier(max_iter=1000, random_state=rs,
                                        class_weight=cw), cv=3
        ),
        "SVM (RBF)": SVC(
            kernel="rbf", C=1.0, gamma="scale", probability=True,
            class_weight=cw, random_state=rs,
        ),
        "SVM (Linear)": _calibrate(
            LinearSVC(max_iter=2000, class_weight=cw, random_state=rs), cv=3
        ),
        "NuSVC": NuSVC(
            nu=0.3, kernel="rbf", gamma="scale", probability=True,
            class_weight=cw, random_state=rs,
        ),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5, weights="uniform"),
        "KNN (k=11)": KNeighborsClassifier(n_neighbors=11, weights="distance"),
        "Gaussian NB": GaussianNB(),
        "Bernoulli NB": BernoulliNB(),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, class_weight=cw, random_state=rs,
        ),
        "Extra Tree": ExtraTreeClassifier(
            max_depth=8, class_weight=cw, random_state=rs,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=2,
            class_weight=cw, random_state=rs, n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=2,
            class_weight=cw, random_state=rs, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=rs,
        ),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=300, max_depth=5, learning_rate=0.05,
            class_weight=cw, random_state=rs,
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=200, learning_rate=0.5, random_state=rs,
        ),
        "Bagging": BaggingClassifier(
            n_estimators=200, random_state=rs, n_jobs=-1,
        ),
        "MLP (100-50)": MLPClassifier(
            hidden_layer_sizes=(100, 50), max_iter=500,
            random_state=rs, early_stopping=True,
        ),
        "MLP (200-100-50)": MLPClassifier(
            hidden_layer_sizes=(200, 100, 50), max_iter=500,
            random_state=rs, early_stopping=True,
        ),
        "LDA": LinearDiscriminantAnalysis(),
        "QDA": QuadraticDiscriminantAnalysis(reg_param=0.1),
    }

    if _XGB_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=rs,
            scale_pos_weight=10, verbosity=0,
        )

    return models


if __name__ == "__main__":
    print(f"Dostępne modele ({len(get_models())}):")
    for i, name in enumerate(get_models(), 1):
        print(f"  {i:2}. {name}")
