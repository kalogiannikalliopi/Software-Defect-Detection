from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

# models
def get_models():
    models = {
        "Logistic Regression": LogisticRegression(
            solver='liblinear', max_iter=10000, C=1, random_state=1
        ),
        "Perceptron": Perceptron(random_state=1),
        "LinearSVC": LinearSVC(C=1.0, random_state=1, max_iter=10000, dual=True),
        "SVC_rbf": SVC(kernel='rbf', C=1.0, random_state=1),
        "Decision Tree": DecisionTreeClassifier(
            criterion='gini', max_depth=3, random_state=1
        ),
        "Random Forest": RandomForestClassifier(
            criterion='gini', n_estimators=20, max_depth=3, random_state=1, n_jobs=2
        ),
        "Neural Network": MLPClassifier(
            activation='relu', hidden_layer_sizes=(800, 100, 50), alpha=0.3, random_state=1
        )
    }
    return models