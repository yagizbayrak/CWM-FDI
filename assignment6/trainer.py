import sys
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

EPOCHS = 300

layers = int(sys.argv[1])
neurons = int(sys.argv[2])
seed = int(sys.argv[3])

X, y = load_digits(return_X_y=True)
X = X / 16.0

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

hidden_layers = tuple([neurons] * layers)
model = MLPClassifier(
    hidden_layer_sizes=hidden_layers,
    max_iter=EPOCHS,
    n_iter_no_change=EPOCHS + 1,
    tol=0.0,
    random_state=seed,
)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("ACC", round(accuracy, 4))
