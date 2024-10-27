{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "from sklearn.datasets import fetch_openml\n",
    "from sklearn.model_selection import train_test_split\n",
    "# Carrega o dataset CIFAR-10\n",
    "cifar10 = fetch_openml('cifar_10', version=1, cache=True)\n",
    "# Separa os dados e os rótulos\n",
    "X = cifar10.data\n",
    "y = cifar10.target\n",
    "# Converte os rótulos para inteiros\n",
    "y = y.astype(np.uint8)\n",
    "# Divide os dados em conjuntos de treinamento e teste\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "# Normaliza os dados\n",
    "X_train = X_train / 255.0\n",
    "X_test = X_test / 255.0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "relatório: DecisionTreeClassifier()in              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.35      0.37      0.36      1193\n",
      "           1       0.31      0.30      0.31      1211\n",
      "           2       0.21      0.22      0.21      1218\n",
      "           3       0.18      0.18      0.18      1208\n",
      "           4       0.21      0.22      0.22      1168\n",
      "           5       0.21      0.20      0.20      1203\n",
      "           6       0.26      0.26      0.26      1185\n",
      "           7       0.28      0.26      0.27      1241\n",
      "           8       0.37      0.39      0.38      1183\n",
      "           9       0.26      0.25      0.26      1190\n",
      "\n",
      "    accuracy                           0.26     12000\n",
      "   macro avg       0.26      0.27      0.26     12000\n",
      "weighted avg       0.26      0.26      0.26     12000\n",
      "\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn import metrics\n",
    "from sklearn import tree\n",
    "modelo = tree.DecisionTreeClassifier()\n",
    "modelo.fit(X_train, y_train)\n",
    "esperado=y_test\n",
    "predito=modelo.predict(X_test)\n",
    "print(\"relatório: %sin%s\\n\" % (modelo, metrics.classification_report(esperado, predito)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "relatório: RidgeClassifier(alpha=0.5)in              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.42      0.48      0.44      1193\n",
      "           1       0.41      0.48      0.44      1211\n",
      "           2       0.29      0.19      0.23      1218\n",
      "           3       0.27      0.19      0.22      1208\n",
      "           4       0.32      0.27      0.29      1168\n",
      "           5       0.31      0.29      0.30      1203\n",
      "           6       0.36      0.43      0.39      1185\n",
      "           7       0.44      0.43      0.43      1241\n",
      "           8       0.44      0.54      0.48      1183\n",
      "           9       0.38      0.45      0.41      1190\n",
      "\n",
      "    accuracy                           0.37     12000\n",
      "   macro avg       0.36      0.37      0.36     12000\n",
      "weighted avg       0.36      0.37      0.36     12000\n",
      "\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn import metrics\n",
    "from sklearn import linear_model\n",
    "modelo = linear_model.RidgeClassifier(alpha=.5)\n",
    "modelo.fit(X_train, y_train)\n",
    "esperado=y_test\n",
    "predito=modelo.predict(X_test)\n",
    "print(\"relatório: %sin%s\\n\" % (modelo, metrics.classification_report(esperado, predito)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "relatório: SGDClassifier(random_state=104)in              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.59      0.30      0.40      1193\n",
      "           1       0.66      0.23      0.34      1211\n",
      "           2       0.27      0.11      0.15      1218\n",
      "           3       0.37      0.04      0.08      1208\n",
      "           4       0.17      0.73      0.28      1168\n",
      "           5       0.40      0.13      0.20      1203\n",
      "           6       0.35      0.38      0.36      1185\n",
      "           7       0.64      0.18      0.28      1241\n",
      "           8       0.56      0.35      0.43      1183\n",
      "           9       0.31      0.68      0.43      1190\n",
      "\n",
      "    accuracy                           0.31     12000\n",
      "   macro avg       0.43      0.31      0.29     12000\n",
      "weighted avg       0.43      0.31      0.29     12000\n",
      "\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn.linear_model import SGDClassifier\n",
    "modelo = SGDClassifier(random_state=104)\n",
    "modelo.fit(X_train, y_train)\n",
    "esperado=y_test\n",
    "predito=modelo.predict(X_test)\n",
    "print(\"relatório: %sin%s\\n\" % (modelo, metrics.classification_report(esperado, predito)))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
