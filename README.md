# Iris ML Classifier

Simple machine learning pipeline for classifying iris flower species using Python and scikit-learn.

This repository demonstrates a minimal and reproducible ML workflow including:
- dataset loading
- preprocessing
- model training
- evaluation

The project is intended for educational purposes and as a minimal example of a machine learning experiment repository.


## Dataset

This project uses the Iris dataset, a classic dataset in machine learning used for classification tasks.

Dataset characteristics:

- Samples: 150
- Features: 4 numerical measurements
  - sepal length
  - sepal width
  - petal length
  - petal width
- Classes:
  - setosa
  - versicolor
  - virginica

The goal is to predict the species of an iris flower based on these measurements.

## Installation

Clone the repository:

```bash
git clone https://github.com/andredemedeiros/iris-ml-classifier.git
cd iris-ml-classifier
```

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Model training and validation

Run the training script:

```bash
python model.py
```

## License

This project is licensed under the MIT License.