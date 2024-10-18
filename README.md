# Numbers_Zero_to_Ten_Classification

 The **"Numbers Zero to Ten Classification"** project develops a machine learning model to classify images of handwritten digits from zero to ten. It uses convolutional neural networks (CNNs) to recognize and predict digits based on a labeled dataset.

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.x
- pip (Python package installer)

### Clone the Repository

```bash
git clone https://github.com/do-martin/Numbers_Zero_to_Ten_Classification.git
cd Numbers_Zero_to_Ten_Classification
```

### Install Required Packages

It is recommended to create a virtual environment first. Ensure that you have Python 3.11 or below installed:

#### Your default Python version is 3.11 or below: 

```bash
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
```

#### Alternative: Specify the path to Python 3.11:

```bash
C:\Path\to\python311\python.exe -m venv env
env\Scripts\activate
pip install -r requirements.txt
```

## Usage

The project is implemented in a Jupyter Notebook (.ipynb). Open the notebook in Jupyter Lab, Jupyter Notebook, or Visual Studio Code (VSCode) to explore and run the code.

## Model Evaluation

To facilitate an accurate assessment of the model's performance, please ensure that the image named `number_1.png` is present in the root directory. This image should be used exclusively for evaluating the model's accuracy and must not be utilized during the training phase.


### Guidelines for Sample Images

- The images should be in JPEG or PNG format.
- Ensure that each image name is unique to avoid any confusion.
- Use these images exclusively for evaluation purposes and not for training the model.

## Required Libraries

- TensorFlow
- Keras
- NumPy
- Matplotlib
- Pillow

You can find the specific library versions in the `requirements.txt` file.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Special thanks to the developers of TensorFlow and Keras for their contributions to the field of deep learning.
- Inspiration from various Kaggle competitions that focus on image classification.
