# ml_ops_project

Machine Learning Operation Project

## Project description

**Image-to-LaTeX Translation of Mathematical Equations**

> Overall goal of the project:

The primary goal of this project is to develop a machine learning model capable of translating images of mathematical equations into their corresponding LaTeX string representations. This system aims to effectively convert visual inputs, whether they are handwritten notes or digitally rendered equations, into valid, editable LaTeX code.

> What framework are you going to use, and you do you intend to include the framework into your project?

The project will be implemented using **PyTorch** as the primary deep learning framework for defining the model architecture and managing the training loop. Additionally, the **torchvision** library and **PIL** (Python Imaging Library) will be integrated to handle image preprocessing tasks, perform necessary data transformations, and manage dataset operations to prepare inputs for the network.

> What data are you going to run on (initially, may change)

The initial dataset is sourced from the **Hugging Face** repository under the name **LaTeX_OCR**. As indicated in the code snippet below, the project loads the "small" configuration of the training split. This downloaded dataset has a total size of approximately $1.49$ GB and contains $268,764$ unique samples. Each sample consists of two key columns:
- *image*: An image file displaying a mathematical equation (with width specified in pixels).
- *text*: A string containing the corresponding ground-truth LaTeX representation of the equation.

```
from datasets import load_dataset
train_dataset = load_dataset("linxy/LaTeX_OCR", name="small", split="train")
```

> What models do you expect to use

The model architecture employs a hybrid CNN-Transformer network designed to sequence visual features into text:
1.  **Encoder:** A ResNet-18 serves as the visual feature extractor. The final two layers are removed, and the resulting output feature map is projected to the model dimension and combined with 2D positional encodings to preserve spatial information.
2.  **Decoder:** A multi-layer Transformer Decoder acts as a causal language model to autoregressively predict the sequence of LaTeX tokens.

*Note: This architecture is a streamlined adaptation of [DGurgurov/im2latex](https://huggingface.co/DGurgurov/im2latex). It simplifies the reference model by replacing the original Swin Transformer and GPT-2 components with vanilla PyTorch implementations.*


## Overall project checklist

The checklist is *exhaustive* which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.
The parenthesis at the end indicates what module the bullet point is related to. Please be honest in your answers, we
will check the repositories and the code to verify your answers.

### Week 1

* [X] Create a git repository (M5)
* [X] Make sure that all team members have write access to the GitHub repository (M5)
* [X] Create a dedicated environment for you project to keep track of your packages (M2)
* [X] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [X] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [X] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [X] Remember to either fill out the `requirements.txt`/`requirements_dev.txt` files or keeping your
    `pyproject.toml`/`uv.lock` up-to-date with whatever dependencies that you are using (M2+M6)
* [X] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [X] Do a bit of code typing and remember to document essential parts of your code (M7)
* [X] Setup version control for your data or part of your data (M8)
* [X] Add command line interfaces and project commands to your code where it makes sense (M9)
* [X] Construct one or multiple docker files for your code (M10)
* [X] Build the docker files locally and make sure they work as intended (M10)
* [X] Write one or multiple configurations files for your experiments (M11)
* [X] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [ ] Use profiling to optimize your code (M12)
* [X] Use logging to log important events in your code (M14)
* [ ] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [ ] Consider running a hyperparameter optimization sweep (M14)
* [ ] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [X] Write unit tests related to the data part of your code (M16)
* [X] Write unit tests related to model construction and or model training (M16)
* [X] Calculate the code coverage (M16)
* [X] Get some continuous integration running on the GitHub repository (M17)
* [X] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [X] Add a linting step to your continuous integration (M17)
* [X] Add pre-commit hooks to your version control setup (M18)
* [X] Add a continues workflow that triggers when data changes (M19)
* [ ] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [X] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [X] Create a trigger workflow for automatically building your docker images (M21)
* [X] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [X] Create a FastAPI application that can do inference using your model (M22)
* [X] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [X] Write API tests for your application and setup continues integration for these (M24)
* [X] Load test your application (M24)
* [ ] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [X] Create a frontend for your API (M26)

### Week 3

* [X] Check how robust your model is towards data drifting (M27)
* [X] Setup collection of input-output data from your deployed application (M27)
* [ ] Deploy to the cloud a drift detection API (M27)
* [X] Instrument your API with a couple of system metrics (M28)
* [ ] Setup cloud monitoring of your instrumented application (M28)
* [ ] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [ ] Write some documentation for your application (M32)
* [ ] Publish the documentation to GitHub Pages (M32)
* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Create an architectural diagram over your MLOps pipeline
* [ ] Make sure all group members have an understanding about all parts of the project
* [ ] Uploaded all your code to GitHub




## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions workflows
│   └── workflows/
│       ├── test.yaml
│       ├── linting.yaml
│       ├── docker_build.yaml
│       ├── data_validation.yaml
│       └── pre-commit-update.yaml
├── configs/                  # Hydra configuration files
│   ├── train.yaml
│   ├── data.yaml
│   ├── preprocess.yaml
│   ├── drift.yaml
│   ├── cloudbuild_api.yaml
│   ├── cloudbuild_frontend.yaml
│   └── cloudbuild_train.yaml
├── data/                     # Data directory
│   ├── raw/
│   │   └── default_train/
│   └── drifted_current/
├── dockerfiles/              # Docker container definitions
│   ├── api.dockerfile
│   ├── data.dockerfile
│   ├── train.dockerfile
│   └── frontend.dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yaml
│   ├── README.md
│   └── source/
│       └── index.md
├── logs/                     # Training logs
├── models/                   # Trained models
│   ├── model.pth
│   ├── model1.pth
│   ├── model2.pth
│   └── vocab.pt
├── notebooks/                # Jupyter notebooks
├── outputs/                  # Timestamped experiment outputs
├── reports/                  # Reports and analysis
│   ├── figures/
│   ├── drift/
│   └── profiling/
├── src/                      # Source code
│   └── ml_ops_project/
│       ├── __init__.py
│       ├── api.py
│       ├── data.py
│       ├── data_drift.py
│       ├── evaluate.py
│       ├── model.py
│       ├── preprocess.py
│       ├── tokenizer.py
│       ├── train.py
│       ├── visualize.py
│       └── py.typed
├── tests/                    # Test suite
│   ├── unittests/
│   │   ├── test_api.py
│   │   ├── test_data.py
│   │   ├── test_model.py
│   │   ├── test_preprocess.py
│   │   ├── test_tokenizer.py
│   │   ├── test_train.py
│   │   └── test_visualize.py
│   ├── integrationtests/
│   │   └── test_apis.py
│   ├── data_qualitytests/
│   │   └── test_data_quality.py
│   └── performancetests/
│       └── locustfile.py
├── AGENTS.md                 # Guidance for autonomous coding agents
├── frontend.py               # Streamlit frontend application
├── LICENSE
├── pyproject.toml            # Python project configuration
├── README.md                 # Project README
├── tasks.py                  # Invoke tasks
├── data.zip.dvc              # DVC data versioning
└── .pre-commit-config.yaml   # Pre-commit hooks configuration
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

## Exam Report

The exam report is located in the `reports/` folder and uses a template provided by the course. The `reports/report.py` script provides utilities to validate and generate your report.

### Running the Report Script

First, ensure the required dependencies are installed:

```bash
uv add typer markdown pydantic loguru
```

Then you can use the report script in two ways:

**Generate an HTML version of your report:**

```bash
uv run python reports/report.py html
```

This will create a `reports/report.html` file that you can view in a browser.

**Check if your answers meet the constraints:**

```bash
uv run python reports/report.py check
```
