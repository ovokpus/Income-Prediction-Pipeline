# Income-Prediction-Pipeline

---

## Online Prediction Machine Learning System Designed and Deployed and Maintained with MLOps Practices

### Business Problem

The problem at hand revolves around determining the income levels of individuals registered in a past census exercise. This was originally released by the US Census bureau.

The predictions from this model will be useful in determining how to allocate resources for urban development and civil infrastructure projects. The insights gleaned from this model prediction will also aid the government in determining what kind of economic programs to initiate and implement, and better decide on how to improve the economic lot of citizens and residents.

---

### Data Source and Preparation

The dataset can be accessed from [this link](https://archive.ics.uci.edu/ml/datasets/census+income). Certain transformations are necessary to unzip the data, convert it to csv format and append columns. A robust pipeline can be built to facilitate feeding in the data on a monthly basis, which is not covered in this PoC MLOps project.

---

### The MLOps Stack

Below is a representation of the various tools and technologies that make up the MLOps stack used in this project:

1. Experiment Tracking with `MLFlow`, served from a Google Cloud Platform (GCP) Virtual Machine.
2. Exploratory Data Analysis, research and experimentation with `Jupyter Notebooks` in a `Jupyter Lab` environment.
3. Version control with `git` and `github`
4. Training Pipeline Orchestration and Scheduling with `prefect`, deployed as a `prefect orion server` in a GCP VM. Prefect Storage set in a GCP storage bucket.
5. Model and artifact registration using the `MLFlow model registry`, and the default artifact storage located in a Google Cloud Storage Bucket.
6. Model served from `MLflow` registry, using the `flask` framework.
7. Model monitoring using `Grafana`, `Prometheus` and `Evidently AI`
8. Stream data simulated and sent to the prediction service by running the script in `./stream-generator/send_data.py`
9. MLOps Engineering best practices implemented, including unit tests, linting and formatting using `pylint` and `black`, as well as `git pre-commit hooks`.
10. Docker images for prediction and monitoring tagged and pushed to docker hub for easy redeployment in the future.

![image](https://user-images.githubusercontent.com/64817005/189513833-3e36e0a6-3737-4a33-9231-25e43f36395d.png)

---

### Setup and Run system

#### Run experimantation and training pipeline

1. Provision 2 Virtual Machines and 2 storage buckets on Google Cloud platform.  and the other is for prefect storage.

One VM is for development, experimentation and the MLflow server.

```bash
# project-vm
ssh -i ~/.ssh/id_rsa ovokpus@35.247.121.140
```

Clone repository into the VM and set up the project environment by running the following in the project root directory:

```bash
sudo apt update

sudo apt upgrade

sudo apt install make

make setup

make build
```

One bucket is for storing the artifacts and models in MLflow

```bash
# start mlflow server
mlflow server --host 10.138.0.5 --backend-store-uri=sqlite:///mlflow.db --default-artifact-root=gs://project-mlflow-bucket/

# configure connection with prefect server (after configuring prefect server in the other VM)
prefect config set PREFECT_API_URL="http://35.247.100.48:4200/api"

# start jupyter lab
jupyter lab --ip 0.0.0.0 --port 8888 --no-browser
```

The other VM is for serving the prefect orion server

```bash
# prefect-vm
ssh -i ~/.ssh/id_rsa ovokpus@35.247.100.48

# configure prefect storage
prefect storage create # follow the steps to configure the other storage bucket

# configure prefect server API
prefect config set PREFECT_ORION_UI_API_URL="http://<external-ip>:4200/api"

prefect orion start --host 0.0.0.0
```

In the project VPC, ensure external IP addresses are created for the Virtual machines and enable the following ports to allow you connect with various services:

9696 for prediction service
27017 for mongodb
3000 for evidently service
8085 for grafana
4200 for prefect
5000 for mlflow
8888 for jupyter.

Connect locally with these services using the respective ports

```bash
# Connect locally to jupyter lab
http://35.247.121.140:8888/lab

# mlflow
http://35.247.121.140:5000/

# prefect orion
http://35.247.100.48:4200/

# Grafana Evidently dashboard
http://35.247.100.48:8085/
```

---

#### Exploratory Data Analysis

Here, the data was analyzed and labels were found to have class imbalance. SMOTE was applied to upsample the minority class and XGBoost Classifier was also determined to be the optimal algorithm with which the model was built. This was done in comparison with Logistic Regression, and Gradient Boosting Classifier and Random Forest Classifier.

---

#### Experiment Tracking and Model Registry

Best model can be obtained either from the Mlflow UI, the prefect training output and the run id stored as an environment variable

```bash
mlflow artifacts download \
    --run-id ${MODEL_RUN_ID} \
    --artifact-path models_mlflow \
    --dst-path ./models
```

XGBoost Pipeline experiment run with hyperoptimization
![image](https://user-images.githubusercontent.com/64817005/189514018-3faa361d-f30d-45b5-b371-e92050e45aba.png)

Sample artifact logged with mlflow default storage shown as the google cloud storage bucket
![image](https://user-images.githubusercontent.com/64817005/189514029-95af99b5-aa60-4275-bfa8-5f987494a208.png)
![image](https://user-images.githubusercontent.com/64817005/189514040-af70aaf7-29da-48d8-989d-ab860283bc7f.png)

Registered models
![image](https://user-images.githubusercontent.com/64817005/189514044-0285f7ea-0e43-4827-9978-decda5ddb974.png)

---

#### Training Pipeline Orchestration

Start Pipeline Orchestration by creating a training agent on orion and running the following in the projecyt VM:

```bash
prefect agent start 7f3e5fba-334a-414d-83cd-9495cda6f3fd
```

Prefect Orion Dashboard showing training deployment task runs and scheduling
![image](https://user-images.githubusercontent.com/64817005/189514049-7108f39b-705f-4476-9021-61e8f1c81608.png)

---

#### Prediction Service and Monitoring

Prediction Service can be started up by running the command

```bash
docker-compose build
```

Evidently Data Drift Monitoring
![image](https://user-images.githubusercontent.com/64817005/189514061-464f6b69-d39b-45b1-9d4f-901b8ddeae23.png)

Feature Data Drift Score
![image](https://user-images.githubusercontent.com/64817005/189514064-1846f663-f98b-4fa6-afb4-542277988927.png)

Evidently Categorical Target Drift Monitoring
![image](https://user-images.githubusercontent.com/64817005/189514072-bf3229f1-a8f6-4c09-8e4f-83085c013169.png)

Alert rule creation for getting notifications when the drift figures cross set thresholds.
![image](https://user-images.githubusercontent.com/64817005/189514093-956741f9-db16-4428-bf7a-c9a5a9accd00.png)

---

### Concluding remarks

This project is still ongoing at time of submission for peer evaluation. Technical debt include

- Test coverage for integration testing.
- CI/CD pipeline with git workflows.

---
