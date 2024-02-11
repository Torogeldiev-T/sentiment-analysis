## Project Structure:

```
Final-Project
├── data                     #Will be loaded                
│   ├── train.csv
│   └── test.csv
|
├── data_process             # Scripts used for data loading
│   ├── data_generation.py   
│   └── __init__.py      
|     
├── inference                # Scripts and Dockerfiles used for inference
│   ├── Dockerfile
│   ├── run.py               
│   └── __init__.py
|
├── models                    # Folder where trained models are stored
│   └── best_model.pickle
|
├── training                  # Scripts and Dockerfiles used for training
│   ├── Dockerfile
│   ├── train.py
│   └── __init__.py
|
|-- results                   # Will be generated and store all the results
|   |-- accuracy_results.txt 
|   |-- inference_log.txt
|   |-- predictions.csv
|
|-- notebook
|   |-- Bakyt_Yrysov,_Final_Project.ipynb   #Notebook that contains the training 
|
├── utils.py                  # Utility functions and classes that are used in scripts
|
└── README.md
```

## Overview
The training notebook is stores in "notebook" folder 

The steps below show how to create and run the docker image:

## Training:

- Build the training Docker image:
```bash
 docker build -f ./training/Dockerfile -t training_image .
```
 - Run the training Docker image: 
```bash
 docker run -it training_image /bin/bash
```

- Then, move the trained model from the directory inside the Docker container `/app/models` to the local machine using:
```bash
docker cp <container_id>:/app/models/model.pickle ./models
```
Replace `<container_id>` with your running Docker container ID


- Copy training logs directory to local machine: 
```bash
docker cp <container_id>:/app/results/ ./results
```

Replace `<container_id>` with your running Docker container ID

## Inference:

- Build the inference Docker image:
```bash
 docker build -f ./inference/Dockerfile -t inference_image .
 ```

 - Run the inference Docker container:
```bash
 docker run -it inference_image /bin/bash 
 ```

- Copy results directory to local machine: 
```bash
docker cp <container_id>:/app/results/ ./results
```
Replace `<container_id>` with your running Docker container ID

## Conclusion 

The best model(Logistic Regression) predictions and accuracy on inference are stored in the results folder

## Thank you!