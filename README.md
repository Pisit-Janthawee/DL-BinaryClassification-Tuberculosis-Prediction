# DL-BinaryClassification-Tuberculosis-Prediction
- Updated Version on Kaggle: https://www.kaggle.com/code/pisitjanthawee13/tuberculosis-classification-cnns-with-lime

<img src="https://github.com/Pisit-Janthawee/DL-BinaryClassification-Tuberculosis-Prediction/blob/main/images/gradio.png" align="center">

Binary classification | Convolutional Neural Network (CNN) | TensorFlow | Medical X-Ray Imaging | Image Classification | LIME | Local Interpretable Model-Agnostic Explanations (LIME) | Image Processing | Image Segmentation | Data Augmentation | MLOps | Machine learning Engineer | Model interpretability | Model Explainability | Class imbalance | 

- **CRISP Data Mining Methodology**:

  - **Overview**: This project focuses on Developed a Tuberculosis Prediction model using Chest X-ray Images (Binary classification) Implemented MLOps scripts for operational efficiency, transformed images to Numpy, applied data augmentation, and addressed class imbalance. Conducted statistical analysis, preprocessed data for a TensorFlow-based Convolutional Neural Network (CNN), and ensured model interpretability using LIME, This allows us to explain the model's predictions and visualize the regions or areas in the X-rays that contribute most to the disease detection. Deployed the model as a web application via Gradio for accessible Tuberculosis predictions.

  - **Objective**: To create a Convolutional Neural Network (CNN) model for Tuberculosis prediction to detect the chest X-ray image input.

  - **Business Type**: Healthcare and Medical

  - **Business Objective**:
    Enable accurate and automated detection of Tuberculosis from chest X-ray images. Facilitate early diagnosis and intervention for Tuberculosis cases, contributing to improved patient outcomes.

  - **Learning Problems**: Deep Learning for Binary Classification

  - **Reason for Choosing this Project**: Addressing a crucial healthcare challenge by leveraging advanced Deep Learning techniques for Tuberculosis prediction.

  - **Expected Result**: Probability of disease

  - **Utilization of Results**: Serve model prediction as API and Integration of the Deep Learning model into healthcare systems for automated Tuberculosis screening from chest X-ray images.

  - **Benefits of this Project**:
    - Improved my first deep learning study by applying the concepts from the Machine Learning Specialization by Andrew Ng, while simultaneously leveraging my object-oriented programming (OOP) skills to enhance MLOps practices for the project.
    - Automated and efficient Tuberculosis detection from chest X-ray images.
    - Early diagnosis leads to timely medical interventions and improved patient outcomes.
    - Utilization of cutting-edge Deep Learning technology in healthcare for predictive analytics.
    Contribution to medical research by exploring the potential of Deep Learning in Tuberculosis diagnosis.
    - Enhanced capabilities for healthcare professionals in screening and managing Tuberculosis cases.
  
# Tuberculosis

## What is Tuberculosis?

Tuberculosis (TB) is an infectious disease caused by the bacterium Mycobacterium tuberculosis. It primarily affects the lungs but can also target other parts of the body, such as the kidneys, spine, and brain. TB is a contagious disease that spreads through the air when an infected individual coughs, sneezes, or speaks.

## Symptoms

The symptoms of tuberculosis can vary depending on the stage and type of infection. Common symptoms include persistent cough (often with blood-containing sputum), chest pain, fatigue, weight loss, night sweats, fever, and chills. In some cases, TB may be asymptomatic or present with mild symptoms, making it harder to diagnose.

## Causes

Tuberculosis is caused by the bacterium Mycobacterium tuberculosis. It is primarily transmitted through the inhalation of airborne droplets containing the bacteria from an infected person. Factors that increase the risk of developing tuberculosis include weakened immune system (such as HIV/AIDS), close contact with an infected individual, malnutrition, smoking, and living in crowded or unsanitary conditions.

## Diagnosis

The diagnosis of tuberculosis involves various tests, including:

1. Tuberculin skin test (TST): This test involves injecting a small amount of purified protein derivative (PPD) tuberculin into the forearm and evaluating the skin reaction after 48-72 hours.
2. Interferon-gamma release assays (IGRAs): Blood tests that detect the release of interferon-gamma by white blood cells in response to Mycobacterium tuberculosis antigens.
3. Chest X-ray: X-ray imaging of the chest to detect any abnormalities in the lungs, such as the presence of lesions or cavities.
4. Sputum culture: A sample of sputum (mucus coughed up from the lungs) is collected and cultured in a laboratory to identify the presence of Mycobacterium tuberculosis.

## Treatment

Tuberculosis is treated with a combination of antibiotics to ensure the effective elimination of the bacteria and prevent the development of drug resistance. The standard treatment regimen for active tuberculosis involves an initial phase of intensive treatment with multiple antibiotics, followed by a continuation phase with fewer medications. The duration of treatment can range from 6 to 9 months or longer, depending on the severity of the infection and the specific treatment protocol.

## Types of Tuberculosis

From ICD-10 (International Classification of Diseases 10th Revision)

| Diagcode | DiagEngName                        | DiagThName          | Description                                                                                                                                                                                                                             |
| -------- | ---------------------------------- | ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A15      | Respiratory tuberculosis           | วัณโรคทางเดินหายใจ  | Tuberculosis primarily affecting the respiratory system, particularly the lungs. It is the most common form of tuberculosis.                                                                                                            |
| A16      | Tuberculosis of other organs       | วัณโรคของอวัยวะอื่น | Tuberculosis affecting organs other than the respiratory system, such as the kidneys, spine, brain, and lymph nodes. This category includes extrapulmonary tuberculosis.                                                                |
| A17      | Tuberculosis of the nervous system | วัณโรคของระบบประสาท | Tuberculosis specifically affecting the nervous system, including the brain and spinal cord.                                                                                                                                            |
| A18      | Tuberculosis of other organs       | วัณโรคของอวัยวะอื่น | Tuberculosis affecting organs other than the respiratory system and nervous system. This category includes tuberculosis of bones and joints, genitourinary tuberculosis, and other forms of extrapulmonary tuberculosis.                |
| A19      | Miliary tuberculosis               | วัณโรคเมลียะ        | A severe form of tuberculosis in which the bacteria spread throughout the body, leading to widespread infection in multiple organs. Miliary tuberculosis is characterized by the appearance of tiny millet seed-like lesions on X-rays. |
| B90      | Sequelae of tuberculosis           | ผลึกของวัณโรค       | Long-term complications or sequelae resulting from previous tuberculosis infection, such as scar tissue formation in the lungs or other affected organs.                                                                                |

# Problem Statement

Imagine you are embarking on a mission to develop an advanced deep-learning system for the detection of tuberculosis (TB) using chest X-ray images.

With the rising global health concerns surrounding tuberculosis, it's crucial to create a tool that can accurately identify TB cases from these X-ray images.
You are fortunate to have access to a substantial dataset of chest X-ray images, which can be utilized to train and validate your deep learning convolutional neural network (CNN) model.

# Tuberculosis Detection using Chest X-ray with Deep Learning

## Dataset

The dataset consists of:

- Amount: 700 Tuberculosis images / 3500 normal images.
- File type .PNG
- Image size: 512x512
- Channel: 3 channel (RGB )

This dataset is obtained from Kaggle (696 MB): https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset

## Project's purpose and scope

### **Purpose**

The primary purpose of this project is to leverage deep learning techniques, specifically Convolutional Neural Networks (CNNs), for the detection of Tuberculosis (TB) using chest X-ray images. This project serves multiple educational and practical objectives:

1. **Knowledge Improvement**: Gain a deeper understanding of CNNs, their architecture, and their application in medical image analysis.
2. **Dataset Exploration**: Familiarize myself with handling medical image datasets, including preprocessing, data augmentation, and data visualization.
3. **Code Implementation**: Develop practical coding skills by implementing a CNN model for binary classification of TB and normal chest X-ray images.
4. **Script Development**: Create a structured and efficient script to facilitate machine learning operations, from data preparation to model evaluation.
5. **Hyperparameter Tuning**: Experiment with various hyperparameters to optimize the model's performance, learning the impact of these hyperparameters on training outcomes.
6. **Data Visualization**: Learn how to visualize X-ray images and model performance metrics in a way that is easy to comprehend and interpret.
7. **Binary Classification Task**: Understand the nuances of handling binary classification tasks, dealing with class imbalance, and interpreting model predictions.
8. **Unstructured Data Handling**: Gain experience in working with unstructured medical data and transforming it into meaningful insights through deep learning.

### **Scope**

aims to facilitate a holistic understanding of tuberculosis detection using deep learning while equipping the project with the tools and methodologies necessary for meaningful analysis and model interpretation

## Introduction

Tuberculosis (TB) is a highly contagious infectious disease that primarily affects the lungs. Early and accurate detection of TB is crucial for effective treatment and control of the disease. Chest X-rays are commonly used in TB diagnosis due to their ability to reveal abnormalities in the lungs.

This project aims to develop a deep learning model for the binary classification of TB and normal chest X-ray images. The dataset consists of 700 TB images and 3500 normal images, ensuring a balanced representation of both classes. The goal is to train a model that can accurately classify new chest X-ray images as either TB-positive or TB-negative.

## Algorithm Recommendation

For Tuberculosis Detection using Chest X-ray images with a binary classification task (TB vs. normal), Convolutional Neural Networks (CNNs) are commonly used and have shown promising results in medical image analysis. You can consider using pre-existing CNN architectures, such as

- VGGNet
- ResNet
- InceptionNet
  as a starting point for your Tuberculosis Detection project. These architectures have demonstrated strong performance on image classification tasks and can be fine-tuned to suit your specific needs.

## Machine Learning Steps

1. Data Preprocessing
   - 1.1 Import Dataset
   - 1.2 Data Transforming
     - 1.2.1 Tabulating: Converting Image.PNG Files to NumPy Arrays
     - 1.2.2 Color Conversion
     - 1.2.3 Resizing the Images
   - 1.3 Data Augmentation (Optional) + Clinical Perspective
   - 1.4 Data Preparation
   - 1.5 Image Enhancement
     - 1.5.1 Contrast Stretching
     - 1.5.2 Histogram Equalization
     - 1.5.3 Adaptive Histogram Equalization
   - 1.6 Normalization
     - 1.6.1 Min-Max Normalization
     - 1.6.2 Standardization for PCA
   - 1.7 Principal Component Analysis (PCA)
     - 1.7.1 Finding Optimum Number of Principal Components
     - 1.7.2 Image Reconstruction
   - 1.8 Splitting the Dataset
2. CNN
   - 2.1 How CNNs Work
   - 2.2 Layers
     - 2.2.1 Convolutional Layers
     - 2.2.2 Pooling Layers
     - 2.2.3 Dropout Layer
     - 2.2.4 Flatten Layer
     - 2.2.5 Fully Connected (Dense) Layers
     - 2.2.6 Output Layers
   - 2.3 Hyperparameters
     - 2.3.1 Padding
     - 2.3.2 Kernel Size
     - 2.3.3 Stride
   - 2.4 Activation Functions
     - 2.4.1 ReLU Activation Function
     - 2.4.2 Sigmoid Activation Function
   - 2.5 Tensorflow and Keras
   - 2.6 Architecture
   - 2.7 Adam Gradient Descent Optimization
3. Experimentation

   - 3.1 Importing Utility
   - 3.2 Loading the Dataset
   - 3.3 Splitting the Dataset
   - 3.4 Generating Real-world/Unseen Data
   - 3.5 Experimentation
     - Experiment 1: Building the First CNN Model
     - Experiment 2: Comparing Optimizers
     - Experiment 3: Comparing Pooling Techniques
     - Experiment 4: Tuning Hyperparameters - Padding, Kernel Size, and Stride
     - Experiment 5: Hyperparameter Optimization
       - Experiment 5.1: Learning Rate ($\lambda$) for Adam
       - Experiment 5.2: Epsilon ($\epsilon$) for Adam
       - Experiment 5.3: Batch Size
       - Experiment 5.4: Number of Epochs
     - Experiment 6: Image Enhancement Techniques
     - Experiment 7: Impact of Varying Unit Sizes in Neural Network Layers
     - Experiment 8: Exploring Different Image Input Sizes
   - 3.6 Summary of Overall Experiment

4. Explainability
   - 4.1 Why Explainability Matters
   - 4.2 Explainer
     - 4.2.1 Lime Library
       - Image Segmentation
       - Model Explainability   
     - 4.2.2 Shap Explainability Library / No more Explainer

# File Description

## Folder

1. **artifacts**
   - _Explanation_: This folder may contain saved model weights, configurations, or any artifacts resulting from training and experimentation. It's where you store the final trained model for use in deployment.
2. **images**

   - _Explanation_: This folder can store image files used for explanatory purposes in Jupyter notebooks or any other documentation.

3. **repositorys**

   - _Explanation_: This folder is used to store various datasets, metadata, and other data-related resources. It contains subfolders such as CSV files, Excel files, NumPy arrays, references, and experiment-related data.

4. **scripts**

   - _Explanation_: This folder contains utility scripts that facilitate machine learning operations and support CNN experiments. Each script file may contain classes and functions to simplify various tasks related to data preprocessing, model training, evaluation, and explainability.

5. **notebooks**
   - _Explanation_: This folder can house Jupyter notebooks where you conduct exploratory data analysis (EDA), experiment with different machine learning models and hyperparameters, and explain model results.

## 01-05 .ipynb Files

1. **01_init_notebook.ipynb**
   - _Explanation_: This initial notebook is used for exploring the data and performing data preprocessing tasks as outlined in the "Data Preprocessing" section. It serves as a starting point for the project.
2. **02_experiment.ipynb**
   - _Explanation_: This notebook is dedicated to experimentation. It explores various hyperparameters of the CNN model as mentioned in the "Experimentation" section. Different configurations are tested to find the best-performing model.
3. **03_modeling.ipynb**
   - _Explanation_: In this notebook, the CNN model is constructed and tuned based on the findings from the experimentation notebook. The best model is saved in the "artifacts" folder.
4. **04_explainability.ipynb**
   - _Explanation_: Model explainability is crucial for understanding what the model sees and its decision-making process. This notebook explores explainability techniques and libraries.
5. **05_deployment.ipynb**
   - _Explanation_: This notebook focuses on model deployment using Gradio. It provides an easy-to-use interface for displaying input images and model predictions as probabilities.

## **Incorporating Machine Learning Specialization Knowledge**

This project builds upon the knowledge gained from the Machine Learning Specialization program by AI visionary Andrew Ng. The concepts covered in the program, including Supervised Machine Learning (Regression and Classification), Advanced Learning Algorithms, Unsupervised Learning, Recommenders, and Reinforcement Learning, will be applied to the Tuberculosis Detection project. The aim is to bridge theoretical understanding with practical implementation, leveraging the expertise acquired from the specialization program.

The **Machine Learning Specialization** is a foundational online program created in collaboration between Stanford Online and DeepLearning.AI
**Here** if interested: https://online.stanford.edu/courses/soe-ymls-machine-learning-specialization

## **References**

To enrich my project's insights and methodologies, consider referencing various sources, including **Kaggle**, **Google**, **articles** from **[frontiersin.org](http://frontiersin.org/)**, and other relevant academic and practical resources. These sources can provide valuable guidance on topics such as data augmentation, image enhancement, clinical perspectives, and performance evaluation in medical image analysis.
