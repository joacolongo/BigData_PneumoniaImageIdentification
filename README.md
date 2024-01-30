# Pneumonia Image Identification: Big Data Engineering Final Project

## Overview

This repository contains the final project for the Big Data Engineering course as part of the MSc in Computational Biology program at UPM. The project focuses on the identification of pneumonia in chest radiographs using machine learning and deep learning techniques.

## Authors

- Joaquin Algorta Bove
- Laura Masa Martínez
- Miguel La Iglesia Mirones
- Lucía Muñoz Gil


*Master of Science in Computational Biology, Big Data Engineering*  
*January 2024*

---

## Introduction

Pneumonia is an infectious disease primarily affecting the lungs, caused by various infectious agents including bacteria, viruses or even fungi (Prayle, Atkinson, & Smyth, 2011). It leads to inflammation in the small air sacs (alveoli) and the surrounding tissues, causing breathing difficulties and severe symptoms as they fill with fluid or pus (Torres et al., 2021). This repository addresses the identification of pneumonia in chest radiographs using machine learning and deep learning techniques.

## Background

Pneumonia is a significant public health concern worldwide due to its impact on morbidity and mortality, several factors contribute to the risk of developing pneumonia (Torres et al., 2021; Centers for Disease Control and Prevention, 2022):

- Age: Young children and elderly individuals are at higher risk due to immature or weakened immune systems.
- Immunological status: Immunocompromised individuals, such as those with HIV/AIDS or chronic diseases, are more susceptible.
- Smoking: Tobacco smoking damages the lungs and weakens the immune system, increasing susceptibility to respiratory infections.
- Environmental exposure: Exposure to pollutants like smoke, toxic chemicals, or air pollution can elevate the risk.
- Underlying medical conditions: Chronic diseases like heart disease, chronic lung conditions, neurological disorders, or malnutrition increase vulnerability.

Pneumonia can affect different parts of the respiratory system, and its complications may include respiratory failure and acute respiratory distress syndrome, often requiring oxygen therapy and mechanical ventilation (Torres et al., 2021). Common symptoms of pneumonia include cough, fever, shortness of breath, fatigue and chest pain. Diagnosis typically involves a chest X-ray and, in some cases, other tests such as respiratory or blood cultures. Treatment usually involves the use of antibiotics and, in severe cases, may require hospitalization and respiratory support. 


Prevention and control measures are crucial to addressing this ongoing public health challenge, including vaccination, respiratory hygiene promotion, and early identification and treatment of cases (Wee, Lye, & Lee, 2023).  Prevention of pneumonia includes vaccination, practicing good hygiene, and maintaining a strong immune system (Torres et al., 2021). Pneumonia diagnosis involves chest X-rays and other tests, with treatment typically including antibiotics and sometimes hospitalization.  Rapid, efficient diagnosis can significantly benefit the patient's prognosis, enabling quick initiation of treatment and implementation of necessary measures (Wee, Lye, & Lee, 2023). 

The use of machine learning and deep learning models for predicting pneumonia from chest radiographs is a promising approach in medical imaging analysis. Machine learning algorithms, including logistic regression, decision trees, SVM, and random forests, analyze features extracted from chest radiographs to predict pneumonia (Ramgopal et al., 2022). Deep learning models like CNNs, such as DenseNet, ResNet, and VGG, automatically learn hierarchical features directly from raw images, achieving high accuracy and sensitivity in pneumonia detection (Sharma & Guleria, 2023; Kim et al., 2023). Proper data preprocessing, training, and validation techniques, including cross-validation and data augmentation, enhance model performance. Model evaluation is done using metrics like accuracy, sensitivity, specificity, and AUC-ROC, with high sensitivity being crucial for accurate pneumonia diagnosis.


## Project Goals

The project aims to develop machine learning and deep learning models for predicting pneumonia from chest radiographs, focusing on big data namagement. These models would analyze features extracted from a high number of chest radiographs to detect pneumonia.

## Approach

1. **Data Management and Preprocessing**: Proper data preprocessing techniques will be applied to manage high volume of data.
2. **Model Development and Training**: ML-DL models will be trained and evaluated.
3. **Evaluation Metrics**: Models will be evaluated using metrics like accuracy.
4. **Performance Enhancement**: Techniques such as cross-validation and data augmentation will be used to enhance model performance.

## Material and methods

### Data collection and preparation:

The dataset was obtained from Kaggle, specifically from the repository provided by Paul Mooney. The Kaggle API was utilized in Google Colab to seamlessly integrate the dataset into the environment. The downloaded dataset was stored in Google Drive for easy access and management. The Kaggle API authentication allowed direct download of the dataset to Google Colab. The dataset, named "chest-xray-pneumonia," was retrieved using the Kaggle API's dataset_download_files function. Upon download, the dataset was extracted using the unzip command and organized in Google Drive.

### Exploratory Data Analysis

In our exploratory data analysis (EDA), we utilized Python along with several libraries to gain insights into the dataset. We employed Pandas for data manipulation and analysis, Matplotlib for creating visualizations such as bar plots and histograms, and Seaborn to enhance the aesthetics of our plots. Additionally, we leveraged PySpark for distributed data processing and analysis, enabling efficient handling of large-scale datasets.

Throughout our EDA process, we undertook several key steps. Firstly, we partitioned the dataset into training, testing, and validation sets, providing a structured framework for subsequent analysis. Using Matplotlib, we generated bar plots to visualize the distribution of images across the normal and pneumonia classes for each dataset, facilitating a clear understanding of class imbalances and dataset compositions.

Furthermore, we conducted in-depth analysis by creating separate datasets for images belonging to normal individuals and patients with pneumonia. This involved data manipulation tasks executed with PySpark, ensuring scalability and efficiency in handling the dataset's large volume.

To gain insights into the physical characteristics of the images, we calculated pixel counts for the height and width dimensions in both categories. PySpark's distributed computation capabilities enabled us to perform these calculations efficiently, even on large-scale datasets. We visualized the distribution of these pixel counts through histograms, providing valuable insights into the variability and characteristics of images within each class.

A comprehensive dataset was constructed using the create_image_dataframe function, which facilitated the extraction of crucial information including image names, file paths, dataset categorizations (train, test, or validation), and image labels denoting normal or pneumonia conditions. This initial dataset served as the foundation for subsequent analysis and model development.

In order to assess the physical characteristics of the images, the calculate_pixel_counts function was employed to ascertain the number of pixels present in both the height and width dimensions of each image. This step provided essential insights into the dimensions and resolutions of the images, aiding in further preprocessing and analysis.
Subsequently, the process_data function was utilized to carry out image processing tasks. This involved adjusting the resolution of the images, identifying lung regions within the images, and removing a predetermined number of pixels from the borders to refine the dataset for subsequent analysis.

With the individual image processing steps completed, the compose_dataset function was employed to compile the dataset. This involved applying the previously executed image processing techniques, transforming the pixel matrices into vectors, and assembling these flattened vectors into a Spark DataFrame. The resulting DataFrame contained the processed images in the form of flattened vectors (stored in the 'images' column) along with their corresponding labels denoting the class (stored in the 'class' column).

For effective visualization and interpretation of the dataset composition, various visualizations were generated using Matplotlib. Bar plots were utilized to visualize the distribution of images across different datasets, providing insights into dataset composition and balance. Additionally, histograms were employed to depict the pixel counts for both height and width dimensions in normal and pneumonia images, aiding in understanding the variability and characteristics of the images within each class.

### Preprocessing

The preprocessing phase began with normalization, where resizing parameters were established to ensure uniform image dimensions across the dataset. This involved setting the image resolution to (100, 100, 1) and defining a border of 30 pixels. The datasets were then partitioned into train, test, and validation sets to facilitate systematic analysis. Each set underwent preprocessing using the compose_dataset function, which involved image processing techniques to refine and enhance the dataset.

Following preprocessing, we visually inspected the validation set to assess image quality. This involved examining each image alongside its corresponding label, indicating whether it fell under the 'Normal' or 'Pneumonia' class.

To prepare the data for modeling, we conducted several steps to ensure proper dataset formatting. We evaluated the dimensions of the preprocessed test set to understand its structure and size. Additionally, we extracted distinct values from the "class" column of the test set to analyze class distribution.

Furthermore, we computed the length of the first vector in the preprocessed train set to verify data consistency. We also developed a user-defined function, numpy_tovector_udf, to convert numpy arrays into Spark Vector types, enabling seamless integration into the Spark framework.

Finally, we transformed the train, test, and validation sets into DataFrames with features and label columns using the VectorAssembler functionality. This transformation organized the datasets for machine learning modeling by creating separate columns for features and labels.

### Build the model

We utilized the Keras framework to construct a neural network model, specifying its architecture to include an input layer, two hidden layers, and an output layer. The input layer was designed with a shape of 10000 elements, representing the number of features in the dataset. The model incorporated two dense hidden layers, each comprising 12 and 8 neurons, respectively, and utilized the rectified linear unit (ReLU) activation function. The output layer comprised two neurons, devoid of any activation function, suitable for binary classification tasks.

Following the model's definition, we compiled it using the Adam optimizer and the Cross Entropy Criterion as the loss function. The optimizer method was set to Adam, and the batch size was configured as 64 instances per batch. Additionally, we specified the maximum number of epochs for training the model to be 50.

To encapsulate the neural network model and its configurations effectively, we instantiated an NNClassifier object. This encapsulation allowed us to apply optimizer method, batch size, and maximum epoch settings to the NNClassifier instance, streamlining the training process of the neural network classifier.

### Model training

Before training the model, we conducted a preprocessing step to randomize the order of the training dataset, aiming to reduce potential biases that could arise from grouping all images with one label together. Randomizing the dataset order helps ensure that the model learns effectively across all classes and minimizes the risk of biases during training. This involved adding a new column called 'rand_col' to the training dataset and populating it with random values. Subsequently, the DataFrame was ordered by the 'rand_col' column to achieve a random order of instances, after which the 'rand_col' column was dropped to maintain the original structure.
Once the dataset was randomized, we trained the neural network model using the NNClassifier instance, which had been configured with desired settings for optimization method, batch size, and maximum epoch. The fit method of the classifier was then applied to the randomized training DataFrame, resulting in the training of the neural network model on the randomized dataset.

### Predict and evaluation of the results

Following model training, we evaluated its performance using an external set of images. This involved predicting classifications for these unseen instances and assessing the accuracy of the predictions, which serves as the standard metric for classification tasks.

First, the model predicted classifications for the test set using the trained neural network model, storing the predictions in a DataFrame named predictionDF, which was then cached for efficiency. A sample of the prediction DataFrame was displayed for inspection.

Next, we evaluated the model's performance using various metrics, including accuracy, precision, recall, and F1 score. These metrics were calculated using the MulticlassClassificationEvaluator, comparing predicted labels with true labels in the test set. The results were visualized using a bar plot to facilitate comparison across different metrics. Additionally, the accuracy of the model was reiterated for clarity.

Finally, we created a confusion matrix to visualize the model's performance in more detail, illustrating true positive, false positive, true negative, and false negative predictions. The matrix was plotted as a heatmap, with true labels represented on the y-axis and predicted labels on the x-axis, providing insights into the model's classification performance.

## Results and discussion

The classification accuracy of our model stands at 78%, indicating a reasonable performance. However, it's important to note that this accuracy depends on various factors, and further evaluation is warranted. In addition to accuracy, other evaluation metrics such as precision, recall, and F1-score can provide a more comprehensive understanding of the model's performance, especially in the case of imbalanced classes, which is evident in our dataset.

Calculating precision and recall yields valuable insights into the model's performance in classifying both pneumonia and normal images. Precision, which indicates the percentage of positive predictions that were correct, is particularly crucial in medical diagnoses, where minimizing false positives is vital. Our model achieved a precision of 82.50%, indicating its ability to accurately identify positive instances. Conversely, recall, which measures the ability to capture all positive instances, is essential in detecting diseases where identifying all positive cases is critical. Our model exhibited a recall rate that complements its precision, resulting in an F1-score of 75.92%, suggesting a reasonable balance between precision and recall.

While these metrics indicate a reasonable performance overall, further development and refinement of the model are warranted to achieve a more efficient and precise classification model for pneumonia identification. The confusion matrix provides valuable insights into the model's performance across different classes, highlighting areas where the model may be overfitted or biased due to the nature of the training data.

Analysis of the confusion matrix reveals that our model excels in predicting patients with pneumonia, with an efficiency of 98.72%. This high accuracy in diagnosing pneumonia patients is crucial for ensuring timely and efficient treatment. However, the model exhibits limitations in identifying healthy individuals, potentially leading to unnecessary treatment administration and hospitalization. This discrepancy may stem from the imbalanced classes in the training set, where pneumonia images were overrepresented. Future studies should explore techniques to create a more balanced training set to enhance the model's accuracy and address these limitations.

## Conclusion

In conclusion, our model demonstrates high efficiency in identifying patients with pneumonia, which aligns with the primary objective of the project. However, its limitations in identifying healthy individuals underscore the need for further development and refinement. Addressing these challenges through balanced training data and advanced modeling techniques will be crucial in enhancing the model's accuracy and applicability in clinical settings.


## References

- Prayle, A., Atkinson, M., & Smyth, A. (2011). Pneumonia in the developed world. *Pediatric Respiratory Medicine, 12*(1):60-9.
- Torres, A., Cilloniz, C., Niederman, M. S., et al. (2021). Pneumonia. *Nat Rev Dis Primers, 7*, 25.
- Centers for Disease Control and Prevention. (2022). Risk Factors for Pneumonia. Retrieved from [https://www.cdc.gov/pneumonia/riskfactors.html](https://www.cdc.gov/pneumonia/riskfactors.html). Obtained on January 27, 2024.
- Wee, L. E., Lye, D. C., & Lee, V. J. (2023). Developments in pneumonia and priorities for research. *The Lancet Respiratory Medicine, 11*(12), 1046-1047.
- Ramgopal, S., Ambroggio, L., Lorenz, D., Shah, SS., Ruddy, RM., & Florin, TA. (2022). A Prediction Model for Pediatric Radiographic Pneumonia. *Pediatrics, 149*(1), e2021051405.
- Sharma, A., & Guleria, R. (2023). A Deep Learning based model for the Detection of Pneumonia from Chest X-Ray Images using VGG-16 and Neural Networks. *Procedia Computer Science, 218*, 357-366.
- Kim, C., Hwang, E. J., Choi, Y. R., Choi, H., Goo, J. M., Kim, Y., Choi, J., & Park, C. M. (2023). A Deep Learning Model Using Chest Radiographs for Prediction of 30-Day Mortality in Patients With Community-Acquired Pneumonia: Development and External Validation. *AJR Am J Roentgenol, 221*(5), 586-598.


---
