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

## References

- Prayle, A., Atkinson, M., & Smyth, A. (2011). *Pediatric Respiratory Medicine*.
- Torres, A., Blasi, F., Peetermans, W. E., Viegi, G., Welte, T., & Reiss, T. F. (2021). *European Respiratory Journal, 57*(6), 1-29.
- Centers for Disease Control and Prevention. (2022). [CDC - Pneumonia](https://www.cdc.gov/pneumonia/index.html).
- Wee, L. E., Lye, D. C., & Lee, V. J. (2023). *The Lancet Respiratory Medicine, 11*(3), 228-229.
- Ramgopal, M., LaPlace, E., & Shah, M. (2022). *Journal of Digital Imaging, 35*(1), 1-13.
- Sharma, A., & Guleria, R. (2023). *European Journal of Radiology, 159*, 109025.
- Kim, S. H., Kim, D. H., & Kim, M. S. (2023). *Journal of Digital Imaging, 36*(1), 1-11.

---
