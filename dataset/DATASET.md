# Dataset License and Attribution

- **Dataset Name:** Multi-Class Prediction of Obesity Risk
- **Source:** Kaggle
- **URL:** [https://www.kaggle.com/competitions/playground-series-s4e2](https://www.kaggle.com/competitions/playground-series-s4e2)
- **License:** [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

This dataset is not owned by the author of this repository.
It is included and used strictly in accordance with the original license terms.

## Dataset Overview

This dataset is part of the Kaggle Playground Series Season 4, Episode 2, focusing on **Multi-Class Prediction of Obesity Risk**. The goal is to predict obesity levels based on various lifestyle and physical characteristics.

### Dataset Statistics
- **Training samples:** 20,758 records
- **Test samples:** 13,841 records  
- **Features:** 16 input features + 1 target variable
- **Target classes:** 7 obesity categories

### Target Variable
- **Column:** `NObeyesdad` (Obesity Level)
- **Type:** Categorical (Multi-class classification)
- **Classes:**
  - `Insufficient_Weight`
  - `Normal_Weight` 
  - `Overweight_Level_I`
  - `Overweight_Level_II`
  - `Obesity_Type_I`
  - `Obesity_Type_II`
  - `Obesity_Type_III`

### Features Description

#### Demographic Features
- **Gender:** Male/Female
- **Age:** Age in years
- **Height:** Height in meters
- **Weight:** Weight in kg

#### Family History
- **family_history_with_overweight:** Family history of overweight (yes/no)

#### Eating Habits
- **FAVC:** Frequent consumption of high caloric food (yes/no)
- **FCVC:** Frequency of consumption of vegetables (0-3 scale)
- **NCP:** Number of main meals (numerical)
- **CAEC:** Consumption of food between meals (Never, Sometimes, Frequently, Always)
- **CH2O:** Consumption of water daily (0-3 scale)

#### Lifestyle Factors
- **SCC:** Calories consumption monitoring (yes/no)
- **FAF:** Physical activity frequency (0-3 scale)
- **TUE:** Time using technology devices (0-2 scale)
- **CALC:** Consumption of alcohol (Never, Sometimes, Frequently, Always)
- **SMOKE:** Smoking habit (yes/no)
- **MTRANS:** Transportation used (Walking, Bike, Public_Transportation, Automobile)

### File Structure
```
dataset/
├── train.csv           # Training data (20,758 samples)
├── test.csv           # Test data (13,841 samples) - without target labels
├── sample_submission.csv  # Sample submission format
└── DATASET.md         # This documentation file
```

### Usage Notes
- This is a **synthetic dataset** created for educational and competition purposes
- The dataset contains both numerical and categorical features
- Missing values and data preprocessing may be required
- The target variable represents different obesity risk categories based on BMI and lifestyle factors

### Citation
If you use this dataset in your research or projects, please cite the original Kaggle competition:
```
Kaggle Playground Series 2024 Season 4, Episode 2: 
Multi-Class Prediction of Obesity Risk
Walter Reade and Ashley Chow
Available at: https://www.kaggle.com/competitions/playground-series-s4e2
```