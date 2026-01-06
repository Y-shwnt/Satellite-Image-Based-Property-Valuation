# Satellite–Tabular Fusion for Property Price Prediction

This project investigates whether satellite imagery can provide complementary
neighborhood-level information to improve residential property price prediction
when combined with traditional tabular housing data.

The focus of the project is not to replace structured features, but to critically
evaluate the contribution of visual context in a multimodal machine learning
pipeline.

---

## Project Overview

Traditional property valuation models rely on structured attributes such as living
area, number of bedrooms, construction quality, and geographic coordinates. While
these features capture internal property characteristics, they often fail to
represent surrounding environmental and neighborhood context.

In this project, satellite images corresponding to property locations are used to
capture visual information such as green cover, built-up density, road connectivity,
and urban layout. These visual features are combined with tabular data to assess
whether multimodal learning improves price prediction performance.

---

## Modeling Approach

The project follows a staged approach:

1. **Tabular-only baseline**  
   A strong baseline model is trained using structured housing features and an
   XGBoost regressor. The target variable is the log-transformed property price.

2. **Satellite image feature extraction**  
   Satellite images are processed using a pretrained ResNet-18 model used strictly
   as a frozen feature extractor. The CNN is not fine-tuned.

3. **Image-only model**  
   Image embeddings are evaluated independently using XGBoost to assess whether
   satellite imagery contains standalone predictive signal.

4. **Multimodal fusion model**  
   A feature-level fusion strategy is applied by concatenating tabular features
   with PCA-reduced image embeddings and training an XGBoost regressor on the
   combined representation.

---

## Repository Structure

```text
satellite-tabular-property-valuation/
│
├── notebooks/          # End-to-end experiments
│   ├── 1_preprocessing.ipynb
│   ├── 2_tabular_model.ipynb
│   ├── 3_image_model.ipynb
│   ├── 4_fusion_model.ipynb
│   └── 5_generate_predictions.ipynb
│
├── src/                # Reusable source code
│   ├── data_fetcher_train.py
│   └── data_fetcher_test.py
│
├── data/
│   ├── raw/            # Raw datasets (not included)
│   ├── processed/      # Processed datasets (not included)
│   ├── plots/          # Generated plots (optional)
│   └── README.md
│
├── 23119058_report.pdf   # Project report
│   
├── 23119058_final.csv    # Final Predictions
│
├── requirements.txt
└── README.md
```
## Models Implemented

### 1. Tabular-Only Model (Baseline)

This model uses only structured housing attributes such as property size, quality,
and geographic location. An XGBoost regressor is trained on log-transformed property
prices.

- Strong predictive performance  
- Interpretable feature behavior  
- Fast and stable training  

This model serves as the baseline for comparison.

---

### 2. Image-Only Model (Exploratory)

In this setup, satellite images are converted into fixed-length embeddings using a
pretrained ResNet-18 model. The CNN is used strictly as a frozen feature extractor,
and the embeddings are passed to an XGBoost regressor.

- Contains limited predictive signal  
- Performance is substantially weaker than tabular models  
- Sensitive to noise and missing internal property attributes  

This model is exploratory and not competitive on its own.

---

### 3. Multimodal Fusion Model

The fusion model combines tabular features with PCA-reduced satellite image
embeddings using a feature-level fusion strategy. An XGBoost regressor is trained
on the combined representation.

- Improves performance over the tabular baseline  
- Demonstrates complementary value of satellite imagery  
- Requires careful dimensionality reduction and feature selection  

This model represents the final system evaluated in the project.

---

## Key Findings

- Structured tabular features provide the strongest predictive signal for property
  valuation.
- Satellite imagery alone is insufficient for accurate price prediction.
- When integrated carefully, satellite imagery contributes additional
  neighborhood-level information that improves performance.
- Image-derived features such as green cover, built-up density, and road density
  influence predictions in the fusion model.

### Key Takeaway

**Tabular housing attributes remain the dominant drivers of property valuation.**  
Satellite imagery captures meaningful neighborhood-level context and can enhance
predictions when combined appropriately, but naïve fusion strategies may introduce
noise.

---

## Explainability

Model behavior is analyzed using both numerical and visual explainability methods.

- **SHAP** is used to interpret the contribution of tabular and image-derived
  features, including built-up density, green cover, and road density.
- **Grad-CAM** is applied to satellite images to visualize spatial regions that
  influence CNN feature extraction.

The explainability results confirm that the model attends to semantically meaningful
environmental patterns such as vegetation, open spaces, and road connectivity.

---

## How to Set Up the Project

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/satellite-tabular-property-valuation.git
cd satellite-tabular-property-valuation

```
### 2. Create a virtual environment

#### Mac / Linux
```bash
python -m venv venv
source venv/bin/activate
```

#### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set API key (optional)

Satellite images are fetched using an external API. Create a .env file in the
project root:

```
MAPBOX_TOKEN=your_mapbox_api_key_here
```

### 5. Satellite images and trained models are not included in the repository due to size 
      and reproducibility constraints.
