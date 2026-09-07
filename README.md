# E-Waste AI + Pricing + Recycler Matching — SIH Starter

This project is built around the uploaded datasets.

## What your data contains

### 1. Image dataset (`data/waste_images.zip`)
6,782 labeled images across 17 classes:
Battery, PCB, Mobile, Keyboard, Mouse, Printer, Microwave, Player,
Television, Washing Machine, cardboard, glass, metal, organic, paper,
plastic, trash.

**Important:** the image dataset does NOT currently contain Cable or LCD Panel images.
Do not claim the model can classify Cable/LCD until labeled images for those classes are added.

### 2. Material value dataset
`material_dataset_rupee_fixed.csv` — 6,782 rows with Material, Subcategory,
Condition, Weight and Estimated Value.

### 3. Location price dataset
`price_dataset_india_locations.csv` — 7,650 rows with category, Indian location,
date, price/kg and source.

### 4. Recycler dataset
`recycler_dataset_large.csv` — 5,000 recycler records including accepted materials,
authorization status, rates, pickup availability and service area.

## Recommended SIH pipeline

Photo
  -> image classifier
  -> predicted category + confidence
  -> approximate/entered weight
  -> location price engine
  -> estimated value
  -> recycler matching

The AI classifier should predict the material/category. It should NOT directly invent
the price. Pricing comes from the supplied price dataset.

## Dataset preparation

1. Extract `data/waste_images.zip` into `data/balanced_waste_images/`.
2. Run:
   `python train.py`

The script uses the folder names as labels, creates train/validation/test splits,
trains a lightweight PyTorch CNN, saves the best model and writes metrics.

For stronger accuracy later, replace the scratch CNN with a pretrained
MobileNet/EfficientNet once pretrained weights are available.

## API

After training:
`pip install -r requirements.txt`
`uvicorn api:app --reload`

POST an image to `/predict`.

Example:
{
  "category": "PCB",
  "confidence": 0.91
}

## Pricing

`pricing.py`:
- normalizes classifier labels to pricing labels where there is a safe direct match
- filters prices by selected location
- calculates an indicative min/median/max value using price/kg × weight
- returns "no direct price" when the category is not represented safely

## Recycler matching

`recycler_matching.py`:
- checks whether the predicted category is accepted
- checks authorization status
- checks pickup availability
- gives a simple match score based on material acceptance and service area

## Classes that need more images

For the original proposed demo categories:
- Cable: add labeled Cable images
- LCD Panel: add labeled LCD Panel images

Do not map Television -> LCD Panel or similar automatically unless your team
explicitly defines and validates that business rule.
