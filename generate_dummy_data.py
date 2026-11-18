import os
import cv2
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
acne_dir = f'{BASE_DIR}/acne_raw'
celeba_dir = f'{BASE_DIR}/celeba_raw/img_align_celeba'

for lvl in ['Level_0', 'Level_1', 'Level_2']:
    os.makedirs(f'{acne_dir}/{lvl}', exist_ok=True)
os.makedirs(celeba_dir, exist_ok=True)

print("Generating images...")
for i in range(15):
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    cv2.imwrite(f'{acne_dir}/Level_0/img_{i}.jpg', img)
    cv2.imwrite(f'{acne_dir}/Level_1/img_{i}.jpg', img)
    cv2.imwrite(f'{acne_dir}/Level_2/img_{i}.jpg', img)
    
for i in range(30):
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    cv2.imwrite(f'{celeba_dir}/img_{i}.jpg', img)

print("Generating csv data...")
os.makedirs(f'{BASE_DIR}/cosmetics', exist_ok=True)
cosm_data = {
    'Brand': ['BrandA'] * 30,
    'Name': [f'Prod{i}' for i in range(30)],
    'Price': [10] * 30,
    'Rank': [5] * 30,
    'Ingredients': [
        'water, glycerin, niacinamide',
        'water, salicylic acid',
        'dimethicone, fragrance',
        'retinol, water',
        'hyaluronic acid, water',
        'ceramide, glycerin',
        'water, squalane',
        'zinc oxide, titanium dioxide',
        'benzoyl peroxide',
        'water, centella asiatica'
    ] * 3,
    'Combination': [1, 0, 1, 0, 1, 1, 1, 1, 0, 1] * 3,
    'Dry': [1, 0, 0, 0, 1, 1, 1, 1, 0, 1] * 3,
    'Normal': [1, 0, 1, 1, 1, 1, 1, 1, 0, 1] * 3,
    'Oily': [0, 1, 1, 1, 0, 0, 1, 1, 1, 0] * 3,
    'Sensitive': [1, 0, 0, 0, 1, 1, 1, 1, 0, 1] * 3
}
pd.DataFrame(cosm_data).to_csv(f'{BASE_DIR}/cosmetics/cosmetics.csv', index=False)

os.makedirs(f'{BASE_DIR}/sephora', exist_ok=True)
sephora_prod = {
    'product_id': [f'P{i}' for i in range(30)],
    'ingredient': cosm_data['Ingredients']
}
pd.DataFrame(sephora_prod).to_csv(f'{BASE_DIR}/sephora/product_info.csv', index=False)

sephora_rev = {
    'product_id': [f'P{i}' for i in range(30)] * 2,
    'rating': [5, 4, 3, 5, 5, 5, 5, 5, 4, 5] * 6,
    'skin_type': ['normal', 'oily', 'combination', 'oily', 'dry', 'dry', 'normal', 'normal', 'oily', 'sensitive'] * 6
}
pd.DataFrame(sephora_rev).to_csv(f'{BASE_DIR}/sephora/reviews.csv', index=False)
print("Dummy data generation complete!")
