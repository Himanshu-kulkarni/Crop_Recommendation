import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# --------------------------
# 1. Load Datasets
# --------------------------
crop_df = pd.read_csv("Crop_recommendation.csv")
soil_df = pd.read_csv("soil_data.csv")

# --------------------------
# 2. Train Crop Prediction Model
# --------------------------
X = crop_df[['N','P','K','temperature','humidity','ph','rainfall']]
y = crop_df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Optional: check accuracy
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Save model for later use
joblib.dump(model, "crop_model.pkl")

# --------------------------
# 3. Soil Feature Lookup
# --------------------------
def get_soil_features(soil_type):
    matched = soil_df[soil_df['soil_type'].str.lower() == soil_type.lower()]
    if matched.empty:
        print("Soil type not found! Using default: Black")
        matched = soil_df[soil_df['soil_type'].str.lower() == "black"]
    soil_row = matched.iloc[0]
    return soil_row['N'], soil_row['P'], soil_row['K'], soil_row['pH']

# --------------------------
# 4. Offline Average Weather Data
# --------------------------
average_weather = {
    "Pune": {"temperature": 28, "humidity": 60, "rainfall": 50},
    "Mumbai": {"temperature": 30, "humidity": 70, "rainfall": 100},
    "Delhi": {"temperature": 35, "humidity": 50, "rainfall": 20},
    "Kolkata": {"temperature": 32, "humidity": 75, "rainfall": 150},
    "Bangalore": {"temperature": 27, "humidity": 65, "rainfall": 40},
    "Chennai": {"temperature": 33, "humidity": 70, "rainfall": 60},
}

def get_weather(city):
    city_name = city.title()
    if city_name in average_weather:
        w = average_weather[city_name]
        return w['temperature'], w['humidity'], w['rainfall']
    else:
        print(f"No data for {city}. Using default values.")
        return 25, 60, 0  # default

# --------------------------
# 5. Crop Recommendation
# --------------------------
def recommend_crop(soil_type, city):
    N, P, K, ph = get_soil_features(soil_type)
    temperature, humidity, rainfall = get_weather(city)
    input_df = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                            columns=['N','P','K','temperature','humidity','ph','rainfall'])
    crop = model.predict(input_df)[0]
    return crop

# --------------------------
# 6. Interactive Farmer Input
# --------------------------
print("=== AI Crop Recommendation System (Offline Mode) ===")
soil_type = input("Enter soil type (Black, Red, Alluvial, Laterite, Loamy, Clay, Sandy): ").strip()
city = input("Enter your city (Pune, Mumbai, Delhi, Kolkata, Bangalore, Chennai): ").strip()

recommended_crop = recommend_crop(soil_type, city)
print(f"\nRecommended Crop for {soil_type} soil in {city}: {recommended_crop}")
