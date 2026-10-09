
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Emergency message dataset
data = {
    "message": [
        # FIRE
        "There is a fire in an apartment building",
        "Smoke is spreading inside the house",
        "Fire broke out in a shopping mall",
        "Kitchen caught fire please send help",
        "A building is burning",
        "There are flames coming from the house",
        "Fire accident reported near the market",
        "Smoke coming from a factory building",
        "My house is on fire",
        "A vehicle caught fire on the road",

        # MEDICAL
        "A person is injured in a road accident",
        "Someone is unconscious and needs medical help",
        "A person is not breathing",
        "Severe bleeding after an accident",
        "An elderly person needs urgent medical assistance",
        "Someone collapsed near the bus stand",
        "A person has suffered a serious injury",
        "Medical help is required immediately",
        "A person is having difficulty breathing",
        "An injured person is lying on the road",

        # FLOOD
        "Flood water is entering our house",
        "Heavy rain has flooded the streets",
        "Water level is rising in the village",
        "Flooding has affected several homes",
        "People are stranded because of flood water",
        "The river has overflowed into nearby houses",
        "Our street is completely flooded",
        "Heavy rainfall caused flooding in the town",
        "Water is entering buildings due to flooding",
        "Families are trapped by rising flood water",

        # ROAD BLOCKAGE
        "A fallen tree is blocking the road",
        "A tree fell across the highway",
        "A road is blocked after a landslide",
        "An accident is blocking traffic",
        "Debris is blocking the main road",
        "A fallen electric pole is blocking the street",
        "The highway is blocked by a fallen tree",
        "Road blockage reported near the school",
        "A large branch is blocking vehicle movement",
        "The road is blocked by fallen rocks",

        # OTHER
        "Fire safety seminar is scheduled for tomorrow",
        "I am learning first aid today",
        "Rain is expected tomorrow",
        "Road construction may cause traffic delays",
        "The weather is pleasant today",
        "A community meeting is planned for Sunday",
        "The school will remain closed tomorrow",
        "Traffic is normal in the city",
        "I want information about disaster preparedness",
        "A safety awareness program is being organized"
    ],

    "category": (
        ["fire"] * 10
        + ["medical"] * 10
        + ["flood"] * 10
        + ["road_blockage"] * 10
        + ["other"] * 10
    )
}

# Convert data into a DataFrame
df = pd.DataFrame(data)

# Create the NLP model
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LinearSVC(class_weight="balanced")
    )
])

# Train the model
model.fit(df["message"], df["category"])

# Save the trained model
joblib.dump(
    model,
    "emergency_model.pkl"
)

print("Emergency NLP model trained successfully!")
print("Total training messages:", len(df))
print("Categories:", sorted(df["category"].unique()))
print("Saved model as emergency_model.pkl")

# Test sample messages
test_messages = [
    "There is a fire in my house",
    "A person is unconscious",
    "Flood water is entering our street",
    "A fallen tree is blocking the highway",
    "The weather is pleasant today"
]

print("\nSample predictions:")

for message in test_messages:
    prediction = model.predict([message])[0]
    print(f"Message: {message}")
    print(f"Predicted category: {prediction}\n")

