from xgboost import XGBRegressor
import numpy as np
model = XGBRegressor()
model.load_model('xgb_model.json')
new_data = [[
    13.0,       # MedInc
    2.0,      # HouseAge
    54.5,       # AveRooms
    12.0,       # AveBedrms
    1000,      # Population
    3.0,       # AveOccup
    34.2,      # Latitude
    -118.3     # Longitude
]]
predict = model.predict(new_data)
print(f"prediction of new data:{predict}")
predict_price = predict * 100000
print(predict_price)
 #Requirement test 1: model loaded
assert model is not None

# Requirement test 2: correct number of features
assert len(new_data[0]) == 8

# Run inference
prediction = model.predict(new_data)

# Requirement test 3: prediction exists
assert prediction is not None

# Requirement test 4: one input → one prediction
assert len(prediction) == 1

# Requirement test 5: prediction is valid
assert np.isfinite(prediction[0])

print("All requirement tests: PASSED")
print("Prediction:", prediction[0])