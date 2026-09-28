import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Load CSV files
circuits = pd.read_csv('circuits.csv')
results = pd.read_csv('results.csv')
drivers = pd.read_csv('drivers.csv')
pit_stops = pd.read_csv('pit_stops.csv')
races = pd.read_csv('races.csv')
qualifying = pd.read_csv('qualifying.csv')
lap_times = pd.read_csv('lap_times.csv')
constructors = pd.read_csv('constructors.csv')

# Replace '\\N' with NaN
results.replace('\\N', np.nan, inplace=True)
drivers.replace('\\N', np.nan, inplace=True)
pit_stops.replace('\\N', np.nan, inplace=True)
races.replace('\\N', np.nan, inplace=True)
qualifying.replace('\\N', np.nan, inplace=True)
lap_times.replace('\\N', np.nan, inplace=True)

# Calculate age from dob
drivers['dob'] = pd.to_datetime(drivers['dob'])
drivers['age'] = drivers['dob'].apply(lambda x: (datetime.now() - x).days // 365)

# Rename milliseconds columns
pit_stops.rename(columns={'milliseconds': 'milliseconds_pit_stop'}, inplace=True)
lap_times.rename(columns={'milliseconds': 'milliseconds_lap_time'}, inplace=True)

# Merge datasets to create a comprehensive dataset
data = results.merge(drivers, on='driverId') \
              .merge(races, on='raceId') \
              .merge(circuits, on='circuitId') \
              .merge(pit_stops, on=['raceId', 'driverId'], how='left', suffixes=('', '_pit_stop')) \
              .merge(qualifying, on=['raceId', 'driverId'], how='left', suffixes=('', '_qualifying')) \
              .merge(lap_times, on=['raceId', 'driverId'], how='left', suffixes=('', '_lap_time'))

# Select relevant features and target variable
features = ['age', 'nationality', 'constructorId', 'grid', 'positionOrder', 'fastestLapSpeed', 'milliseconds_pit_stop', 'position', 'milliseconds_lap_time']
target = 'win'

# Preprocess data
data['win'] = data['positionOrder'].apply(lambda x: 1 if x == 1 else 0)
data = data.dropna(subset=features + [target])
X = data[features]
y = data[target]

# Convert categorical variables to dummy variables
X = pd.get_dummies(X, columns=['nationality', 'constructorId'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the neural network
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model with fewer epochs
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model
evaluation = model.evaluate(X_test, y_test)

# Plot accuracy and loss
plt.figure(figsize=(12, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Get feature importances
importances = model.layers[0].get_weights()[0]
feature_importances = pd.DataFrame(np.mean(np.abs(importances), axis=1), index=X.columns, columns=['importance']).sort_values('importance', ascending=False)

# Plot top 10 feature importances
top_10_features = feature_importances.head(10)
plt.figure(figsize=(10, 6))
plt.barh(top_10_features.index, top_10_features['importance'])
plt.xlabel('Importancia')
plt.title('Top 10 Variables de mayor importancia')
plt.gca().invert_yaxis()
plt.show()

# Train a decision tree classifier
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)

# Plot the decision tree
plt.figure(figsize=(20, 10))
plot_tree(tree_clf, feature_names=X.columns, class_names=['Not Win', 'Win'], filled=True, rounded=True, fontsize=10)
plt.title('Decision Tree')
plt.show()

# Get the names of the constructors in the top 10 features
top_10_constructor_ids = [col for col in top_10_features.index if 'constructorId_' in col]
top_10_constructors = [col.split('_')[1] for col in top_10_constructor_ids]

# Map constructor IDs to names
constructor_names = constructors.set_index('constructorId')['name'].to_dict()
top_10_constructors_with_names = [(cid, constructor_names[int(cid)]) for cid in top_10_constructors]
top_10_constructors_with_names