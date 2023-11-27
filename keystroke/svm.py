import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# Example data, replace it with your actual data
data = pd.read_csv("Jacob_keypress_data.csv")
bad_data = pd.read_csv("Mikayla_keypress_data.csv")
fred_data = pd.read_csv("Fred_keypress_data.csv")
df = pd.DataFrame(data)
df2 = pd.DataFrame(bad_data)
df3 = pd.DataFrame(fred_data)

# Encode categorical features
label_encoder = LabelEncoder()
df["Key"] = label_encoder.fit_transform(df["Key"])
df2["Key"] = label_encoder.fit_transform(df2["Key"])
df3["Key"] = label_encoder.fit_transform(df3["Key"])


# Feature engineering: Create sequences of 'Key' and 'Delta Time (ms)'
sequence_length = 50 # Adjust as needed
sequences = [df.loc[i:i+sequence_length-1, ['Key', 'Delta Time (ms)']].values.flatten() for i in range(len(df) - sequence_length + 1)]

# Convert sequences to a NumPy array
sequences = np.array(sequences)

# Apply label encoding to the first column of each sequence
sequences[:, 0] = label_encoder.fit_transform(sequences[:, 0])

# Label encoding for 'Key'
label_encoder = LabelEncoder()
sequences[:, 0] = label_encoder.fit_transform(sequences[:, 0])

# Feature scaling
scaler = StandardScaler()
scaled_sequences = [scaler.fit_transform(seq.reshape(-1, 2)).flatten() for seq in sequences]

# Split data into training and testing sets
X_train, X_test = train_test_split(scaled_sequences, test_size=0.2, random_state=42)

# Train the One-Class SVM model
model = OneClassSVM(nu=0.1, kernel='rbf')  # nu is an important parameter, adjust as needed
model.fit(X_train)

# Predict on the test set
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score([1] * len(predictions), predictions)  # Assuming 1 represents the inlier class
print(f'Same User: {accuracy * 100:.2f}%')


# Test on bad data
bad_sequences = [df2.loc[i:i+sequence_length-1, ['Key', 'Delta Time (ms)']].values.flatten() for i in range(len(df2) - sequence_length + 1)]
bad_sequences = np.array(bad_sequences)
bad_sequences[:, 0] = label_encoder.fit_transform(bad_sequences[:, 0])
bad_scaled_sequences = [scaler.fit_transform(seq.reshape(-1, 2)).flatten() for seq in bad_sequences]

# Predict on the bad data
bad_predictions = model.predict(bad_scaled_sequences)

# Evaluate the model
bad_accuracy = accuracy_score([1] * len(bad_predictions), bad_predictions)  # Assuming -1 represents the outlier class
print(f'Different User: {bad_accuracy * 100:.2f}%')

fred_sequences = [df3.loc[i:i+sequence_length-1, ['Key', 'Delta Time (ms)']].values.flatten() for i in range(len(df3) - sequence_length + 1)]
fred_sequences = np.array(fred_sequences)
fred_sequences[:, 0] = label_encoder.fit_transform(fred_sequences[:, 0])
fred_scaled_sequences = [scaler.fit_transform(seq.reshape(-1, 2)).flatten() for seq in fred_sequences]

# Predict on the bad data
fred_predictions = model.predict(fred_scaled_sequences)

# Evaluate the model
fred_accuracy = accuracy_score([1] * len(fred_predictions), fred_predictions)  # Assuming -1 represents the outlier class

print(f'Fred: {fred_accuracy * 100:.2f}%')



# plot the sequences in 1 color and the bad sequences in another color  
import matplotlib.pyplot as plt

# convert each sequence to a single value by summing the values in each row
summed_sequences = [sum(seq) for seq in scaled_sequences]
bad_summed_sequences = [sum(seq) for seq in bad_scaled_sequences]

# plot the sequences in 1 color and the bad sequences in another color
plt.plot(summed_sequences, 'b')
plt.plot(bad_summed_sequences, 'r')
plt.show()





















