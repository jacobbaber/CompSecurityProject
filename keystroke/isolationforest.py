import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import make_scorer, f1_score


def load_and_preprocess_data(file_path, scaler=None):
    df = pd.read_csv(file_path)

    # Extract relevant columns
    data = df[['Key', 'Delta Time (ms)']]

    # Convert the 'Key' column to numeric values (if it's not already)
    data['Key'] = pd.factorize(data['Key'])[0]

    # Normalize the 'DeltaTime' column
    if scaler is None:
        scaler = StandardScaler()
        data['Delta Time (ms)'] = scaler.fit_transform(data['Delta Time (ms)'].values.reshape(-1, 1))
    else:
        data['Delta Time (ms)'] = scaler.transform(data['Delta Time (ms)'].values.reshape(-1, 1))

    return data, scaler

def evaluate_anomaly_score(model, new_data, scaler, name):
    new_data, _ = load_and_preprocess_data(new_data, scaler)
    
    # Create features for the Isolation Forest
    new_X = new_data.values
    
    # Evaluate the model on the new dataset
    new_data_scores = model.decision_function(new_X)
    
    return new_data_scores, name

    

def main():
    # Get the user's name
    user_name = input("Enter your name: ")

    # Construct file paths using the user's name
    user_file = f'{user_name}_keypress_data.csv'
    datasetA_name = input("Enter the name for dataset A: ")
    datasetB_name = input("Enter the name for dataset B: ")
    datasetA_file = f'{datasetA_name}_keypress_data.csv'
    datasetB_file = f'{datasetB_name}_keypress_data.csv'

    # Load and preprocess user data
    user_data, scaler = load_and_preprocess_data(user_file)
    user_data = user_data.values

    isolation_forest = IsolationForest(n_estimators=500, max_samples=256, contamination=0.07333333333333333)

    # Train the model
    isolation_forest.fit(user_data)


    # Evaluate the model on the new datasets
    datasetA_scores, datasetA_name = evaluate_anomaly_score(isolation_forest, datasetA_file, scaler, datasetA_name)
    datasetB_scores, datasetB_name = evaluate_anomaly_score(isolation_forest, datasetB_file, scaler, datasetB_name)

    # Lower values indicate abnormal behavior, set a threshold to classify sequences as normal or anomalous.
    threshold = 0.02
    datasetA_anomalies = datasetA_scores < threshold
    datasetB_anomalies = datasetB_scores < threshold

    # Print the average dataset scores
    print(f'{datasetA_name} average score: {np.mean(datasetA_scores)}')
    print(f'{datasetB_name} average score: {np.mean(datasetB_scores)}')


    # Print the average anomaly score
    print(f'{datasetA_name} anomaly score: {np.mean(datasetA_anomalies)}')
    print(f'{datasetB_name} anomaly score: {np.mean(datasetB_anomalies)}')

    # Print the percentage of sequences classified as normal
    percentage_normal_datasetA = (1 - np.mean(datasetA_anomalies)) * 100
    percentage_normal_datasetB = (1 - np.mean(datasetB_anomalies)) * 100

    print(f'Percentage of sequences classified as normal for {datasetA_name}: {percentage_normal_datasetA}%')
    print(f'Percentage of sequences classified as normal for {datasetB_name}: {percentage_normal_datasetB}%')

    # Print who is most similar to the user
    if percentage_normal_datasetA > percentage_normal_datasetB:
        print(f'{datasetA_name} is most similar to {user_name}')
    else:
        print(f'{datasetB_name} is most similar to {user_name}')


    # Plot the Dataset A and Dataset B scores on a plane
    datasetA_df, _ = load_and_preprocess_data(datasetA_file, scaler)
    datasetB_df, _ = load_and_preprocess_data(datasetB_file, scaler)

    # Create a 3D figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the scores
    ax.scatter(datasetA_df['Key'], datasetA_df['Delta Time (ms)'], datasetA_scores, label=datasetA_name)
    ax.scatter(datasetB_df['Key'], datasetB_df['Delta Time (ms)'], datasetB_scores, label=datasetB_name)

    # Set the axes labels
    ax.set_xlabel('Key')
    ax.set_ylabel('Delta Time (ms)')
    ax.set_zlabel('Normal Score')

    # Set the legend
    ax.legend()

    # Show the plot
    plt.show()




if __name__ == "__main__":
    main()





### Used to find best parameters for Isolation Forest
"""
    param_sampler = ParameterSampler(param_dist, n_iter=10, random_state=42)

    best_params = None 
    best_score = 0
    for params in param_sampler:
        isolation_forest = IsolationForest(**params)
        isolation_forest.fit(user_data)
        datasetA_scores, datasetA_name = evaluate_anomaly_score(isolation_forest, datasetA_file, scaler, datasetA_name)
        datasetB_scores, datasetB_name = evaluate_anomaly_score(isolation_forest, datasetB_file, scaler, datasetB_name)
        threshold = 0.05
        datasetA_anomalies = datasetA_scores < threshold
        datasetB_anomalies = datasetB_scores < threshold
        percentage_normal_datasetA = (1 - np.mean(datasetA_anomalies)) * 100
        percentage_normal_datasetB = (1 - np.mean(datasetB_anomalies)) * 100
        print(f'Percentage of sequences classified as normal for {datasetA_name}: {percentage_normal_datasetA}%')
        print(f'Percentage of sequences classified as normal for {datasetB_name}: {percentage_normal_datasetB}%')
        print(f'Parameters: {params}')
        print(f'{datasetA_name} anomaly score: {np.mean(datasetA_anomalies)}')
        print(f'{datasetB_name} anomaly score: {np.mean(datasetB_anomalies)}')
        print('-----------------------------------')

        if best_params is None:
            best_params = params
        else:
            if abs(percentage_normal_datasetA - percentage_normal_datasetB > best_score):
                best_params = params
                best_score = abs(percentage_normal_datasetA - percentage_normal_datasetB)
        
    print(f'Best parameters: {best_params}')

"""