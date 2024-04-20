import socket
import datetime
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import os
import xgboost as xgb
import joblib
from sklearn.metrics import classification_report

SERVER = socket.gethostbyname(socket.gethostname())
PORT = 5050
ADDR = (SERVER, PORT)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(ADDR)

def send(msg):
    message = msg.encode()
    client.send(message)

def authenticate():
    username = input("Enter your username: ")
    password = input("Enter your password: ")
    credentials = f"{username},{password}"
    send(credentials)

    response = client.recv(1024).decode()
    print(response)
    if response.startswith("Authenticated:"):
        # Extract machine number from the response
        machine_number = response.split(":")[1].strip()
        return machine_number
    else:
        return None

def main():
    machine_number = authenticate()
    if machine_number:
        # Data processing and model training
        df = pd.read_csv('C:\\Users\\wahid\\PycharmProjects\\GenoStagePredictor\\breast-cancer.csv')
        y = LabelEncoder().fit_transform(df['pam50_+_claudin-low_subtype'])
        X = StandardScaler().fit_transform(df[df.columns[31:520]].dropna())
        X = pd.DataFrame(X, columns=df.columns[31:520])

        def isolation_forest(X, y):
            df = pd.concat([X, y], axis=1)
            iso_forest_with_target = IsolationForest(contamination=0.2)
            iso_forest_with_target.fit(df)
            df['outlier_with_target'] = iso_forest_with_target.predict(df)
            df_cleaned_with_target = df[df['outlier_with_target'] != -1].drop('outlier_with_target', axis=1)
            X_out = df_cleaned_with_target.drop(['label'], axis=1)
            y_out = df_cleaned_with_target['label']
            return X_out, y_out

        def smote(X, y):
            smote = SMOTE(k_neighbors=1, random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            return X_resampled, y_resampled

        def pca(X, dim=50):
            pca_ap = PCA(n_components=dim)
            X_pca = pca_ap.fit_transform(X)
            return X_pca, pca_ap

        X_iso, y_iso = isolation_forest(X, pd.DataFrame(y, columns=['label']))
        X_smote, y_smote = smote(X_iso, y_iso)
        X_pca, pca_ap = pca(X_smote)
        X_train, X_test, y_train, y_test = train_test_split(X_pca, y_smote, test_size=0.15, stratify=y_smote)

        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(7, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=50, validation_split=0.2)

        model.evaluate(X_test, y_test)
        y_prob = model.predict(X_test)
        y_pred = tf.argmax(y_prob, axis=1)
        print(classification_report(y_test, y_pred))

        local_xgb_model = xgb.XGBClassifier()

        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        local_models_folder = 'local_models'
        os.makedirs(local_models_folder, exist_ok=True)
        local_model_filename = os.path.join(local_models_folder, f'local_machine_{machine_number}_{timestamp}.pkl')
        joblib.dump(local_xgb_model, local_model_filename)

        # Inform server about disconnection request
        send("Disconnect")

        # Inform server to remove machine number from active machine list
        send("RemoveMachine")

        # Close the client
        client.close()
        exit()
    else:
        send("Disconnect")
        client.close()
        exit()

if __name__ == "__main__":
    main()
