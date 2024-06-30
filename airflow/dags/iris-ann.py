from datetime import datetime
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pickle
from airflow.decorators import dag, task
from airflow.providers.postgres.hooks.postgres import PostgresHook


class IrisAnn:
    def __init__(self, hidden_weights=None, hidden_bias=None, output_weights=None, output_bias=None):
        self.hidden_weights = hidden_weights
        self.hidden_bias = hidden_bias
        self.output_weights = output_weights
        self.output_bias = output_bias

    def load_data(self):
        iris = datasets.load_iris()
        return iris

    def normalize_data(self, X):
        return (X - X.mean(axis=0)) / X.std(axis=0)

    def one_hot_encode(self, y):
        num_classes = len(np.unique(y))
        return np.eye(num_classes)[y]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def split_data(self, X, y, test_size=0.2, random_state=42):
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def train(self, X_train, y_train, learning_rate=0.1, epochs=1000):
        input_neurons = X_train.shape[1]
        hidden_neurons = 8
        output_neurons = y_train.shape[1]

        self.hidden_weights = np.random.rand(input_neurons, hidden_neurons)
        self.hidden_bias = np.random.rand(1, hidden_neurons)

        self.output_weights = np.random.rand(hidden_neurons, output_neurons)
        self.output_bias = np.random.rand(1, output_neurons)

        for epoch in range(epochs):
            hidden_layer_activation = np.dot(X_train, self.hidden_weights) + self.hidden_bias
            hidden_layer_output = self.sigmoid(hidden_layer_activation)

            output_layer_activation = np.dot(hidden_layer_output, self.output_weights) + self.output_bias
            output = self.sigmoid(output_layer_activation)

            output_error = y_train - output
            output_delta = output_error * self.sigmoid_derivative(output)

            hidden_error = output_delta.dot(self.output_weights.T)
            hidden_delta = hidden_error * self.sigmoid_derivative(hidden_layer_output)

            self.output_weights += hidden_layer_output.T.dot(output_delta) * learning_rate
            self.output_bias += np.sum(output_delta, axis=0, keepdims=True) * learning_rate

            self.hidden_weights += X_train.T.dot(hidden_delta) * learning_rate
            self.hidden_bias += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    def export_models(self, output_path):
        with open(output_path, 'wb') as f:
            pickle.dump(self, f)

    def test_accuracy(self, X_test, y_test):
        hidden_layer_activation = np.dot(X_test, self.hidden_weights) + self.hidden_bias
        hidden_layer_output = self.sigmoid(hidden_layer_activation)

        output_layer_activation = np.dot(hidden_layer_output, self.output_weights) + self.output_bias
        test_output = self.sigmoid(output_layer_activation)

        predicted_labels = np.argmax(test_output, axis=1)
        true_labels = np.argmax(y_test, axis=1)
        accuracy = np.mean(predicted_labels == true_labels)
        return accuracy

    def predict(self, X_test):
        hidden_layer_activation = np.dot(X_test, self.hidden_weights) + self.hidden_bias
        hidden_layer_output = self.sigmoid(hidden_layer_activation)

        output_layer_activation = np.dot(hidden_layer_output, self.output_weights) + self.output_bias
        output = self.sigmoid(output_layer_activation)
        result = np.argmax(output, axis=1)
        return result


default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 6, 1),
}


@dag(
    'TrainingAndPredictionPipeline',
    default_args=default_args,
    description='Train and Predict Iris dataset using ANN',
    schedule_interval='0 10 * * *',  # This schedules the DAG to run at 10 AM every day
)
def run_pipeline():
    @task()
    def train_model():
        print('start training')
        iris_ann = IrisAnn()
        iris = iris_ann.load_data()
        X = iris.data
        X = iris_ann.normalize_data(X)
        y = iris.target
        y = iris_ann.one_hot_encode(y)
        X_train, X_test, y_train, y_test = iris_ann.split_data(X, y, test_size=0.2, random_state=42)
        iris_ann.train(X_train, y_train, learning_rate=0.1, epochs=1000)
        export_path = 'dags/models/iris_ann.pkl'
        iris_ann.export_models(export_path)

    @task()
    def get_data():
        print('start getting data')
        conn = PostgresHook(postgres_conn_id='local_postgres').get_conn()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM input_data')
        data = cursor.fetchall()
        return data

    @task()
    def predict(data):
        print('start prediction')
        with open('dags/models/iris_ann.pkl', 'rb') as f:
            iris_ann = pickle.load(f)
        index = [x[0] for x in data]
        payload = [x[1:] for x in data]
        X = np.array(payload)
        X = iris_ann.normalize_data(X)
        result = iris_ann.predict(X)
        conn = PostgresHook(postgres_conn_id='local_postgres').get_conn()
        cursor = conn.cursor()
        cursor.execute('CREATE TABLE IF NOT EXISTS predictions (id SERIAL PRIMARY KEY, prediction INTEGER)')
        for idx, data in zip(index, result):
            cursor.execute(f'INSERT INTO output_data (executed_at, id, class) VALUES (now(), {idx}, {data})')
        conn.commit()
        conn.close()

    train_model()
    data = get_data()
    predict(data)


run_pipeline()
