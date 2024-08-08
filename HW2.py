import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import cv2
from mlxtend.plotting import plot_decision_regions
# data partition
import random





def train_partition(train_path):
    train_fruits = os.listdir(train_path)
    carambula = []
    lychee = []
    pear = []

    for fruit in train_fruits:
        child = sorted(os.listdir(os.path.join(train_path, fruit)))
        for img in child:
            fruit_img = cv2.imread(os.path.join(train_path, fruit, img), cv2.IMREAD_GRAYSCALE)
            fruit_img = fruit_img.flatten()
            # print(fruit_img.shape)
            # print(type(fruit_img))
            if fruit == "Carambula":
                carambula.append(fruit_img)
            if fruit == "Lychee":
                lychee.append(fruit_img)
            if fruit == "Pear":
                pear.append(fruit_img)

    random.shuffle(carambula)
    random.shuffle(pear)
    random.shuffle(lychee)
    train_caram = carambula[:350]
    valid_caram = carambula[350:]
    train_lych = lychee[0:350]
    valid_lych = lychee[350:]
    train_pear = pear[0:350]
    valid_pear = pear[350:]

    valid_data = valid_caram + valid_lych + valid_pear
    train_data = train_lych + train_caram + train_pear
    train_data_array = np.array(train_data)
    valid_data_array = np.array(valid_data)  

    return train_data_array, valid_data_array


def test_partition(test_path):
    test_fruits = os.listdir(test_path)
    carambula = []
    lychee = []
    pear = []
    for fruit in test_fruits:
        child = os.listdir(os.path.join("Data_test", fruit))
        for img in child:
            fruit_img = cv2.imread(os.path.join("Data_test", fruit, img), cv2.IMREAD_GRAYSCALE)
            fruit_img = fruit_img.flatten()
            if fruit == "Carambula":
                carambula.append(fruit_img)
            if fruit == "Lychee":
                lychee.append(fruit_img)
            if fruit == "Pear":
                pear.append(fruit_img)
    random.seed(50)
    random.shuffle(carambula)
    random.shuffle(pear)
    random.shuffle(lychee)

    test_data = carambula + lychee + pear
    test_data_array = np.array(test_data)
    return test_data_array

def pca(train_data, valid_data, test_data):
    # total_data = np.concatenate((train_data, valid_data, test_data))
    # pca = PCA(n_components=2)
    # pca_total = pca.fit_transform(total_data)
    # pca_train = pca_total[:1050]
    # pca_valid = pca_total[1050:1470]
    # pca_test = pca_total[1470:]

    pca = PCA(n_components=2)
    pca_train = np.load("pca_train.npy")
    pca_valid = np.load("pca_vld.npy")
    pca_test = np.load("pca_test.npy")
    # pca_train = pca.fit_transform(train_data)
    # pca_valid = pca.transform(valid_data)
    # pca_test = pca.transform(test_data)
    return pca_train, pca_valid, pca_test

# Activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sig_deri(x):
    return sigmoid(x) * (1 - sigmoid(x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def CrossEntropyLoss(groundtruth, predictions):
    groundtruth = groundtruth + 1e-5
    return -np.sum(np.multiply(groundtruth, np.log(predictions)))


class TwoLayerNN:
    def __init__(self, neuron=40):
        self.hidden_neuron = neuron
        self.w1 = np.random.random((neuron, 3)) * np.sqrt(1. / 3)
        self.w2 = np.random.random((3, neuron)) * np.sqrt(1. / 3)
        self.z0 = np.zeros((neuron, 1))
        self.z1 = np.zeros((neuron, 1))
        self.bias = np.random.randn()
        self.x1 = np.zeros((3, 1))

    def forward(self, x1): # x1 is a 2*1 vector from pca
        x1 = np.concatenate((x1, np.array([self.bias])))
        x1 = np.expand_dims(x1, axis=1)
        self.x1 = x1
        # print(x1.shape)
        z0 = np.matmul(self.w1 , x1)
        # print(z0.shape)
        self.z0 = z0
        z1 = sigmoid(z0)
        self.z1 = z1
        # print(z0.shape)
    
        z2 = np.matmul(self.w2, z1)
        # print(z1.shape)
        self.z2 = z2
        y = softmax(z2)
    
        return y
    
    def train(self, pca_train, pca_valid, learning_rate=1e-3, epochs=200):
        fig_valid_loss = []
        fig_train_loss = []
        fig_train_acc = []
        fig_valid_acc = []
        for epoch in range(epochs):
            
            train_epoch_loss = 0
            counter = 0
            for cnt in range(len(pca_train)):
                temp = len(pca_train) // 3 
                if cnt < temp:
                    ground_truth = np.array([[1], [0], [0]])
                if temp <= cnt < 2 * temp: 
                    ground_truth = np.array([[0], [1], [0]])
                if cnt >= 2 * temp: 
                    ground_truth = np.array([[0], [0], [1]])

                
                prediction = self.forward(pca_train[cnt])
    
                # print(prediction.argmax())
                if prediction.argmax() == ground_truth.argmax():
                    counter += 1
                    # print(counter)

                dy = prediction - ground_truth
                
                # print("dy_shape = ", dy.shape, dy)

                dw2 = np.matmul(dy, np.transpose(self.z1)) # dw2_shape = 3x20
                
                dz1 = np.matmul(np.transpose(self.w2), dy)
                
                dz0 = np.multiply(dz1, sig_deri(self.z0))

                dw1 = np.matmul(dz0, np.transpose(self.x1))
                
                self.w2 = self.w2 - dw2 * learning_rate
                self.w1 = self.w1 - dw1 * learning_rate
                train_loss = CrossEntropyLoss(ground_truth, prediction)
                train_epoch_loss += train_loss
                train_acc = counter / len(pca_train)
            train_epoch_loss /= len(pca_train)
            fig_train_loss.append(train_epoch_loss)
            fig_train_acc.append(train_acc)
            # print(f"{epoch}, train_epoch_loss = {train_epoch_loss}  train_accuracy = {train_acc} ")

            counter = 0
            valid_epoch_loss = 0
            for cnt in range(len(pca_valid)):
                temp = len(pca_valid) // 3 
                if cnt < temp:
                    ground_truth = np.array([[1], [0], [0]])
                if temp <= cnt < 2 * temp: 
                    ground_truth = np.array([[0], [1], [0]])
                if cnt >= 2 * temp: 
                    ground_truth = np.array([[0], [0], [1]])

                prediction = self.forward(pca_valid[cnt])
    
                # print(prediction.argmax(), ground_truth.argmax())
                if prediction.argmax() == ground_truth.argmax():
                    counter += 1
    
                
                valid_loss = CrossEntropyLoss(ground_truth, prediction)
                valid_epoch_loss += valid_loss
            valid_acc = counter / len(pca_valid)
            valid_epoch_loss /= len(pca_valid)
            # print(f"{epoch} valid_epoch_loss = {valid_epoch_loss} vld_accuracy = {valid_acc}")

            
            fig_valid_loss.append(valid_epoch_loss)
            fig_valid_acc.append(valid_acc)
        
        plt.plot(range(epochs), fig_train_acc, label="train acc")
        plt.plot(range(epochs), fig_valid_acc, label="validation acc")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.title("two-layer network accuracy curve")
        plt.legend()
        plt.savefig("2layer_acc_curve")


        plt.figure()
        plt.plot(range(epochs), fig_train_loss, label="train loss")
        plt.plot(range(epochs), fig_valid_loss, label="validation loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("two-layer network loss curve")
        plt.legend()
        plt.savefig("2layer_loss_curve")

    def test(self, pca_test):
        counter = 0
        for cnt in range(len(pca_test)):
            temp = len(pca_test) // 3 
            if cnt < temp:
                ground_truth = np.array([[1], [0], [0]])
            if temp <= cnt < 2 * temp: 
                ground_truth = np.array([[0], [1], [0]])
            if cnt >= 2 * temp: 
                ground_truth = np.array([[0], [0], [1]])
            prediction = self.forward(pca_test[cnt])

            # print(prediction.argmax())
            if prediction.argmax() == ground_truth.argmax():
                counter += 1
        print(f"test_accuracy = {counter / len(pca_test)} counter = {counter} len = {len(pca_test)}")
class ThreeLayerNN:
    def __init__(self, neuron=20):
        self.hidden_neuron = neuron
        self.w1 = np.random.random((neuron, 3)) * np.sqrt(1. / 3)
        self.w2 = np.random.random((neuron, neuron)) * np.sqrt(1. / 3)
        self.w3 = np.random.random((3, neuron)) * np.sqrt(1. / 3)
        self.z0 = np.zeros((neuron, 1))
        self.z1 = np.zeros((neuron, 1))
        self.z2 = np.zeros((neuron, 1))
        self.z3 = np.zeros((neuron, 1))
        self.z4 = np.zeros((3, 1))
        self.bias = np.random.randn()
        self.x1 = np.zeros((3, 1))

    def forward(self, x1): # x1 is a 2*1 vector from pca
        x1 = np.concatenate((x1, np.array([self.bias])))
        x1 = np.expand_dims(x1, axis=1)
        self.x1 = x1
        # print(x1.shape)
        z0 = np.matmul(self.w1 , x1)
        # print(z0.shape)
        self.z0 = z0
        z1 = sigmoid(z0)
        self.z1 = z1
        # print(z0.shape)
        # print(z1.shape)
        z2 = np.matmul(self.w2, z1)
        self.z2 = z2

        z3 = sigmoid(z2)
        self.z3 = z3
        z4 = np.matmul(self.w3, z3)
        y = softmax(z4)
        return y
    
    def train(self, pca_train, pca_valid,learning_rate=0.0015, epochs=200):
        fig_valid_loss_b = []
        fig_train_loss_b = []
        fig_train_acc_b = []
        fig_valid_acc_b = []
        for epoch in range(epochs):
            train_epoch_loss = 0
            counter = 0
            for cnt in range(len(pca_train)):
                temp = len(pca_train) // 3 
                if cnt < temp:
                    ground_truth = np.array([[1], [0], [0]])
                if temp <= cnt < 2 * temp: 
                    ground_truth = np.array([[0], [1], [0]])
                if cnt >= 2 * temp: 
                    ground_truth = np.array([[0], [0], [1]])

                
                prediction = self.forward(pca_train[cnt])
    
                # print(prediction.argmax())
                if prediction.argmax() == ground_truth.argmax():
                    counter += 1
                    # print(counter)

                dy = prediction - ground_truth
                
                # print("dy_shape = ", dy.shape, dy)

                dw3 = np.matmul(dy, np.transpose(self.z3)) # dw2_shape = 3x20
                dz3 = np.matmul(np.transpose(self.w3), dy)
                dz2 = np.multiply(dz3, sig_deri(self.z2))
                dw2 = np.matmul(dz2, np.transpose(self.z1))
                dz1 = np.matmul(np.transpose(self.w2),dz2)
                dz0 = np.multiply(dz1, sig_deri(self.z0))
                dw1 = np.matmul(dz0, np.transpose(self.x1))
                
                self.w3 = self.w3 - dw3 * learning_rate
                self.w2 = self.w2 - dw2 * learning_rate
                self.w1 = self.w1 - dw1 * learning_rate
                train_loss = CrossEntropyLoss(ground_truth, prediction)
                train_epoch_loss += train_loss
            train_acc = counter / len(pca_train)
            train_epoch_loss /= len(pca_train)


            fig_train_loss_b.append(train_epoch_loss)
            fig_train_acc_b.append(train_acc)
            print(f"{epoch}, train_epoch_loss = {train_epoch_loss}  train_accuracy = {train_acc} ")


            counter = 0
            valid_epoch_loss = 0
            for cnt in range(len(pca_valid)):
                temp = len(pca_valid) // 3 
                if cnt < temp:
                    ground_truth = np.array([[1], [0], [0]])
                if temp <= cnt < 2 * temp: 
                    ground_truth = np.array([[0], [1], [0]])
                if cnt >= 2 * temp: 
                    ground_truth = np.array([[0], [0], [1]])

                prediction = self.forward(pca_valid[cnt])
    
                # print(prediction.argmax(), ground_truth.argmax())
                if prediction.argmax() == ground_truth.argmax():
                    counter += 1
    
                
                valid_loss = CrossEntropyLoss(ground_truth, prediction)
                valid_epoch_loss += valid_loss
            valid_acc = counter / len(pca_valid)
            valid_epoch_loss /= len(pca_valid)
            print(f"{epoch}, valid_epoch_loss = {valid_epoch_loss}  valid_accuracy = {valid_acc} ")


            
            fig_valid_loss_b.append(valid_epoch_loss)
            fig_valid_acc_b.append(valid_acc)
        plt.figure()
        plt.plot(range(epochs), fig_train_acc_b, label="train acc")
        plt.plot(range(epochs), fig_valid_acc_b, label="validation acc")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.title("three-layer network accuracy curve")
        plt.legend()
        plt.savefig("3layer_acc_curve")


        plt.figure()
        plt.plot(range(epochs), fig_train_loss_b, label="train loss")
        plt.plot(range(epochs), fig_valid_loss_b, label="validation loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("three-layer network loss curve")
        plt.legend()
        plt.savefig("3layer_loss_curve")

    def test(self, pca_test):
        counter = 0
        for cnt in range(len(pca_test)):
            temp = len(pca_test) // 3 
            if cnt < temp:
                ground_truth = np.array([[1], [0], [0]])
            if temp <= cnt < 2 * temp: 
                ground_truth = np.array([[0], [1], [0]])
            if cnt >= 2 * temp: 
                ground_truth = np.array([[0], [0], [1]])
            prediction = self.forward(pca_test[cnt])

            # print(prediction.argmax())
            if prediction.argmax() == ground_truth.argmax():
                counter += 1
        print(f"test_accuracy = {counter / len(pca_test)}")

class Classifier():

    def __init__(self, network:TwoLayerNN):
        self.network = network
    def predict(self, data):
        prediction = []
        
        for i in data:
            prediction.append(np.argmax(self.network.forward(i)))
        return np.array(prediction)

def answer(network):
    cls = Classifier(network)
    y = np.zeros(1050)
    for i in range(1050):
        if i < 350:
            y[i] = 0
        if 350 <= i < 700:
            y[i] = 1
        if 700 <= i:
            y[i] = 2    

    plt.figure()
    plot_decision_regions(X=pca_train, y=y.astype(int), clf=cls, legend=2)
    plt.title("decision region of 2layer vanilla")
    plt.savefig("dereof2.png")
    plt.show()

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def reduce_dimensions(train_images, test_images, num_components=2):
    flattened_train = train_images.reshape(train_images.shape[0], -1)
    flattened_test = test_images.reshape(test_images.shape[0], -1)

    transformed_train = pca._fittransform(flattened_train)
    transformed_test = pca.transform(flattened_test)

    # normalize each part separately
    transformed_train_parts = []
    transformed_test_parts = []
    for i in range(num_components):
        part_train_normalized = normalize(transformed_train[:, i])
        part_test_normalized = normalize(transformed_test[:, i])
        transformed_train_parts.append(part_train_normalized)
        transformed_test_parts.append(part_test_normalized)
    transformed_train = np.column_stack(transformed_train_parts)
    transformed_test = np.column_stack(transformed_test_parts)

    return transformed_train, transformed_test

train_x, train_y = load_image(train_path)
test_x, test_y = load_image(test_path)

train_x, test_x = reduce_dimensions(train_x, test_x)


if __name__ == '__main__':
    train_data, valid_data = train_partition("/home/hao/homework/HW2/Data_train")
    test_data = test_partition("/home/hao/homework/HW2/Data_test")
    pca_train, pca_valid, pca_test = pca(train_data, valid_data, test_data)

    two_nn = TwoLayerNN(90)
    two_nn.train(pca_train, pca_valid)
    two_nn.test(pca_test)
    # three_nn = ThreeLayerNN(15)
    # three_nn.train(pca_train, pca_valid)
    # three_nn.test(pca_test)
    answer(two_nn)
    answer(two_nn)

