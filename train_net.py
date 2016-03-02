import load_data
import vgg16

def train_net():
    net = vgg16.build_model()
    X_train, y_train, X_val, y_val, X_test, y_test = load_data.load_data()

if __name__ == "__main__":  
    train_net()
	
