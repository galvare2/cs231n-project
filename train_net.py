from __future__ import print_function
import load_data, sys, os, time, theano, vgg16
import lasagne
import numpy as np
import theano.tensor as T
import pickle

REG = 0.002
LAST_FIXED_LAYER = 'pool5'

# Best: REG = 0.002, LAST_FIXED = pool5, DROPOUT = 0.5

# Best for unseen: REG = 0.004, LAST_FIXED = pool5,
# DROPOUT = 0.2 (in vgg16.py)

def iterate_minibatches(inputs, targets, batchsize):
    assert len(inputs) == len(targets)
    for start_idx in range(0, len(inputs) - batchsize+1, batchsize):
        excerpt = slice(start_idx, start_idx+batchsize)
        yield inputs[excerpt], targets[excerpt]

def train_net(num_epochs=10, batch_size=100, learning_rate=1e-4, unseen=False):
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    net = vgg16.build_model(input_var, batch_size)
    network = net['prob']
    # Load the dataset
    if unseen:
        print("Loading data, unseen val/test signatories task...")
        X_train, y_train, X_val, y_val, X_test, y_test = load_data.load_data_unseen_test()
    else:
        print("Loading data, standard task...")
        X_train, y_train, X_val, y_val, X_test, y_test = load_data.load_data()
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    all_params = lasagne.layers.get_all_params(network, trainable=True)
    # Get all the parameters we don't want to train
    fixed_params = lasagne.layers.get_all_params(net[LAST_FIXED_LAYER])
    params = [x for x in all_params if x not in fixed_params]
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var) + REG * lasagne.regularization.apply_penalty(params, lasagne.regularization.l2)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    
    # First get all the parameters
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=learning_rate, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var) + REG * lasagne.regularization.apply_penalty(params, lasagne.regularization.l2) 
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    print("train_fn set up.")

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()

        for batch in iterate_minibatches(X_train, y_train, batch_size):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, batch_size):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, batch_size):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss: withheld until final submission lolol")
    print(" test accuracy: withheld until final submission lolol")

    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)

if __name__ == "__main__":  
    if len(sys.argv)>1 and sys.argv[1]=='unseen':
        train_net(unseen=True)
    else:
        train_net()

