## ANN, universal function approximator
It is known given sufficient capacity (i.e., number of neurons and layers, activation
functions),  NN models (even with one hidden layer) are universal function 
approximator;  that is they can approximate every continuous functions.

This Python program provides functionality to define fully connected ANN
to approximate a 1D function (sin is default). User can provide other functions
as well. 

Although more advanced ANN (e.g. RNN) can be used to better approximate the
function using the linear or nonlinear correlation in the sequence, here the
standard ANN is used. The input is just the coordinate points.

Scalar, distribution and histogram summaries of the models weight and 
graph are generated for visualization in Tensorboard.

## Usage

- specify log directory for tensorboard data: e.g. `LOGDIR = "./graph"`
- Specify function in `GenTrainTestData(num_samples, rand_fact)`
- Follow `main()` for defining model architecture, performing training and test
- main take a string argument to create a sub-folder for log files
related to that run `main(RunNum)`, it's useful for Tensorboard visualization.
- User inputs the RunNum

```python
    # Main Function
    # Data
    num_samples = 4000
    X_train, Y_train, X_test, Y_test = GenTrainTestData(num_samples, 0.01)

    # writer
    writer = tf.summary.FileWriter(os.path.join(LOGDIR, RunNum))

    # Interpolator, last node must be of size one
    SinInterp = Interpolator(nSizeInput=1, HiddenLayersSizeList=[10, 10, 10, 1], writer=writer)
    SinInterp.Fit(X_train, Y_train, epoch=6000, batch_sz=num_samples)
    Yh = SinInterp.Predict(X_test)
``` 

## Author
Mehdi Paak

## License
MIT