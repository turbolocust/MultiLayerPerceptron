/*
 * The MIT License
 * 
 * Copyright 2018 Matthias Fussenegger, Johannes Stadelmann
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy 
 * of this software and associated documentation files (the "Software"), to deal 
 * in the Software without restriction, including without limitation the rights 
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
 * copies of the Software, and to permit persons to whom the Software is 
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in 
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
 * THE SOFTWARE.
 */
package at.fhv.machinelearning;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;

/**
 * Neural network (MLP) implementation supporting three layers (one hidden).
 *
 * @author Matthias Fussenegger
 * @author Johannes Stadelmann
 */
public class NN implements Serializable {

    private static final long serialVersionUID = 2892539756486253209L;

    private final String _id; // network identifier
    private List<Layer> _network;
    private Double _learningRate = 0.3;

    public NN(String id) {
        _id = id;
    }

    /**
     * Factory method to create a new instance of {@link NN}.
     *
     * @param id the network identifier.
     * @param numInputs number of input neurons.
     * @param numsHiddenLayers numbers of neurons for each hidden layer. Each
     * list element stands for a hidden layer.
     * @param numOutputs number of output neurons.
     * @return a new instance of {@link NN}.
     */
    public static NN create(String id, int numInputs,
            List<Integer> numsHiddenLayers, int numOutputs) {
        NN nn = new NN(id);
        nn.init(numInputs, numsHiddenLayers, numOutputs);
        return nn;
    }

    /**
     * Initializes the neural network.
     *
     * @param numInputs number of input neurons.
     * @param numsHiddenLayers numbers of neurons for each hidden layer. Each
     * list element stands for a hidden layer.
     * @param numOutputs number of output neurons.
     */
    public void init(int numInputs, List<Integer> numsHiddenLayers, int numOutputs) {
        _network = new ArrayList<>();
        int temp = numInputs;
        for (Integer i : numsHiddenLayers) {
            _network.add(initLayer(temp, i));
            temp = i;
        }
        _network.add(initLayer(temp, numOutputs));
    }

    private Layer initLayer(int numWeights, int numNeurons) {
        Layer layer = new Layer(numNeurons);
        final Random rand = new Random();
        for (int i = 0; i < numNeurons; ++i) {
            double[] values = new double[numWeights];
            for (int j = 0; j < values.length; ++j) {
                values[j] = rand.nextGaussian(); // random numbers with a mean of 0 and a standard deviation of 1
            }
            layer.getNeurons().add(new Neuron(new ArrayRealVector(values)));
        }
        return layer;
    }

    /**
     * Transfers neuron activation.
     *
     * @param activation the activation value.
     * @return the transferred value.
     */
    private double transfer(double activation) {
        return 1d / (1d + Math.exp(-activation));
    }

    /**
     * Calculates the derivative of an neuron output.
     *
     * @param output an output of a neuron.
     * @return the derivative of an neuron output.
     */
    private double transfer_sigmoid(double output) {
        return output * (1d - output);
    }

    /**
     * Updates the weights based on the specified values of data (row) using the
     * learning rate as specified in {@link #_learningRate}.
     *
     * @param values current vector (row) consisting of data.
     */
    private void updateWeights(RealVector values) {
        for (int i = 0; i < _network.size(); ++i) {
            RealVector inputs = values.copy();
            if (i != 0) {
                inputs = _network.get(i - 1).getOutput();
            }
            int index = 0;
            for (Neuron neuron : _network.get(i)) {
                RealVector temp = inputs.mapMultiply(-_learningRate
                        * _network.get(i).getDelta().getEntry(index));
                // update weights of neuron in layer...
                neuron.setWeights(neuron.getWeights().add(temp));
                // ...then update weight of bias neuron
                double bias = _network.get(i).getBias().getEntry(index);
                _network.get(i).getBias()
                        .setEntry(index, bias += (-_learningRate
                                * _network.get(i).getDelta().getEntry(index++)));
            }
        }
    }

    /**
     * Backpropagate error and store in neurons.
     *
     * @param expected vector consisting of expected values.
     */
    private void backpropagateError(RealVector expected) {
        final int outputLayerPos = _network.size() - 1;
        for (int i = outputLayerPos; i >= 0; --i) { // reverse loop (!)
            Layer currentLayer = _network.get(i);
            List<Double> errors = new ArrayList<>();
            if (i != outputLayerPos) {
                Layer prevLayer = _network.get(i + 1);
                for (int j = 0; j < currentLayer.getNeurons().size(); ++j) {
                    double error = 0d;
                    int counter = 0;
                    for (Neuron neuron : prevLayer) {
                        error += (neuron.getWeights().getEntry(j)
                                * prevLayer.getDelta().getEntry(counter++));
                    }
                    errors.add(error);
                }
            } else { // starts with output layer
                for (int j = 0; j < currentLayer.getNeurons().size(); ++j) {
                    errors.add(currentLayer.getOutput().getEntry(j) - expected.getEntry(j));
                }
            }
            // set delta values of neurons, multiply error with derivative
            for (int j = 0; j < currentLayer.getNeurons().size(); ++j) {
                currentLayer.getDelta().setEntry(j,
                        errors.get(j) * transfer_sigmoid(currentLayer.getOutput().getEntry(j)));
            }
        }
    }

    /**
     * Forward propagation for the specified vector with this network.
     *
     * @param values vector of data to be processed.
     * @return the values from the output layer.
     */
    private RealVector forwardPropagate(RealVector values) {
        RealVector inputs = values;
        for (Layer layer : _network) {
            int index = 0;
            for (Neuron neuron : layer) {
                double bias = layer.getBias().getEntry(index);
                double activation = bias + inputs.dotProduct(neuron.getWeights());
                layer.getOutput().setEntry(index++, transfer(activation));
            }
            inputs = layer.getOutput();
        }
        return inputs;
    }

    /**
     * Trains this network with the specified data.
     *
     * @param trainSet the data set the network is to be trained with.
     * @param numEpoch number of epochs.
     * @param numOutputs number of output values.
     */
    public void trainNetwork(DataSet trainSet, int numEpoch, int numOutputs) {
        trainNetwork(trainSet.getData(), numEpoch, numOutputs);
    }

    /**
     * Trains this network with the specified data.
     *
     * @param trainData the data the network is to be trained with.
     * @param numEpoch number of epochs.
     * @param numOutputs number of output values.
     */
    public void trainNetwork(List<InputVector> trainData, int numEpoch, int numOutputs) {
        for (int epoch = 0; epoch < numEpoch; ++epoch) {
            for (int j = 0; j < trainData.size(); ++j) {
                RealVector values = trainData.get(j).getValues();
                forwardPropagate(values); // discard output
                // create a binary vector for our expected output (one-hot encoding)
                RealVector expected = new ArrayRealVector(numOutputs);
                int expectedValue = trainData.get(j).getExpectedValueAsInteger();
                expected.setEntry(expectedValue, 1);
                backpropagateError(expected);
                updateWeights(values);
            }
        }
    }

    /**
     * Performs a prediction with an already trained network.
     *
     * @param values the values to be predicted.
     * @return the index of the element with the largest probability.
     */
    public int predict(RealVector values) {
        RealVector output = forwardPropagate(values);
        return output.getMaxIndex();
    }

    /**
     * Returns the identifier of this network.
     *
     * @return the identifier of this network.
     */
    public String getId() {
        return _id;
    }

    /**
     * Returns the currently set learning rate.
     *
     * @return the currently set learning rate.
     */
    public double getLearningRate() {
        return _learningRate;
    }

    /**
     * Sets a new learning rate. There are no checks made whatsoever.
     *
     * @param rate the new learning rate.
     */
    public void setLearningRate(double rate) {
        _learningRate = rate;
    }

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder();
        int counter = 0;
        for (Layer layer : _network) {
            sb.append("Layer").append(++counter)
                    .append(": ").append(layer.toString()).append("\n");
        }
        return sb.toString();
    }
}
