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

import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author Matthias Fussenegger
 */
public class NetworkRunner {

    protected final NN _network;

    public NN getNetwork() {
        return _network;
    }

    /**
     * Constructs a new instance of {@link NetworkRunner} which can be used for
     * synchronous execution.
     *
     * @param network the network to be aggregated.
     */
    public NetworkRunner(NN network) {
        _network = network;
    }

    /**
     * Trains the aggregated network with the specified data set for the
     * specified amount of epochs.
     *
     * @param trainData training data to be used.
     * @param numEpoch the amount of training epochs.
     */
    public final void trainNetwork(DataSet trainData, int numEpoch) {
        // count number of output neurons first
        final Set<String> labels = new HashSet<>();
        trainData.forEach(data -> labels.add(data.getExpectedValue()));
        _network.trainNetwork(trainData, numEpoch, labels.size());
    }

    /**
     * Performs a prediction with the aggregated network that should already be
     * trained.
     *
     * @param testData the data to be used for prediction.
     * @return a list consisting of predictions for each {@code testData} entry.
     */
    public final List<Integer> predict(List<RealVector> testData) {
        List<Integer> predictions = new LinkedList<>();
        testData.forEach((vector) -> {
            predictions.add(_network.predict(vector));
        });
        return predictions;
    }

    /**
     * Calculates the metrics from the predicted classes of a network.
     *
     * @param predicted the predicted classes of a network.
     * @param expected the actual classes to be compared.
     * @return the success rate in percentage.
     */
    public double calculateMetrics(
            List<Integer> predicted, List<Integer> expected) {
        int numCorrect = 0;
        for (int i = 0; i < expected.size(); ++i) {
            if (Objects.equals(expected.get(i), predicted.get(i))) {
                ++numCorrect;
            }
        }
        return numCorrect / (double) expected.size() * 100;
    }

}
