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

/**
 *
 * @author Matthias Fussenegger
 */
public final class NetworkRunners {

    private NetworkRunners() {
        throw new AssertionError("Holds static members only.");
    }

    /**
     * Creates a new {@link NetworkRunner} which can be used for synchronous
     * execution.
     *
     * @param network the network to be aggregated.
     * @return a new instance of {@link NetworkRunner}.
     */
    public static NetworkRunner createForSynchronousExecution(NN network) {
        return new NetworkRunner(network);
    }

    /**
     * Creates a new {@link NetworkRunner} which can be executed in a thread to
     * perform training of network asynchronously.
     *
     * @param network the network to be aggregated.
     * @param dataSet the data set to be used for training.
     * @param numEpochs the number of epochs to be performed during training.
     * @return a new instance of {@link NetworkRunner}.
     */
    public static NetworkRunnerAsync createForAsynchronousExecution(
            NN network, DataSet dataSet, int numEpochs) {
        return new NetworkRunnerAsync(network, dataSet, numEpochs);
    }

}
