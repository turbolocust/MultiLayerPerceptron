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

import java.util.function.Consumer;

/**
 *
 * @author Matthias Fussenegger
 */
public class NetworkRunnerAsync extends NetworkRunner implements Runnable {

    private final DataSet _dataSet;
    private final int _numEpochs;

    private Consumer<String> _callback;

    /**
     * Constructs a new instance of {@link NetworkRunner} which can be executed
     * in a thread to perform training of network asynchronously.
     *
     * @param network the network to be aggregated.
     * @param dataSet the data set to be used for training.
     * @param numEpochs the number of epochs to be performed during training.
     */
    public NetworkRunnerAsync(NN network, DataSet dataSet, int numEpochs) {
        super(network);
        _dataSet = dataSet;
        _numEpochs = numEpochs;
    }

    /**
     * Sets a new callback ({@link Consumer} to be called when the task is
     * completed, which means that the network is trained.
     *
     * @param callback will be called when task is completed.
     */
    public final void setOnCompleted(Consumer<String> callback) {
        _callback = callback;
    }

    @Override
    public void run() {
        trainNetwork(_dataSet, _numEpochs);
        if (_callback != null) {
            _callback.accept(_network.getId());
        }
    }

}
