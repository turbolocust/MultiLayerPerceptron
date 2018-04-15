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

import at.fhv.machinelearning.utils.AppUtils;
import at.fhv.machinelearning.utils.DataSetUtils;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.apache.commons.math3.linear.ArrayRealVector;

/**
 *
 * @author Matthias Fussenegger
 * @author Johannes Stadelmann
 */
public final class App {

    private static final Logger LOG = Logger.getLogger(App.class.getName());

    private App() {
        throw new AssertionError("Holds static members only.");
    }

    /**
     * Main entry point of application.
     *
     * @param args the command line arguments.
     */
    public static void main(String[] args) {
        ExecutorService executor = Executors.newCachedThreadPool();
        CountDownLatch countDownLatch = new CountDownLatch(1);
        try {
            executor.execute(testWithLetterOrigCSV(countDownLatch)); // with 66% split
            testWithSeedsCSV(); // synchronously, with CrossValidation (10 folds)
            countDownLatch.await();
        } catch (InterruptedException ex) {
            LOG.log(Level.SEVERE, null, ex);
        } finally {
            executor.shutdown();
        }
    }

    @SuppressWarnings("SleepWhileInLoop")
    private static Runnable testWithLetterOrigCSV(CountDownLatch countDownLatch) {
        try {
            final String res = "/assets/lettersOrig1000.csv";
            String csv = AppUtils.getResource(App.class, res);
            DataSet dataSet = readCSV(csv, ",", true, ClassLabelPos.LAST);

            Collections.shuffle(dataSet.getData());

            final Fold fold = Fold.forPercentageSplit(dataSet, 66, NormalizationMethod.MIN_MAX);

            final int numInput = fold.getTestSet().get(0).getDimension();
            final int numOutput = DataSetUtils.determineNumberOfOutputNeurons(dataSet);
            final int numHidden = 40;

            NN network = NN.create("MNIST", numInput, numHidden, numOutput);
            network.setLearningRate(0.01);

            final NetworkRunnerAsync asyncRunner = NetworkRunners
                    .createForAsynchronousExecution(network, fold.getTrainSet(), 500);

            // train network asynchronously
            asyncRunner.setOnCompleted((String s) -> {
                List<Integer> predicted = asyncRunner.predict(fold.getTestSet());
                List<Integer> expected = fold.getValidationSet();

                double result = asyncRunner.calculateMetrics(predicted, expected);
                LOG.log(Level.INFO, "Success rate: {0}", result);

                try { // serialize created network
                    serialize(network, "nn_letters.ser");
                } catch (IOException ex) {
                    LOG.log(Level.SEVERE, null, ex);
                }
                countDownLatch.countDown();
            });

            return asyncRunner;

        } catch (IOException | URISyntaxException ex) {
            LOG.log(Level.SEVERE, null, ex);
        }

        return () -> {
            // no-op
        };
    }

    private static void testWithSeedsCSV() {
        try {
            final String res = "/assets/seeds_dataset.csv";
            String csv = AppUtils.getResource(App.class, res);
            DataSet dataSet = readCSV(csv, ",", false, ClassLabelPos.LAST);

            Collections.shuffle(dataSet.getData());

            List<DataSet> splits = DataSetUtils.crossSplit(dataSet, 10); // 10 folds
            final List<Fold> folds = Fold.forCrossValidation(splits, NormalizationMethod.MIN_MAX);

            final int numOutput = DataSetUtils.determineNumberOfOutputNeurons(dataSet);
            final int numHidden = 10;

            final StringBuilder sb = new StringBuilder();

            folds.stream().map((fold) -> {
                int numInput = fold.getTestSet().get(0).getDimension();
                NN network = NN.create("SEEDS", numInput, numHidden, numOutput);
                NetworkRunner runner = new NetworkRunner(network);
                runner.trainNetwork(fold.getTrainSet(), 100); // epochs
                List<Integer> predicted = runner.predict(fold.getTestSet());
                List<Integer> expected = fold.getValidationSet();
                return runner.calculateMetrics(predicted, expected);
            }).forEachOrdered((result) -> {
                sb.append(result).append(", ");
            });

            // remove last comma and insert brackets at beginning and end
            sb.deleteCharAt(sb.length() - 2).insert(0, "[ ").append("]");
            LOG.log(Level.INFO, "Success rates: {0}", sb);
        } catch (IOException | URISyntaxException ex) {
            LOG.log(Level.SEVERE, null, ex);
        }
    }

    /**
     * Serializes the specified object and stores it at the specified location.
     *
     * @param object the object to be serialized
     * @param location the location where to store the serialized data.
     * @throws IOException if an I/O error occurred.
     */
    static final void serialize(final Object object, final String location) throws IOException {
        // serialize the whole object
        try (FileOutputStream fos = new FileOutputStream(location);
                ObjectOutputStream oos = new ObjectOutputStream(fos);) {
            oos.writeObject(object);
            LOG.log(Level.INFO, "Saved serialized data to {0}", location);
        }
    }

    /**
     * Deserializes the serialized network data at the specified location and
     * returns an instance of {@link NN}.
     *
     * @param location the location of the serialized network.
     * @return an instance of {@link NN} created from the deserialized data.
     * @throws IOException if an I/O error occurred.
     * @throws ClassNotFoundException assertion error, {@link NN} not found.
     */
    static NN deserializeNetwork(final String location) throws IOException, ClassNotFoundException {
        NN network;
        try (final FileInputStream fis = new FileInputStream(location);
                final ObjectInputStream ois = new ObjectInputStream(fis);) {
            network = (NN) ois.readObject();
            LOG.log(Level.INFO, "Deserialized data from ", location);
        }
        return network;
    }

    /**
     * Reads a CSV file from {@code filename} considering the specified
     * {@code separator}.
     *
     * @param filename the file to be read.
     * @param separator the delimiter/separator.
     * @param skipHeader true to skip the first line, false otherwise.
     * @param labelPos the position of the class label.
     * @return a list consisting of {@link InputVector}.
     * @throws IOException if an I/O error occurred.
     */
    static final DataSet readCSV(String filename, String separator,
            boolean skipHeader, ClassLabelPos labelPos) throws IOException {

        List<InputVector> inputVectors = new ArrayList<>();
        try (final FileReader fr = new FileReader(filename);
                final BufferedReader br = new BufferedReader(fr);) {
            if (skipHeader) {
                br.readLine();
            }
            String line; // buffer
            while ((line = br.readLine()) != null) {
                List<String> splits = new ArrayList<>(Arrays.asList(line.split(separator)));
                String label;
                switch (labelPos) {
                    case FIRST: {
                        label = splits.remove(0);
                    }
                    break;
                    case LAST: {
                        label = splits.remove(splits.size() - 1);
                    }
                    break;
                    default:
                        throw new AssertionError(labelPos.name());
                }
                label = label.trim(); // remove leading and trailing whitespace
                double[] values = new double[splits.size()]; // data
                for (int i = 0; i < values.length; ++i) {
                    values[i] = Double.parseDouble(splits.get(i).trim());
                }
                inputVectors.add(new InputVector(new ArrayRealVector(values), label));
            }
        }
        return new DataSet(inputVectors);
    }

    private enum ClassLabelPos {
        FIRST, LAST;
    }
}
