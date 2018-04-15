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

import at.fhv.machinelearning.utils.DataSetUtils;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.stream.Collectors;
import org.apache.commons.math3.linear.RealVector;

/**
 * Consists of a training set, a test set and a validation set which holds the
 * expected values for the test set.
 *
 * @author Matthias Fussenegger
 */
public final class Fold {

    private final DataSet _trainSet;
    private final List<RealVector> _testSet;
    private final List<Integer> _validationSet;

    public Fold(DataSet trainSet, List<RealVector> testSet, List<Integer> validationSet) {
        _trainSet = trainSet;
        _testSet = testSet;
        _validationSet = validationSet;
    }

    /**
     * Returns the aggregated training set.
     *
     * @return the aggregated training set.
     */
    public final DataSet getTrainSet() {
        return _trainSet;
    }

    /**
     * Returns the aggregated test set.
     *
     * @return the aggregated test set.
     */
    public final List<RealVector> getTestSet() {
        return _testSet;
    }

    /**
     * Returns the aggregated validation set which holds the expected values for
     * the test set.
     *
     * @return the aggregated validation set.
     */
    public final List<Integer> getValidationSet() {
        return _validationSet;
    }

    /**
     * Creates and returns a new {@link Fold} from the specified parameters.
     *
     * @param dataSet the data set to be split.
     * @param percentage the position the data set is to be split at.
     * @param method the normalization method to be used.
     * @return a new instance of {@link Fold}.
     * @throws IllegalArgumentException thrown if split data set is too small.
     */
    public static final Fold forPercentageSplit(DataSet dataSet,
            int percentage, NormalizationMethod method) {
        final List<DataSet> split = DataSetUtils.dataSetSplit(dataSet, percentage);

        if (split.size() < 2) {
            throw new IllegalArgumentException("Split data set is too small.");
        }

        final DataSet trainSet = split.get(0);
        final DataSet testData = split.get(1);
        final List<RealVector> testSet = testData.getData().stream().map(row
                -> row.getValues()).collect(Collectors.toList());

        DataSet normTrainSet = method.normalize(dataSet);
        List<RealVector> normTestSet = method.normalize(testSet, trainSet);
        // get expected values from test data
        List<Integer> expected = testData.getData().stream().map(s
                -> s.getExpectedValueAsInteger()).collect(Collectors.toList());
        return new Fold(normTrainSet, normTestSet, expected);
    }

    /**
     * Creates and returns a list consisting of {@link Fold} from the specified
     * parameters to be used for cross validation.
     *
     * @param splits a split data set (e.g. from a cross split).
     * @param method the normalization method to be used.
     * @return a new instance of {@link Fold}.
     */
    public static final List<Fold> forCrossValidation(
            List<DataSet> splits, NormalizationMethod method) {

        List<Fold> folds = new LinkedList<>();

        splits.forEach((split) -> {
            List<DataSet> fullTrainSet = new LinkedList<>(splits);
            fullTrainSet.remove(split); // remove current split
            List<InputVector> trainSet = new ArrayList<>(
                    fullTrainSet.size() * fullTrainSet.get(0).size());
            fullTrainSet.forEach(trainData -> trainSet.addAll(trainData.getData()));
            DataSet normTrainSet = method.normalize(new DataSet(trainSet));
            // fill test set from folded set
            List<RealVector> testSet = split.getData().stream().map(row
                    -> row.getValues()).collect(Collectors.toList());
            List<RealVector> normalizedTestSet = method.normalize(testSet, trainSet);
            // get expected values from current split
            List<Integer> expected = split.getData().stream().map(s
                    -> s.getExpectedValueAsInteger()).collect(Collectors.toList());
            // add new fold which holds trainings, test and validation set
            folds.add(new Fold(normTrainSet, normalizedTestSet, expected));
        });
        return folds;
    }

}
