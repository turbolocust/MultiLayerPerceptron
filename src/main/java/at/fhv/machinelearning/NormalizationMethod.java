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

import at.fhv.machinelearning.utils.MathUtils;
import java.util.ArrayList;
import java.util.List;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author Matthias Fussenegger
 */
public enum NormalizationMethod {

    MIN_MAX {
        @Override
        public void normalizeInPlace(RealVector vector) {
            double min = vector.getMinValue(), max = vector.getMaxValue();
            normalizeInPlace(vector, min, max);
        }

        /**
         * Normalizes the specified {@link RealVector} considering the specified
         * minimum and maximum value.
         *
         * @param vector the vector to be normalized.
         * @param min the minimum value, e.g. from the train set.
         * @param max the maximum value, e.g. from the train set.
         */
        public void normalizeInPlace(RealVector vector, double min, double max) {
            for (int i = 0; i < vector.getDimension(); ++i) {
                double divisor = max - min;
                if (divisor != 0d) {
                    vector.setEntry(i, (vector.getEntry(i) - min) / divisor);
                }
            }
        }

        @Override
        public List<RealVector> normalize(List<RealVector> testSet, DataSet trainSet) {
            return normalize(testSet, trainSet.getData());
        }

        @Override
        public List<RealVector> normalize(List<RealVector> testSet, List<InputVector> trainData) {
            // transpose input vectors first
            final RealMatrix testMatrix = MathUtils.matrixFromRealVectors(testSet),
                    trainMatrix = MathUtils.matrixFromInputVectors(trainData);
            // normalize column-wise
            for (int col = 0; col < testMatrix.getColumnDimension(); ++col) {
                RealVector testCols = new ArrayRealVector(testMatrix.getColumn(col));
                RealVector trainCols = trainMatrix.getColumnVector(col);
                normalizeInPlace(testCols, trainCols.getMinValue(), trainCols.getMaxValue());
                testMatrix.setColumnVector(col, testCols);
            }
            return convertBack(testMatrix);
        }
    },
    Z_SCORE {
        @Override
        public void normalizeInPlace(RealVector vector) {
            int dim = vector.getDimension();
            double[] values = vector.toArray();
            double mean = MathUtils.calcMeanValue(values, dim);
            double sd = MathUtils.calcStandardDeviation(values, mean, dim);
            normalizeInPlace(vector, mean, sd);
        }

        /**
         * Normalizes the specified {@link RealVector} considering the specified
         * mean and standard derivation.
         *
         * @param vector the vector to be normalized.
         * @param mean the mean value, e.g. from the train set.
         * @param sd the standard derivation, e.g. from the train set.
         */
        public void normalizeInPlace(RealVector vector, double mean, double sd) {
            if (mean == 0) {
                return;
            }
            for (int i = 0; i < vector.getDimension(); ++i) {
                double newValue = (vector.getEntry(i) - mean) / sd;
                if (!Double.isNaN(newValue)) {
                    vector.setEntry(i, newValue);
                }
            }
        }

        @Override
        public List<RealVector> normalize(List<RealVector> testSet, DataSet trainSet) {
            return normalize(testSet, trainSet.getData());
        }

        @Override
        public List<RealVector> normalize(List<RealVector> testSet, List<InputVector> trainData) {
            // transpose input vectors first
            final RealMatrix testMatrix = MathUtils.matrixFromRealVectors(testSet),
                    trainMatrix = MathUtils.matrixFromInputVectors(trainData);
            // normalize column-wise
            for (int col = 0; col < testMatrix.getColumnDimension(); ++col) {
                RealVector columns = new ArrayRealVector(testMatrix.getColumn(col));
                double[] values = trainMatrix.getColumn(col);
                double mean = MathUtils.calcMeanValue(values, values.length);
                double sd = MathUtils.calcStandardDeviation(values, mean, values.length);
                normalizeInPlace(columns, mean, sd);
                testMatrix.setColumnVector(col, columns);
            }
            return convertBack(testMatrix);
        }
    };

    /**
     * Normalizes the specified {@link RealVector} in-place.
     *
     * @param vector the vector to be normalized.
     */
    public abstract void normalizeInPlace(RealVector vector);

    /**
     * Normalizes the specified {@code testSet}.
     *
     * @param testSet the test set to be normalized.
     * @param trainSet train set to be used for normalization of the test set.
     * @return normalized list of {@link RealVector}.
     */
    public abstract List<RealVector> normalize(
            List<RealVector> testSet, DataSet trainSet);

    /**
     * Normalizes the specified {@code testSet}.
     *
     * @param testSet the test set to be normalized.
     * @param trainData train data to be used for normalization of the test set.
     * @return normalized list of {@link RealVector}.
     */
    public abstract List<RealVector> normalize(
            List<RealVector> testSet, List<InputVector> trainData);

    /**
     * Normalizes the specified dataset entirely (out-place).
     *
     * @param dataSet the dataset to be normalized.
     * @return the normalized dataset.
     */
    public DataSet normalize(DataSet dataSet) {
        return normalize(dataSet.getData());
    }

    /**
     * Normalizes the specified list of input vectors entirely (out-place).
     *
     * @param data the list of input vectors to be normalized.
     * @return the normalized dataset.
     */
    public DataSet normalize(List<InputVector> data) {
        // transpose input vectors first
        final int cols = data.get(0).getValues().getDimension();
        final RealMatrix matrix = MathUtils.matrixFromInputVectors(data);

        for (int col = 0; col < cols; ++col) { // normalize column-wise
            RealVector columns = new ArrayRealVector(matrix.getColumn(col));
            normalizeInPlace(columns); // train set is to be fully normalized
            matrix.setColumnVector(col, columns);
        }

        // convert back
        final int rows = data.size();
        List<InputVector> normalizedData = new ArrayList<>(rows);
        for (int row = 0; row < rows; ++row) {
            RealVector rowData = matrix.getRowVector(row);
            String expected = data.get(row).getExpectedValue();
            normalizedData.add(new InputVector(rowData, expected));
        }
        return new DataSet(normalizedData);
    }

    private static List<RealVector> convertBack(RealMatrix matrix) {
        final int rows = matrix.getRowDimension();
        List<RealVector> normalizedData = new ArrayList<>(rows);
        for (int row = 0; row < rows; ++row) {
            normalizedData.add(matrix.getRowVector(row));
        }
        return normalizedData;
    }

}
