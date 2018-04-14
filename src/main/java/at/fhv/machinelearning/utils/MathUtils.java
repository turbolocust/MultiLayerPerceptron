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
package at.fhv.machinelearning.utils;

import at.fhv.machinelearning.InputVector;
import java.util.Arrays;
import java.util.List;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author Matthias Fussenegger
 */
public final class MathUtils {

    private MathUtils() {
        throw new AssertionError("Holds static members only.");
    }

    public static double calcMeanValue(double[] values, int n) {
        return Arrays.stream(values).sum() / n;
    }

    public static double calcStandardDeviation(double[] values, double mean, int n) {
        double temp = 0d;
        for (int i = 0; i < values.length; ++i) {
            temp += Math.pow(values[i] - mean, 2);
        }
        return Math.sqrt(Math.abs(temp / n - 1));
    }

    public static RealMatrix matrixFromInputVectors(List<InputVector> dataSet) {
        final int rows = dataSet.size();
        final int cols = dataSet.get(0).getValues().getDimension();
        RealMatrix matrix = MatrixUtils.createRealMatrix(rows, cols);

        for (int row = 0; row < rows; ++row) { // fill matrix
            matrix.setRowVector(row, dataSet.get(row).getValues());
        }
        return matrix;
    }

    public static RealMatrix matrixFromRealVectors(List<RealVector> data) {
        final int rows = data.size();
        final int cols = data.get(0).getDimension();
        RealMatrix matrix = MatrixUtils.createRealMatrix(rows, cols);

        for (int row = 0; row < rows; ++row) { // fill matrix
            matrix.setRowVector(row, data.get(row));
        }
        return matrix;
    }
}
