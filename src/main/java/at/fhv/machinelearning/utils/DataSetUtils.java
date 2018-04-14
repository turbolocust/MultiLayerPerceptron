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

import at.fhv.machinelearning.DataSet;
import at.fhv.machinelearning.InputVector;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.Set;

/**
 *
 * @author Matthias Fussenegger
 */
public final class DataSetUtils {

    private DataSetUtils() {
        throw new AssertionError("Holds static members only.");
    }

    public static List<DataSet> dataSetSplit(DataSet dataSet, int percentage) {

        List<DataSet> dataSplit = new ArrayList<>(2);
        final List<InputVector> data = dataSet.getData();
        final int splitAt = (int) (data.size() * (percentage / 100d));

        int index = 0;
        final List<InputVector> firstSplit = new LinkedList<>();
        while (index < splitAt) {
            firstSplit.add(data.get(index++));
        }
        dataSplit.add(new DataSet(firstSplit));

        final List<InputVector> secondSplit = new LinkedList<>();
        while (index < data.size()) {
            secondSplit.add(data.get(index++));
        }
        dataSplit.add(new DataSet(secondSplit));

        return dataSplit;
    }

    public static List<DataSet> crossSplit(DataSet dataSet, int numFolds) {

        final List<InputVector> data = dataSet.getData();
        final List<DataSet> dataSplit = new ArrayList<>(numFolds);
        final List<InputVector> dataCopy = new LinkedList<>(data);
        final Random rand = new Random();

        for (int i = 0; i < numFolds; ++i) {
            List<InputVector> fold = new ArrayList<>();
            while (fold.size() < (data.size() / numFolds)) {
                int size = dataCopy.size();
                int index = rand.nextInt(size);
                fold.add(dataCopy.remove(index));
            }
            dataSplit.add(new DataSet(fold));
        }
        return dataSplit;
    }

    public static int determineNumberOfOutputNeurons(DataSet dataSet) {
        final Set<String> labels = new HashSet<>();
        dataSet.forEach(data -> labels.add(data.getExpectedValue()));
        return labels.size();
    }

}
