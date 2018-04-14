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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import java.util.Spliterator;
import java.util.function.Consumer;

/**
 *
 * @author Matthias Fussenegger
 */
public final class DataSet implements Iterable<InputVector> {

    private final List<InputVector> _data;

    public DataSet(List<InputVector> data) {
        _data = data;
    }

    @SafeVarargs
    public DataSet(DataSet... sets) {
        Objects.requireNonNull(sets);
        _data = new ArrayList<>(256);
        Arrays.stream(sets).map(set
                -> set._data).forEachOrdered(_data::addAll);
    }

    public final List<InputVector> getData() {
        return _data;
    }

    public final int size() {
        return _data.size();
    }

    @Override
    public Spliterator<InputVector> spliterator() {
        return _data.spliterator();
    }

    @Override
    public Iterator<InputVector> iterator() {
        return _data.iterator();
    }

    @Override
    public void forEach(Consumer<? super InputVector> action) {
        _data.forEach(action);
    }

    /**
     * Creates and returns an empty data set.
     *
     * @return an empty data set.
     */
    public static final DataSet emptyDataSet() {
        return new DataSet(Collections.emptyList());
    }
}
