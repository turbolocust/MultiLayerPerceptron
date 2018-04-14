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
import java.util.Iterator;
import java.util.List;
import java.util.Spliterator;
import java.util.function.Consumer;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author Matthias Fussenegger
 * @author Johannes Stadelmann
 */
public final class Layer implements Serializable, Iterable<Neuron> {

    private static final long serialVersionUID = -709128887004825050L;

    private final List<Neuron> _neurons;
    private RealVector _output;
    private RealVector _bias;
    private RealVector _delta;

    public Layer() {
        _neurons = new ArrayList<>();
    }

    public Layer(int initialSize) {
        _neurons = new ArrayList<>(initialSize);
        _output = new ArrayRealVector(initialSize);
        _bias = new ArrayRealVector(initialSize); // initialize with zeros
//        _bias = new ArrayRealVector(new Random().doubles().toArray());
        _delta = new ArrayRealVector(initialSize);
    }

    public List<Neuron> getNeurons() {
        return _neurons;
    }

    @Override
    public Spliterator<Neuron> spliterator() {
        return _neurons.spliterator();
    }

    @Override
    public Iterator<Neuron> iterator() {
        return _neurons.iterator();
    }

    @Override
    public void forEach(Consumer<? super Neuron> action) {
        _neurons.forEach(action);
    }

    @Override
    public String toString() {
        return "Layer{" + "Neurons=" + _neurons + '}';
    }

    public RealVector getOutput() {
        return _output;
    }

    public void setOutput(RealVector output) {
        _output = output;
    }

    public RealVector getBias() {
        return _bias;
    }

    public void setBias(RealVector bias) {
        _bias = bias;
    }

    public RealVector getDelta() {
        return _delta;
    }

    public void setDelta(RealVector delta) {
        _delta = delta;
    }

}
