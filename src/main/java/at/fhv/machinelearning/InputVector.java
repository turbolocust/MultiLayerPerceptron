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

import org.apache.commons.math3.linear.RealVector;

/**
 * Represents a row of the dataset to be classified. Holds an additional field
 * for the class label with some additional convenience methods.
 *
 * @author Matthias Fussenegger
 */
public final class InputVector {

    private final RealVector _values;

    private final String _expectedValue;

    public InputVector(RealVector values, String classIdentifier) {
        _values = values;
        _expectedValue = classIdentifier;
    }

    /**
     * Returns a vector with input data, which is a row of the dataset excluding
     * the class label information.
     *
     * @return a vector with input data excluding the class label.
     */
    public RealVector getValues() {
        return _values;
    }

    /**
     * Returns the expected value which is the class label.
     *
     * @return the expected value which is the class label.
     */
    public String getExpectedValue() {
        return _expectedValue;
    }

    public double getExpectedValueAsDouble() {
        String numberString = stringToNumberString();
        return Double.parseDouble(numberString); // is safe
    }

    public int getExpectedValueAsInteger() {
        String numberString = stringToNumberString().replace(".", "");
        return Integer.parseInt(numberString); // is safe 
    }

    private String stringToNumberString() {
        final StringBuilder sb = new StringBuilder(_expectedValue.length());
        for (int i = 0; i < _expectedValue.length(); ++i) {
            char nextChar = _expectedValue.charAt(i);
            if ((nextChar > 47 && nextChar < 58) || nextChar == '.') {
                sb.append(nextChar);
            }
        }
        return sb.toString();
    }

}
