package de.ef.neuralnetworks;

import java.io.IOException;

public interface ConvolutionalNeuralNetwork<I, C, O>
	extends NeuralNetwork<I, O>{
	
	@Override
	public default O calculate(I input) throws IOException{
		return this.calculateFullyConnected(this.calculateFilters(input));
	}
	
	public C calculateFilters(I input) throws IOException;
	
	public O calculateFullyConnected(C input) throws IOException;
}
