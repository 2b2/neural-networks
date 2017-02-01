package de.ef.neuralnetworks;

import java.io.Serializable;

/**
 * A structure that holds the input data for a
 * {@link de.ef.neuralnetworks.NeuralNetwork NeuralNetwork}.
 * 
 * @author Erik Fritzsche
 * @version 1.0
 * @since 1.0
 */
public interface NeuralNetworkData
	extends Serializable{
	
	/**
	 * Retrieve input data for a
	 * {@link de.ef.neuralnetworks.NeuralNetwork NeuralNetwork}.
	 * 
	 * @return the input data
	 */
	public double[] getData();
}
