package de.ef.neuralnetworks;

// the core class of all implementations of this API
// the implementation may or may not be thread-safe
// version: 1.1, date: 06.06.2016, author: Erik Fritzsche
public interface NeuralNetwork{
	
	// runs the neural network with the given inputs
	// and returns the values of the output layer
	public double[] calculate(double inputs[]);
	
	// adjusts the network based on the given input set and the expected outputs
	// the implementation of the training algorithm is up to the network
	// the total network output error is returned (before adjustment)
	// the network could be locked while training but this depends on the implementation
	public double train(double inputs[], double outputs[]);
	
	// see the training function with two arguments for more details
	// this function adjusts the learning rate if the underlying implementation
	// supports that (like the often used backpropagation)
	// if not the learningRate argument should have no effect
	public double train(double inputs[], double outputs[], double learningRate);
	
	
	// TODO comment
	public boolean hasBias();
	
	public int getLayerCount();
	
	public int getNeuronCount(int layer);
	
	public double getWeight(int layer, int neuron, int weight);
	
	public void setWeight(int layer, int neuron, int weight, double value);
}