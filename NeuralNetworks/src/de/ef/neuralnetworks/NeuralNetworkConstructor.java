package de.ef.neuralnetworks;

// defines some constructors for neural networks which implement this API
// this is important to register them to the NeuralNetworkConnector
// version: 1.0, date: 05.06.2016, author: Erik Fritzsche
public interface NeuralNetworkConstructor{
	
	public NeuralNetwork createNeuralNetwork(int inputSize, int hiddenSizes[], int outputSize);
}