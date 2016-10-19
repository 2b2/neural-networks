package de.ef.neuralnetworks;

// defines a neural network factory, does what it says
// version: 1.0, date: 19.10.2016, author: Erik Fritzsche
public interface NeuralNetworkFactory{
	
	public NeuralNetwork createNeuralNetwork(int inputSize, int hiddenSizes[], int outputSize);
}