package de.ef.neuralnetworks;

import java.util.Map;

public interface NeuralNetworkContext{
	
	public <I, O> NeuralNetwork<I, O> createNeuralNetwork(Class<I> inputClass, Class<O> outputClass, Map<String, Object> properties);
}
