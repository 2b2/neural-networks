package de.ef.neuralnetworks;

import java.util.Map;

public interface NeuralNetworkContext{
	
	public final static String
		INPUT_LAYER_SIZE = "layers.input.size", OUTPUT_LAYER_SIZE = "layers.output.size",
		HIDDEN_LAYER_COUNT = "layers.hidden.count", HIDDEN_LAYER_SIZE = "layers.hidden[*].size";
	
	
	
	public <I, O> NeuralNetwork<I, O> createNeuralNetwork(Class<I> inputClass, Class<O> outputClass, Map<String, Object> properties);
}
