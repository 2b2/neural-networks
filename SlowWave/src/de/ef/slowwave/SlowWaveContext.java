package de.ef.slowwave;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Map;

import de.ef.neuralnetworks.NeuralNetwork;
import de.ef.neuralnetworks.NeuralNetworkContext;
import de.ef.neuralnetworks.NeuralNetworkContextFactory;
import de.ef.neuralnetworks.NeuralNetworkWrapper;

public class SlowWaveContext
	implements NeuralNetworkContext{
	
	static{
		try{
			Method register =
				NeuralNetworkContextFactory.class.getDeclaredMethod("register", String.class, Class.class);
			register.setAccessible(true);
			register.invoke(null, "SlowWave", SlowWaveContext.class);
		}catch(NoSuchMethodException | IllegalAccessException | IllegalArgumentException | InvocationTargetException e){
			throw new RuntimeException(e);
		}
	}
	
	
	
	public SlowWaveContext(){}
	
	
	@Override
	public <I, O> NeuralNetwork<I, O> createNeuralNetwork(Class<I> inputClass, Class<O> outputClass, Map<String, Object> properties){
		int inputLayerSize = (Integer)properties.get(INPUT_LAYER_SIZE);
		int outputLayerSize = (Integer)properties.get(OUTPUT_LAYER_SIZE);
		int hiddenLayerCount = (Integer)properties.get(HIDDEN_LAYER_COUNT);
		int hiddenLayerSizes[] = new int[hiddenLayerCount];
		for(int i = 0; i < hiddenLayerCount; i++)
			hiddenLayerSizes[i] = (Integer)properties.get(HIDDEN_LAYER_SIZE.replace("*", String.valueOf(i)));
		
		return NeuralNetworkWrapper.wrapPrimitiveArray(
			new SlowWave(inputLayerSize, hiddenLayerSizes, outputLayerSize),
			double[].class, inputClass, outputClass
		);
	}
}
