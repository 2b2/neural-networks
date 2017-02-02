package de.ef.slowwave;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Map;

import de.ef.neuralnetworks.NeuralNetwork;
import de.ef.neuralnetworks.NeuralNetworkContext;
import de.ef.neuralnetworks.NeuralNetworkContextFactory;

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
	@SuppressWarnings("unchecked")
	public <I, O> NeuralNetwork<I, O> createNeuralNetwork(Class<I> inputClass, Class<O> outputClass, Map<String, Object> properties){
		if(inputClass == double[].class && outputClass == double[].class)
			return (NeuralNetwork<I, O>)new SlowWave(0, new int[]{0}, 0); // FIXME sizes
		return null;
	}
}
