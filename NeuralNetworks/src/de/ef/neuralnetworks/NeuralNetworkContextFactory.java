package de.ef.neuralnetworks;

import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public final class NeuralNetworkContextFactory{
	
	private NeuralNetworkContextFactory(){}
	
	
	
	private final static Map<String, Class<NeuralNetworkContext>> IMPLEMENTATIONS =
		new ConcurrentHashMap<>();
	
	@SuppressWarnings("unused")
	private static void register(String implementationName, Class<NeuralNetworkContext> contextClass){
		IMPLEMENTATIONS.put(implementationName.toLowerCase(Locale.ENGLISH), contextClass);
	}
	
	public static NeuralNetworkContext create(String implementationName){
		Class<NeuralNetworkContext> contextClass =
			IMPLEMENTATIONS.get(implementationName.toLowerCase(Locale.ENGLISH));
		
		try{
			return contextClass.newInstance();
		}
		catch(InstantiationException | IllegalAccessException e){
			throw new RuntimeException(e);
		}
	}
}
