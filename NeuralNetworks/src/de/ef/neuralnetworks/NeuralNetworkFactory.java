package de.ef.neuralnetworks;

import java.util.Arrays;
import java.util.Locale;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import de.ef.slowwave.SlowWave;

/**
 * The class {@code NeuralNetworkFactory} creates
 * {@link de.ef.neuralnetworks.NeuralNetwork NeuralNetworks}
 * based on a given configuration and selects a specified
 * implementation.
 * 
 * @author Erik Fritzsche
 * @version 1.0
 * @since 1.0
 */
public class NeuralNetworkFactory{
	
	private NeuralNetworkFactory(){}
	
	
	
	/**
	 * Constructs a {@link de.ef.neuralnetworks.NeuralNetwork NeuralNetwork}
	 * based on the given parameters.
	 * <p>
	 * The configuration string has to be in JSON format and must contain a
	 * {@code implementation} field with the name of the implementation (this
	 * string is case insensitive). Most, if not all, implementations will
	 * require additional fields like a {@code layers} integer array which
	 * describes the number and sizes of the individual layers.
	 * </p>
	 * <p>
	 * <i>Example:</i>
	 * <pre>
	 * {
	 *   "implementation": "SlowWave",
	 *   "layers": [3, 2, 1]
	 * }
	 * </pre>
	 * </p>
	 * 
	 * @param json the configuration of the neural-network in JSON format
	 * 
	 * @return a {@link de.ef.neuralnetworks.NeuralNetwork NeuralNetwork}
	 * with the specified parameters or {@code null} if the requested
	 * implementation was not found
	 */
	public static NeuralNetwork create(String json){
		JsonObject config = new JsonParser().parse(json).getAsJsonObject();
		
		String implementation =
			config.get("implementation").getAsString().toLowerCase(Locale.ENGLISH);
		switch(implementation){
			case "slowwave" : return NeuralNetworkFactory.createSlowWave(config);
			default : return null;
		}
	}
	
	
	// internal creation of a SlowWave NeuralNetwork
	private static SlowWave createSlowWave(JsonObject config){
		JsonArray layers = config.get("layers").getAsJsonArray();
		int layerSizes[] = new int[layers.size()];
		for(int i = 0; i < layers.size(); i++){
			layerSizes[i] = layers.get(i).getAsInt();
		}
		
		// TODO throw exception if layerSizes.length < 2
		
		return new SlowWave(
			layerSizes[0],
			Arrays.copyOfRange(layerSizes, 1, layerSizes.length - 1),
			layerSizes[layerSizes.length - 1]
		);
	}
}
