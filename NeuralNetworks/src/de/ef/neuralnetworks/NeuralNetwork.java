package de.ef.neuralnetworks;

import java.io.IOException;
import java.io.Serializable;
import java.util.Map;

/**
 * The interface {@code NeuralNetwork} is the core of this API.<br>
 * A neural-network is formed from layers of neurons,
 * how these neurons are connected together is up to the implementation.
 * <p>
 * <b>Important:</b> A implementation does <b>not</b> necessary need to be <b>thread-safe</b>.
 * </p>
 * 
 * @param I input type
 * @param O output type
 * 
 * @author Erik Fritzsche
 * @version 3.0
 * @since 1.0
 */
public interface NeuralNetwork<I, O>
	extends Serializable{
	
	/**
	 * Loads a neural-network based on the given class name.
	 * 
	 * @param className full name of the class including package
	 * 
	 * @return a new uninitialized instance of the requested neural-network
	 * 
	 * @throws ClassNotFoundException if the class cannot be located
	 * @throws IllegalAccessException if the class or its constructor is not accessible
	 * @throws InstantiationException if the instantiation failed
	 */
	public static <I, O> NeuralNetwork<I, O> load(String className)
			throws ClassNotFoundException, InstantiationException, IllegalAccessException{
		@SuppressWarnings("unchecked")
		Class<NeuralNetwork<I, O>> classObject = (Class<NeuralNetwork<I, O>>)Class.forName(className);
		
		return classObject.newInstance();
	}
	
	
	
	/**
	 * Initializes the neural-network and configures the underlying implementation.
	 * 
	 * @param inputSize the size of the input layer
	 * @param hiddenSizes the sizes of the hidden layers in order from closest to the input layer to closest to the output layer
	 * @param outputSize the size of the output layer
	 * @param properties additional properties of the neural-network
	 * 
	 * @throws IOException if the underlying implementation experienced an error
	 * @throws NullPointerException
	 *     if {@code hiddenSizes == null} or if {@code properties == null} and the implementation does not permit this
	 * @throws IllegalArgumentException
	 *     if a size is equal or less than zero or an important property for the implementation is missing
	 * @throws IllegalStateException if the neural-network is already initialized
	 */
	public void init(int inputSize, int hiddenSizes[], int outputSize, Map<String, Object> properties) throws IOException;
	
	
	/**
	 * Calculates an output state based on the given input state.
	 * 
	 * @param input the state of the neurons inside the first layer
	 * 
	 * @return the output state of the neurons inside the last layer
	 * 
	 * @throws IOException if the underlying implementation experienced an error
	 * @throws NullPointerException if {@code input == null}
	 */
	public O calculate(I input) throws IOException;
	
	
	/**
	 * Changes the neural-network internally so that the output of {@link #calculate(I) calculate}
	 * called with {@code input} after a call to this function matches the expected {@code output}
	 * closer than before.
	 * <p>
	 * <i>Note:</i> In some cases with some implementations this function may change nothing
	 * or make the difference even bigger.
	 * </p>
	 * 
	 * @param input the state of the neurons inside the first layer
	 * @param output the expected state of the neurons inside the last layer
	 * 
	 * @return the total error of the neural-network before adjustments
	 * 
	 * @throws IOException if the underlying implementation experienced an error
	 * @throws NullPointerException if {@code input == null} or {@code output == null}
	 */
	public double train(I input, O output) throws IOException;
}
