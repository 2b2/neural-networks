package de.ef.neuralnetworks;

import java.io.IOException;
import java.io.Serializable;

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
 * @version 2.0
 * @since 1.0
 */
public interface NeuralNetwork<I, O>
	extends Serializable{
	
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
	
	/**
	 * This function adds control over the learning rate to the
	 * {@link #train(I, O) training} function.
	 * <p>
	 * Some training algorithm like the popular backpropagation support a learning rate
	 * which tunes how drastic the internal changes are on each adjustment.
	 * </p>
	 * <p>
	 * <i>Note:</i> If a learning rate is not supported by the underlying training algorithm
	 * then this function <u>must</u> do exactly the same as {@link #train(I, O)}
	 * and ignore the {@code learningRate}.
	 * </p>
	 * 
	 * @param input the state of the neurons inside the first layer
	 * @param output the expected state of the neurons inside the last layer
	 * @param learningRate a fine-tuning parameter for the training algorithm
	 * 
	 * @return the total error of the neural-network before adjustments
	 * 
	 * @throws IOException if the underlying implementation experienced an error
	 * @throws NullPointerException if {@code input == null} or {@code output == null}
	 */
	public double train(I input, O output, double learningRate) throws IOException;
}
