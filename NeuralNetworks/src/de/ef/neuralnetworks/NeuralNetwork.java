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
 * @author Erik Fritzsche
 * @version 1.0
 * @since 1.0
 */
public interface NeuralNetwork
	extends Serializable{
	
	/**
	 * Calculates an output state based on the given input state.
	 * 
	 * @param inputs the states of the neurons inside the first layer
	 * 
	 * @return the output states of the neurons inside the last layer
	 * 
	 * @throws IOException if the underlying implementation experienced an error
	 * @throws NullPointerException if {@code inputs == null}
	 * @throws ArrayIndexOutOfBoundsException if {@code inputs.length} is smaller then the size of the first layer
	 */
	public double[] calculate(double inputs[]) throws IOException;
	
	
	/**
	 * Changes the neural-network internally so that the output of {@link #calculate(double[]) calculate}
	 * called with {@code inputs} after a call to this function matches the expected {@code outputs}
	 * closer than before.
	 * <p>
	 * <i>Note:</i> In some cases with some implementations this function may change nothing
	 * or make the difference even bigger.
	 * </p>
	 * 
	 * @param inputs the states of the neurons inside the first layer
	 * @param outputs the expected states of the neurons inside the last layer
	 * 
	 * @return the total error of the neural-network before adjustments
	 * 
	 * @throws IOException if the underlying implementation experienced an error
	 * @throws NullPointerException if {@code inputs == null} or {@code outputs == null}
	 * @throws ArrayIndexOutOfBoundsException if {@code inputs.length} is smaller then the size of the first layer
	 * or {@code outputs.length} is smaller then the size of the last layer
	 */
	public double train(double inputs[], double outputs[]) throws IOException;
	
	/**
	 * This function adds control over the learning rate to the
	 * {@link #train(double[], double[]) training} function.
	 * <p>
	 * Some training algorithm like the popular backpropagation support a learning rate
	 * which tunes how drastic the internal changes are on each adjustment.
	 * </p>
	 * <p>
	 * <i>Note:</i> If a learning rate is not supported by the underlying training algorithm
	 * then this function <u>must</u> do exactly the same as {@link #train(double[], double[])}
	 * and ignore the {@code learningRate}.
	 * </p>
	 * 
	 * @param inputs the states of the neurons inside the first layer
	 * @param outputs the expected states of the neurons inside the last layer
	 * @param learningRate a fine-tuning parameter for the training algorithm
	 * 
	 * @return the total error of the neural-network before adjustments
	 * 
	 * @throws IOException if the underlying implementation experienced an error
	 * @throws NullPointerException if {@code inputs == null} or {@code outputs == null}
	 * @throws ArrayIndexOutOfBoundsException if {@code inputs.length} is smaller then the size of the first layer
	 * or {@code outputs.length} is smaller then the size of the last layer
	 */
	public double train(double inputs[], double outputs[], double learningRate) throws IOException;
}
