package de.ef.slowwave;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import de.ef.neuralnetworks.NeuralNetwork;

/**
 * {@code SlowWave} is a simple single-threaded implementation of a
 * {@link de.ef.neuralnetworks.NeuralNetwork NeuralNetwork}.
 * <p>
 * In this implementation every neuron of a layer (the first layer excluded)
 * is connected with every neuron from the previous layer. Every connection
 * has some weight to it and the backpropagation algorithm used changes
 * this weight to train and adjust the network.
 * </p>
 * 
 * @author Erik Fritzsche
 * @version 1.0
 * @since 1.0
 */
public class SlowWave
	implements NeuralNetwork{
	
	/**
	 * Make always same as @version in JavaDoc in format xxxx.yyyy.zzzz
	 */
	private static final long serialVersionUID = 0001_0000_0000L;
	
	
	/**
	 * The default learning rate for the used backpropagation training algorithm.
	 */
	public final static double DEFAULT_LEARNING_RATE = 1.0;
	
	
	
	private final Neuron layers[][];
	
	
	/**
	 * Constructs a neural-network with {@code hiddenSizes.length + 2} layers.
	 * The size of each layer is defined by the according parameter.
	 * 
	 * @param inputSize size of first layer
	 * @param hiddenSizes sizes of the the layers between the first and the last
	 * @param outputSize size of the last layer
	 */
	public SlowWave(int inputSize, int hiddenSizes[], int outputSize){
		this.layers = new Neuron[2 + hiddenSizes.length][];
		
		// create the input layer
		this.layers[0] = new Neuron[inputSize];
		// create each hidden layer
		for(int i = 0; i < hiddenSizes.length; i++){
			this.layers[i + 1] = new Neuron[hiddenSizes[i]];
		}
		// create output layer
		this.layers[this.layers.length - 1] = new Neuron[outputSize];
		// init the neuron layers (except the input layer)
		for(int i = 1; i < this.layers.length; i++){
			// init each neuron
			for(int j = 0; j < this.layers[i].length; j++){
				// init all weights including bias neuron
				// (see documentation for details)
				this.layers[i][j] =
					new Neuron(new double[this.layers[i - 1].length + 1]);
				for(int k = 0; k < this.layers[i - 1].length + 1; k++){
					// (1 - (Math.random() * 2)) element [-1; 1[
					this.layers[i][j].setWeight(k, 1 - (Math.random() * 2));
				}
			}
		}
		// init input layer
		for(int i = 0; i < this.layers[0].length; i++){
			this.layers[0][i] = new Neuron(new double[0]);
		}
	}
	
	
	@Override
	public double[] calculate(double inputs[]){
		// set input neurons
		for(int i = 0; i < this.layers[0].length; i++){
			this.layers[0][i].setOutput(inputs[i]);
		}
		// run through each layer (except input)
		for(int i = 1; i < this.layers.length; i++){
			// run through each neuron in current layer
			for(int j = 0; j < this.layers[i].length; j++){
				// calculate input sum for neuron
				double sum = 0;
				for(int k = 0; k < this.layers[i - 1].length; k++){
					sum +=
						this.layers[i - 1][k].getOutput()
						* this.layers[i][j].getWeight(k);
				}
				// bias neuron weight
				sum += this.layers[i][j].getWeight(this.layers[i - 1].length);
				// calculate output for neuron via sigmoid function
				// (see documentation for details)
				this.layers[i][j].setOutput(1 / (1 + Math.pow(Math.E, -sum)));
			}
		}
		// grab and return outputs from last layer
		double outputs[] =
			new double[this.layers[this.layers.length - 1].length];
		for(int i = 0; i < outputs.length; i++){
			outputs[i] = this.layers[this.layers.length - 1][i].getOutput();
		}
		return outputs;
	}
	
	
	// just call train with default learning rate
	@Override
	public double train(double inputs[], double outputs[]){
		return this.train(inputs, outputs, DEFAULT_LEARNING_RATE);
	}
	
	@Override
	public double train(double inputs[], double outputs[], double learningRate){
		// update neural network to get current output
		this.calculate(inputs);
		// run through each layer (except input), reversed order
		for(int i = this.layers.length - 1; i > 0; i--){
			// run through each neuron in current layer and calculate error
			for(int j = 0; j < this.layers[i].length; j++){
				double lastError = 0;
				// if the current layer is the output layer
				// then the last error is the expected output
				// minus the real output of the current neuron
				if(i == this.layers.length - 1){
					lastError = outputs[j] - this.layers[i][j].getOutput();
				}
				// if not then calculate the last error form
				// the next layer errors
				else{
					for(int k = 0; k < this.layers[i + 1].length; k++){
						lastError +=
							this.layers[i + 1][k].getError()
							* this.layers[i + 1][k].getWeight(j);
					}
				}
				// set error for current neuron
				this.layers[i][j].setError(
					this.layers[i][j].getOutput()
					* (1 - this.layers[i][j].getOutput())
					* lastError
				);
				// adjust weights for current neuron
				for(int k = 0; k < this.layers[i - 1].length; k++){
					this.layers[i][j].setWeight(k,
						this.layers[i][j].getWeight(k)
						+ (
							learningRate
							* this.layers[i][j].getError()
							* this.layers[i - 1][k].getOutput()
						)
					);
				}
				// adjust bias neuron weight
				this.layers[i][j].setWeight(this.layers[i - 1].length,
					this.layers[i][j].getWeight(this.layers[i - 1].length)
					+ (learningRate * this.layers[i][j].getError())
				);
			}
		}
		// calculate and return total error
		double totalError = 0;
		for(Neuron output : this.layers[this.layers.length - 1]){
			totalError += Math.abs(output.getError());
		}
		return totalError;
	}
	
	
	// serialization
	private void writeObject(ObjectOutputStream output) throws IOException{
		SlowWaveSerialization.write(this, output);
	}
	
	private void readObject(ObjectInputStream input) throws IOException{
		SlowWaveSerialization.read(this, input);
	}
	
	
	Neuron[][] getLayers(){
		return this.layers;
	}
	
	
	
	/**
	 * The class {@code Neuron} is basically a container for
	 * an output value, an error value and the weights of the
	 * connections to the previous layer with according
	 * getter and setter methods.
	 * 
	 * @author Erik Fritzsche
	 * @version 1.0
	 * @since 1.0
	 */
	class Neuron{
		
		private double output, error, weights[];
		
		
		Neuron(double weights[]){
			this.weights = weights;
		}
		
		
		void setOutput(double value){
			this.output = value;
		}
		
		double getOutput(){
			return this.output;
		}
		
		
		void setError(double value){
			this.error = value;
		}
		
		double getError(){
			return this.error;
		}
		
		
		void setWeight(int index, double value){
			this.weights[index] = value;
		}
		
		double getWeight(int index){
			return weights[index];
		}
	}
}