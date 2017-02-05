package de.ef.fastflood;

import java.io.IOException;

import de.ef.neuralnetworks.NeuralNetwork;

/**
 * {@code FastFlood} is a template for a native implementation off
 * {@link de.ef.neuralnetworks.NeuralNetwork NeuralNetwork}.
 * <p>
 * Because a native implementation will probably run on an GPU
 * single precision floating point numbers are used.
 * </p>
 * 
 * @author Erik Fritzsche
 * @version 2.0
 * @since 1.0
 */
public abstract class FastFlood
	implements NeuralNetwork<float[], float[]>, AutoCloseable{

	/**
	 * Make always same as @version in JavaDoc in format xxx.yyy.zzz
	 */
	private final static long serialVersionUID = 001_000_000L;
	
	
	/**
	 * The default learning rate for the used backpropagation training algorithm.
	 */
	public final static double DEFAULT_LEARNING_RATE = 1.0;
	
	
	
	protected final int neuronCounts[], neuronOffsets[];
	protected final float inputs[], outputs[], weights[];
	
	
	protected FastFlood(int inputSize, int hiddenSizes[], int outputSize){
		neuronCounts = new int[2 + hiddenSizes.length];
		
		int lastNeuronCount = inputSize;
		int totalNeuronCount = inputSize, totalWeightCount = 0;
		
		neuronCounts[0] = inputSize;
		for(int i = 1; i < hiddenSizes.length; i++){
			neuronCounts[i] = hiddenSizes[i];
			totalNeuronCount += hiddenSizes[i];
			// plus one for bias neuron
			totalWeightCount += hiddenSizes[i] * (lastNeuronCount + 1);
			lastNeuronCount = hiddenSizes[i];
		}
		neuronCounts[neuronCounts.length - 1] = outputSize;
		totalNeuronCount += outputSize;
		totalWeightCount += outputSize * (lastNeuronCount + 1);
		
		// init the inputs, outputs and weights arrays
		inputs = new float[neuronCounts[0]];
		outputs = new float[neuronCounts[neuronCounts.length - 1]];
		weights = new float[totalWeightCount];
		
		// now init the neuron offsets
		neuronOffsets = new int[totalNeuronCount];
		
		// first off set is zero
		neuronOffsets[0] = 0;
		// loop through each neuron, skip input layer because every neuron has zero weights
		for(int i = 1, index = 0, layer = 1; i < totalNeuronCount; i++, index++){
			// neuron offset is last offset plus last layer length plus one for the bias weight
			neuronOffsets[i] = neuronOffsets[i - 1] + neuronCounts[layer - 1] + 1;
			if(index == neuronCounts[layer]){
				layer++;
				index = 0;
			}
		}
		
		for(int i = 0; i < weights.length; i++){
			// (1 - (Math.random() * 2)) element [-1; 1[
			weights[i] = (float)(1 - (Math.random() * 2));
		}
	}
	
	
	@Override
	public float[] calculate(float inputs[]) throws IOException{
		for(int i = 0; i < neuronCounts[0]; i++){
			inputs[i] = (float)inputs[i];
		}
		
		this.writeInputs();
		for(int i = 1; i < neuronCounts.length; i++){
			this.calculateLayer(i);
		}
		this.readOutputs();
		
		return outputs;
	}
	
	protected abstract void calculateLayer(int layer) throws IOException;
	
	
	// just call train with default learning rate
	@Override
	public double train(float inputs[], float outputs[]) throws IOException{
		return this.train(inputs, outputs, DEFAULT_LEARNING_RATE);
	}
	
	@Override
	public double train(float inputs[], float expectedOutputs[], double learningRate) throws IOException{
		// update neural network to get current output
		this.calculate(inputs);
		// run through each layer (except input), reversed order
		for(int i = neuronCounts.length - 1; i > 0; i--)
			this.trainLayer(i);
		
		return this.calculateError(expectedOutputs);
	}
	
	protected abstract void trainLayer(int layer) throws IOException;
	
	protected double calculateError(float expectedOutputs[]) throws IOException{
		// calculate and return total error
		double totalError = 0;
		for(int i = 0; i < outputs.length; i++){
			totalError += Math.abs(outputs[i] - expectedOutputs[i]);
		}
		return totalError;
	}
	
	
	protected abstract void writeInputs() throws IOException;
	
	protected abstract void readOutputs() throws IOException;
	
	
	public abstract void close() throws IOException;
}
