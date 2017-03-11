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
	
	
	
	protected final int layerSizes[], layerOffsets[], neuronOffsets[], weightCount;
	protected float inputs[], outputs[];
	
	
	protected FastFlood(int inputSize, int hiddenSizes[], int outputSize){
		layerSizes = new int[2 + hiddenSizes.length];
		layerOffsets = new int[2 + hiddenSizes.length];
		
		int lastNeuronCount = inputSize;
		int totalNeuronCount = 0, totalWeightCount = 0;
		
		layerSizes[0] = inputSize;
		layerOffsets[0] = 0;
		for(int i = 0; i < hiddenSizes.length; i++){
			layerSizes[i + 1] = hiddenSizes[i];
			// layer offset current total neuron count
			layerOffsets[i + 1] = totalNeuronCount;
			totalNeuronCount += hiddenSizes[i];
			// plus one for bias neuron
			totalWeightCount += hiddenSizes[i] * (lastNeuronCount + 1);
			lastNeuronCount = hiddenSizes[i];
		}
		layerSizes[layerSizes.length - 1] = outputSize;
		layerOffsets[layerSizes.length - 1] = totalNeuronCount;
		totalNeuronCount += outputSize;
		totalWeightCount += outputSize * (lastNeuronCount + 1);
		
		// init the outputs and weights arrays
		outputs = new float[outputSize];
		weightCount = totalWeightCount;
		
		// now init the neuron offsets, without input neurons
		neuronOffsets = new int[totalNeuronCount];
		
		// first offset is zero
		neuronOffsets[0] = 0;
		// loop through each neuron, skip input layer because every neuron has zero weights
		for(int i = 1, index = 0, layer = 1; i < neuronOffsets.length; i++, index++){
			// neuron offset is last offset plus last layer length plus one for the bias weight
			neuronOffsets[i] = neuronOffsets[i - 1] + layerSizes[layer - 1] + 1;
			if(index == layerSizes[layer]){
				layer++;
				index = 0;
			}
		}
	}
	
	
	@Override
	public float[] calculate(float inputs[]) throws IOException{
		if(inputs.length < layerSizes[0])
			throw new ArrayIndexOutOfBoundsException("Input to small");
		this.inputs = inputs;
		
		this.writeInputs();
		for(int i = 1; i < layerSizes.length; i++){
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
		for(int i = layerSizes.length - 1; i > 0; i--)
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
