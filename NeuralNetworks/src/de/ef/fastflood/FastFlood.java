package de.ef.fastflood;

import java.io.IOException;

import de.ef.neuralnetworks.NeuralNetwork;

// TODO comment/document
public abstract class FastFlood
	implements NeuralNetwork, AutoCloseable{

	/**
	 * Make always same as @version in JavaDoc in format xxxx.yyyy.zzzz
	 */
	private static final long serialVersionUID = 0001_0000_0000L;
	
	
	/**
	 * The default learning rate for the used backpropagation training algorithm.
	 */
	public final static double DEFAULT_LEARNING_RATE = 1.0;
	
	
	
	protected final int neuronCounts[], layerOffsets[], neuronOffsets[];
	protected final float neurons[], weights[];
	
	
	protected FastFlood(int inputSize, int hiddenSizes[], int outputSize){
		this.neuronCounts = new int[2 + hiddenSizes.length];
		this.layerOffsets = new int[2 + hiddenSizes.length];
		
		int lastNeuronCount = inputSize;
		int totalNeuronCount = inputSize, totalWeightCount = 0;
		
		this.neuronCounts[0] = inputSize;
		this.layerOffsets[0] = 0;
		for(int i = 1; i < hiddenSizes.length; i++){
			this.neuronCounts[i] = hiddenSizes[i];
			this.layerOffsets[i] = totalNeuronCount;
			totalNeuronCount += hiddenSizes[i];
			// plus one for bias neuron
			totalWeightCount += hiddenSizes[i] * (lastNeuronCount + 1);
			lastNeuronCount = hiddenSizes[i];
		}
		this.neuronCounts[this.neuronCounts.length - 1] = outputSize;
		this.layerOffsets[this.neuronCounts.length - 1] = totalNeuronCount;
		totalNeuronCount += outputSize;
		totalWeightCount += outputSize * (lastNeuronCount + 1);
		
		// init the neurons and weights arrays
		this.neurons = new float[totalNeuronCount];
		this.weights = new float[totalWeightCount];
		
		// now init the neuron offsets
		this.neuronOffsets = new int[this.neurons.length];
		
		// first off set is zero
		this.neuronOffsets[0] = 0;
		// loop through each neuron, skip input layer because every neuron has zero weights
		for(int i = 1, index = 0, layer = 1; i < this.neurons.length; i++, index++){
			// neuron offset is last offset plus last layer length plus one for the bias weight
			this.neuronOffsets[i] = this.neuronOffsets[i - 1] + neuronCounts[layer - 1] + 1;
			if(index == neuronCounts[layer]){
				layer++;
				index = 0;
			}
		}
		
		for(int i = 0; i < this.weights.length; i++){
			// (1 - (Math.random() * 2)) element [-1; 1[
			this.weights[i] = (float)(1 - (Math.random() * 2));
		}
	}
	
	
	@Override
	public double[] calculate(double inputs[]) throws IOException{
		for(int i = 0; i < this.neuronCounts[0]; i++){
			this.neurons[i] = (float)inputs[i];
		}
		
		this.sync();
		for(int i = 1; i < this.neuronCounts.length; i++){
			this.calculateLayer(i);
		}
		this.read();
		
		double[] outputs = new double[this.neuronCounts[this.neuronCounts.length - 1]];
		int outputOffset = this.neuronOffsets[this.neuronCounts.length - 1];
		for(int i = 0; i < outputs.length; i++){
			outputs[i] = this.neurons[outputOffset + i];
		}
		
		return outputs;
	}
	
	protected abstract void calculateLayer(int layer) throws IOException;
	
	
	// just call train with default learning rate
	@Override
	public double train(double inputs[], double outputs[]) throws IOException{
		return this.train(inputs, outputs, DEFAULT_LEARNING_RATE);
	}
	
	@Override
	public double train(double inputs[], double outputs[], double learningRate) throws IOException{
		// TODO train fast-flood
		return 0;
	}
	
	
	protected abstract void sync() throws IOException;
	
	protected abstract void read() throws IOException;
}