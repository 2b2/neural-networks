package de.ef.fastflood;

import java.io.IOException;

import de.ef.neuralnetworks.NeuralNetwork;

// TODO comment/document
public abstract class FastFlood
	implements NeuralNetwork, AutoCloseable{

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
		this.neuronCounts = new int[2 + hiddenSizes.length];
		
		int lastNeuronCount = inputSize;
		int totalNeuronCount = inputSize, totalWeightCount = 0;
		
		this.neuronCounts[0] = inputSize;
		for(int i = 1; i < hiddenSizes.length; i++){
			this.neuronCounts[i] = hiddenSizes[i];
			totalNeuronCount += hiddenSizes[i];
			// plus one for bias neuron
			totalWeightCount += hiddenSizes[i] * (lastNeuronCount + 1);
			lastNeuronCount = hiddenSizes[i];
		}
		this.neuronCounts[this.neuronCounts.length - 1] = outputSize;
		totalNeuronCount += outputSize;
		totalWeightCount += outputSize * (lastNeuronCount + 1);
		
		// init the inputs, outputs and weights arrays
		this.inputs = new float[this.neuronCounts[0]];
		this.outputs = new float[this.neuronCounts[this.neuronCounts.length - 1]];
		this.weights = new float[totalWeightCount];
		
		// now init the neuron offsets
		this.neuronOffsets = new int[totalNeuronCount];
		
		// first off set is zero
		this.neuronOffsets[0] = 0;
		// loop through each neuron, skip input layer because every neuron has zero weights
		for(int i = 1, index = 0, layer = 1; i < totalNeuronCount; i++, index++){
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
			this.inputs[i] = (float)inputs[i];
		}
		
		this.writeInputs();
		for(int i = 1; i < this.neuronCounts.length; i++){
			this.calculateLayer(i);
		}
		this.readOutputs();
		
		double[] outputs = new double[this.neuronCounts[this.neuronCounts.length - 1]];
		for(int i = 0; i < outputs.length; i++){
			outputs[i] = this.outputs[i];
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
	
	
	protected abstract void writeInputs() throws IOException;
	
	protected abstract void readOutputs() throws IOException;
}