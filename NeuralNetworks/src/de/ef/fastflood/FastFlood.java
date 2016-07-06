package de.ef.fastflood;

import de.ef.fastflood.opencl.OpenCLDevices;
import de.ef.neuralnetworks.NeuralNetwork;
import de.ef.neuralnetworks.NeuralNetworkConstructor;

// TODO comment
// neural network core class, implements the NeuralNetworkAPI
// version: 1.2, date: 16.06.2016, author: Erik Fritzsche
public abstract class FastFlood
	implements NeuralNetwork, AutoCloseable{
	
	public final static NeuralNetworkConstructor OPEN_CL[] = OpenCLDevices.CONSTRUCTORS;
	
	public final static double DEFAULT_LEARNING_RATE = 1;
	
	
	
	protected final int layerCount, inputCount, neuronCounts[], layerOffsets[], neuronOffsets[];
	protected final float inputs[], neurons[], weights[];
	
	
	protected FastFlood(int inputSize, int hiddenSizes[], int outputSize){
		this.layerCount = 2 + hiddenSizes.length;
		this.inputCount = inputSize;
		this.neuronCounts = new int[this.layerCount - 1];
		this.layerOffsets = new int[this.layerCount - 1];
		
		int currentLayerOffset = 0, totalWeightCount = 0;
		int lastLayerSize = this.inputCount;
		
		for(int i = 0; i < hiddenSizes.length; i++){
			this.neuronCounts[i] = hiddenSizes[i];
			this.layerOffsets[i] = currentLayerOffset;
			currentLayerOffset += hiddenSizes[i];
			// plus one for bias neuron
			totalWeightCount += hiddenSizes[i] * (lastLayerSize + 1);
			lastLayerSize = hiddenSizes[i];
		}
		this.neuronCounts[this.layerCount - 2] = outputSize;
		this.layerOffsets[this.layerCount - 2] = currentLayerOffset;
		totalWeightCount += outputSize * (lastLayerSize + 1);
		
		this.inputs = new float[this.inputCount];
		this.neurons = new float[currentLayerOffset + outputSize];
		this.weights = new float[totalWeightCount];
		
		// now init the neuron-offsets
		this.neuronOffsets = new int[this.neurons.length];
		
		// reset last layer
		lastLayerSize = this.inputCount;
		
		// first off set is zero
		this.neuronOffsets[0] = 0;
		// loop through each neuron (but not the inputs)
		for(int i = 1, index = 0, layer = 0; i < this.neurons.length; i++, index++){
			// neuron offset is last offset plus last layer length plus one for the bias weight
			this.neuronOffsets[i] = this.neuronOffsets[i - 1] + lastLayerSize + 1;
			// change the last layer size when the end off the current layer is reached
			if(index == this.neuronCounts[layer]){
				lastLayerSize = this.neuronCounts[layer];
				index = 0;
				layer++;
			}
		}
		
		for(int i = 0; i < this.weights.length; i++){
			// (1 - (Math.random() * 2)) element [-1; 1[
			this.weights[i] = (float)(1 - (Math.random() * 2));
		}
		
		this.loadStaticResources();
	}
	
	
	// TODO arraycopy will not work -> for loop
	// calculate the output of the neural network for the given input
	@Override
	public double[] calculate(double inputs[]) throws RuntimeException{
		this.loadResources();
		System.arraycopy(inputs, 0, this.inputs, 0, this.inputs.length);
		for(int i = 1; i < this.layerCount; i++){
			this.calculateLayer(i);
		}
		this.syncResources();
		double[] outputs = new double[this.neuronCounts[this.layerCount - 2]];
		System.arraycopy(this.neurons, this.layerOffsets[this.layerCount - 2], outputs, 0, outputs.length);
		this.releaseResources();
		
		return outputs;
	}
	
	
	// just calls train with default learning rate
	@Override
	public double train(double inputs[], double outputs[]){
		return this.train(inputs, outputs, DEFAULT_LEARNING_RATE);
	}
	
	// lets the neural network adjust itself (aka training)
	// by using backpropagation and returns the total
	// network error (before adjustments are made)
	@Override
	public double train(double inputs[], double outputs[], double learningRate){
		// TODO
		return 0;
	}
	
	
	public void close() throws RuntimeException{
		this.releaseStaticResources();
	}
	
	
	protected abstract void calculateLayer(int layer) throws RuntimeException;
	
	
	protected abstract void loadResources() throws RuntimeException;
	
	protected abstract void releaseResources() throws RuntimeException;
	
	protected abstract void syncResources() throws RuntimeException;
	
	protected abstract void loadStaticResources() throws RuntimeException;
	
	protected abstract void releaseStaticResources() throws RuntimeException;
	
	
	public boolean hasBias(){
		return true;
	}
	
	public int getLayerCount(){
		return this.layerCount;
	}
	
	public int getNeuronCount(int layer){
		// input layer is an exception
		return layer == 0 ? this.inputCount : this.neuronCounts[layer - 1];
	}
	
	public double getWeight(int layer, int neuron, int weight){
		// input layer has no weights -> no special treatment
		return (double)this.weights[this.neuronOffsets[this.layerOffsets[layer] + neuron] + weight];
	}
	
	public void setWeight(int layer, int neuron, int weight, double value){
		// input layer has no weights -> no special treatment
		this.weights[this.neuronOffsets[this.layerOffsets[layer] + neuron] + weight] = (float)value;
	}
}