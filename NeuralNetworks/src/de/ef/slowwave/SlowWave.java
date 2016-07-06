package de.ef.slowwave;

import de.ef.neuralnetworks.NeuralNetwork;
import de.ef.neuralnetworks.NeuralNetworkConstructor;

// neural network core class, implements the NeuralNetworkAPI
// version: 1.5, date: 14.06.2016, author: Erik Fritzsche
// 
// for details on all mathematical formulas, equations and
// a general overview please look into the documentation
public class SlowWave
	implements NeuralNetwork{
	
	public final static NeuralNetworkConstructor CONSTRUCTOR = (i, h, o) -> new SlowWave(i, h, o);
	
	public final static double DEFAULT_LEARNING_RATE = 1;
	
	
	
	private final Neuron layers[][];
	private final int layerCount;
	
	
	public SlowWave(int inputSize, int hiddenSizes[], int outputSize){
		this.layerCount = 2 + hiddenSizes.length;
		this.layers = new Neuron[this.layerCount][];
		
		// create the input layer
		this.layers[0] = new Neuron[inputSize];
		// create each hidden layer
		for(int i = 0; i < hiddenSizes.length; i++){
			this.layers[i + 1] = new Neuron[hiddenSizes[i]];
		}
		// create output layer
		this.layers[this.layerCount - 1] = new Neuron[outputSize];
		// init the neuron layers (except the input layer)
		for(int i = 1; i < this.layerCount; i++){
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
	
	
	// calculate the output of the neural network for the given input
	@Override
	public double[] calculate(double inputs[]){
		// set input neurons
		for(int i = 0; i < this.layers[0].length; i++){
			this.layers[0][i].setOutput(inputs[i]);
		}
		// run through each layer (except input)
		for(int i = 1; i < this.layerCount; i++){
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
			new double[this.layers[this.layerCount - 1].length];
		for(int i = 0; i < outputs.length; i++){
			outputs[i] = this.layers[this.layerCount - 1][i].getOutput();
		}
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
		// updates neural network to get current output
		this.calculate(inputs);
		// run through each layer (except input), reverse order
		for(int i = this.layerCount - 1; i > 0; i--){
			// run through each neuron in current layer and calculate error
			for(int j = 0; j < this.layers[i].length; j++){
				double lastError = 0;
				// if the current layer is the output layer
				// then the last error is the expected output
				// minus the real output of the current neuron
				if(i == this.layerCount - 1){
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
		for(Neuron output : this.layers[this.layerCount - 1]){
			totalError += Math.abs(output.getError());
		}
		return totalError;
	}
	
	
	// TODO comment
	public boolean hasBias(){
		return true;
	}
	
	public int getLayerCount(){
		return this.layerCount;
	}
	
	public int getNeuronCount(int layer){
		return this.layers[layer].length;
	}
	
	public double getWeight(int layer, int neuron, int weight){
		return this.layers[layer][neuron].getWeight(weight);
	}
	
	public void setWeight(int layer, int neuron, int weight, double value){
		this.layers[layer][neuron].setWeight(weight, value);
	}
	
	
	
	// neuron class, basically a container for an output value,
	// an error value and the weights of the connections to the
	// previous layer and according getter and setter methods
	private class Neuron{
		
		private double output, error, weights[];
		
		
		public Neuron(double[] weights){
			this.weights = weights;
		}
		
		
		public void setOutput(double value){
			this.output = value;
		}
		
		public double getOutput(){
			return this.output;
		}
		
		
		public void setError(double value){
			this.error = value;
		}
		
		public double getError(){
			return this.error;
		}
		
		
		public void setWeight(int index, double value){
			this.weights[index] = value;
		}
		
		public double getWeight(int index){
			return weights[index];
		}
	}
}