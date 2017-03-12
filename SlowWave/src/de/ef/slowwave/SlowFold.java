package de.ef.slowwave;

import java.util.Map;

import de.ef.neuralnetworks.ConvolutionalNeuralNetwork;

public class SlowFold
	implements ConvolutionalNeuralNetwork<float[], double[], double[]>{
	
	
	int inputWidth, inputHeight, inputSize, inputDepth;
	int filterWidth, filterHeight, filterSize, filterWidthPadding, filterHeightPadding;
	float filters[][][];
	boolean poolDownSample;
	
	SlowWave fullyConnected;
	
	
	public SlowFold(){}
	
	
	@Override
	public void init(int inputSize, int hiddenSizes[], int outputSize, Map<String, Object> properties){
		this.fullyConnected = new SlowWave();
		this.fullyConnected.init(inputSize, hiddenSizes, outputSize, properties);
		
		// TODO hardcoded
		this.filterWidth = (int)properties.getOrDefault("filters.width", 5); // TODO throw exception if width is even
		this.filterHeight = (int)properties.getOrDefault("filters.height", 5); // TODO throw exception if height is even
		this.filterSize = this.filterWidth * this.filterHeight;
		
		this.filterWidthPadding = (this.filterWidth - 1) / 2;
		this.filterHeightPadding = (this.filterHeight - 1) / 2;
		this.filters = new float[(int)properties.get("filters.layers.count")][][];
		
		// setup input dimensions
		this.inputDepth = (int)properties.getOrDefault("input.depth", 1);
		this.inputWidth = (int)properties.getOrDefault("input.width", inputSize);
		this.inputHeight = inputSize / (this.inputDepth * this.inputWidth);
		this.inputSize = this.inputWidth * this.inputHeight;
		
		// init first layer
		if(this.filters.length > 0){
			this.filters[0] = new float[(int)properties.get("filters.layers.0")][];
			for(int i = 0; i < this.filters[0].length; i++){
				this.filters[0][i] = new float[this.filterSize * this.inputDepth];
			}
		}
		// init layers and filters after the first layer
		for(int i = 1; i < this.filters.length; i++){
			this.filters[i] = new float[(int)properties.get("filters.layers." + i)][];
			for(int j = 0; j < this.filters[i].length; j++){
				this.filters[i][j] = new float[this.filterSize * this.filters[i - 1].length];
				for(int k = 0; k < this.filters[i][j].length; k++){
					this.filters[i][j][k] = (float)Math.random(); // TODO negative values
				}
			}
		}
		
		// other options
		this.poolDownSample = (boolean)properties.getOrDefault("filters.pool.last-layer", false);
	}
	
	
	@Override
	public double[] calculateFilters(float inputs[]){
		int activationIndex;
		float activationMap[];
		
		int depth = this.inputDepth;
		for(int i = 0; i < this.filters.length; i++){
			activationIndex = 0;
			activationMap = new float[this.inputSize * this.filters[i].length];
			
			for(int j = 0; j < this.filters[i].length; j++){
				for(int y = 0; y < this.inputHeight; y++){
					for(int x = 0; x < this.inputWidth; x++){
						float sum = 0;
						for(int filterY = 0; filterY < this.filterHeight; filterY++){
							for(int filterX = 0; filterX < this.filterWidth; filterX++){
								int realX = x + filterX - this.filterWidthPadding, realY = y + filterY - this.filterHeightPadding;
								
								if(realX >= 0 && realX < this.inputWidth && realY >= 0 && realY < this.inputHeight){
									for(int k = 0; k < depth; k++){
										sum +=
											inputs[(k * this.inputSize) + (realX + realY * this.inputWidth)]
											* this.filters[i][j][(k * this.filterSize) + (filterX + filterY * this.filterWidth)];
									}
								}
							}
						}
						// directly apply rectified linear units to activation map
						activationMap[activationIndex++] = Math.max(sum, 0);
					}
				}
			}
			
			inputs = activationMap;
			depth = this.filters[i].length;
		}
		
		double outputs[];
		if(this.poolDownSample == true){
			// use maximum pooling
			int halfWidth = this.inputWidth / 2, halfHeight = this.inputHeight / 2;
			outputs = new double[halfWidth * halfHeight * depth];
			
			for(int d = 0, offset = 0, index = 0; d < depth; d++, offset += this.inputSize){
				for(int y = 0; y + 1 < this.inputHeight; y += 2){
					for(int x = 0; x + 1 < this.inputWidth; x += 2, index++){
						outputs[index] = (double)Math.max(
							inputs[offset + (x + y * this.inputWidth)],
							Math.max(
								inputs[offset + ((x + 1) + y * this.inputWidth)],
								Math.max(
									inputs[offset + (x + (y + 1) * this.inputWidth)],
									inputs[offset + ((x + 1) + (y + 1) * this.inputWidth)]
								)
							)
						);
					}
				}
			}
		}
		else{
			outputs = new double[this.inputSize * depth];
			for(int i = 0; i < outputs.length; i++){
				outputs[i] = (double)inputs[i];
			}
		}
		
		return outputs;
	}
	
	@Override
	public double[] calculateFullyConnected(double inputs[]){
		return this.fullyConnected.calculate(inputs);
	}
	
	
	@Override
	public double train(float inputs[], double outputs[]){
		// forward pass filters
		double filterForward[] = this.calculateFilters(inputs);
		
		// train fully connected layers with filter forward pass
		double totalError = this.fullyConnected.train(filterForward, outputs);
		
		// train filters
		// FIXME train filters
		
		// use total error of fully connected layers as total error of this neural-network
		return totalError;
	}
}
