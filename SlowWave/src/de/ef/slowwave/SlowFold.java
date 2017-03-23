package de.ef.slowwave;

import java.io.Serializable;
import java.util.Locale;
import java.util.Map;

import de.ef.neuralnetworks.ConvolutionalNeuralNetwork;

public class SlowFold
	implements ConvolutionalNeuralNetwork<float[], double[], double[]>{
	
	
	int inputWidth, inputHeight, inputSize, inputDepth;
	int filterWidth, filterHeight, filterSize, filterWidthPadding, filterHeightPadding;
	FilterLayer filterLayers[]; // TODO bias in filters
	private double filterOutputs[];
	
	SlowWave fullyConnected;
	
	
	public SlowFold(){}
	
	
	@Override
	public void init(int inputSize, int hiddenSizes[], int outputSize, Map<String, Object> properties){
		// TODO hardcoded
		this.filterWidth = (int)properties.getOrDefault("filters.width", 5); // TODO throw exception if width is even
		this.filterHeight = (int)properties.getOrDefault("filters.height", 5); // TODO throw exception if height is even
		this.filterSize = this.filterWidth * this.filterHeight;
		
		this.filterWidthPadding = (this.filterWidth - 1) / 2;
		this.filterHeightPadding = (this.filterHeight - 1) / 2;
		
		// setup input dimensions
		this.inputDepth = (int)properties.getOrDefault("input.depth", 1);
		this.inputWidth = (int)properties.getOrDefault("input.width", inputSize);
		this.inputHeight = inputSize / (this.inputDepth * this.inputWidth);
		this.inputSize = this.inputWidth * this.inputHeight;
		
		// init filter layers
		this.filterLayers = new FilterLayer[(int)properties.get("filters.layers.count")];
		
		int previousWidth = this.inputWidth, previousHeight = this.inputHeight, previousDepth = this.inputDepth;
		for(int i = 0; i < this.filterLayers.length; previousDepth = this.filterLayers[i++].filters.length){
			float filters[][] = new float[(int)properties.get("filters.layers." + i)][];
			for(int j = 0; j < filters.length; j++){
				filters[j] = new float[this.filterSize * previousDepth + 2];
				
				float filterMin = 0f, filterMax = 0f;
				for(int k = 0; k + 2 < filters[j].length; k++){
					float weight = (filters[j][k] = (float)(1 - (Math.random() * 2)));
					if(weight < 0)
						filterMin += weight;
					else
						filterMax += weight;
				}
				
				// set minimal and maximal output of filter (assuming an an input between 0 and 1)
				filters[j][filters[j].length - 2] = filterMin;
				filters[j][filters[j].length - 1] = filterMax;
			}
			
			Object poolingModeObject =
				properties.getOrDefault("filters.layers." + i + ".pooling.mode", PoolingMode.NONE);
			PoolingMode poolingMode =
				poolingModeObject instanceof PoolingMode
				? (PoolingMode)poolingModeObject
				: PoolingMode.valueOf(((String)poolingModeObject).toUpperCase(Locale.ENGLISH));
			
			this.filterLayers[i] =
				new FilterLayer(filters, poolingMode, previousWidth, previousHeight, previousDepth);
			
			previousWidth = this.filterLayers[i].width;
			previousHeight = this.filterLayers[i].height;
		}
		
		this.filterOutputs = new double[previousWidth * previousHeight * previousDepth]; // TODO handle pooling after last layer
		
		// init fully connected layers
		this.fullyConnected = new SlowWave();
		this.fullyConnected.init(this.filterOutputs.length, hiddenSizes, outputSize, properties);
	}
	
	
	@Override
	public double[] calculateFilters(float inputs[]){
		for(int i = 0; i < this.filterLayers.length; i++){
			inputs = this.filterLayers[i].calculate(inputs);
		}
		
		double outputs[] = this.filterOutputs;
		for(int i = 0; i < outputs.length; i++){
			outputs[i] = (double)inputs[i];
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
		
		// TODO handle pooling after last layer
		// calculate errors
		for(int i = this.filterLayers.length - 1; i >= 0; i--){
			float filters[][] = this.filterLayers[i].filters;
			
			if(i + 1 == this.filterLayers.length){
				for(int z = 0, index = 0; z < filters.length; z++){
					for(int y = 0; y < this.filterLayers[i].height; y++){
						for(int x = 0; x < this.filterLayers[i].width; x++, index++){
							float errorSum = 0;
							for(int k = 0; k < this.fullyConnected.layers[1].length; k++){
								errorSum +=
									this.fullyConnected.layers[1][k].getError()
									* this.fullyConnected.layers[1][k].getWeight(index);
							}
							
							this.filterLayers[i].errors[index] = (
								this.filterLayers[i].outputs[index]
								* (1 - this.filterLayers[i].outputs[index])
								* errorSum
							);
						}
					}
				}
			}
			else{
				this.filterLayers[i].calculateErrors(
					this.filterLayers[i + 1].errors, this.filterLayers[i + 1].filters,
					this.filterLayers[i + 1].width, this.filterLayers[i + 1].height
				);
			}
		}
		
		// adjust weights after error calculation, the order is not important
		float previousOutputs[] = inputs;
		int previousDepth = this.inputDepth;
		for(int i = 0; i < this.filterLayers.length; i++){
			this.filterLayers[i].train(previousOutputs, previousDepth);
			
			previousOutputs = this.filterLayers[i].outputs;
			previousDepth = this.filterLayers[i].filters.length;
		}
		
		// use total error of fully connected layers as total error of this neural-network
		return totalError;
	}
	
	
	
	public static enum PoolingMode{
		
		NONE(SlowFoldPoolingModes::directOutputSum, SlowFoldPoolingModes::directOutputErrorSum),
		AVERAGE(SlowFoldPoolingModes::averageOutputSum, SlowFoldPoolingModes::averageOutputErrorSum),
		MAXIMUM(SlowFoldPoolingModes::maximumOutputSum, SlowFoldPoolingModes::maximumOutputErrorSum);
		
		
		
		private final OutputSumCalculator outputSum;
		private final ErrorSumCalculator errorSum;
		
		
		private PoolingMode(OutputSumCalculator outputSum, ErrorSumCalculator errorSum){
			this.outputSum = outputSum;
			this.errorSum = errorSum;
		}
		
		
		
		private interface OutputSumCalculator{
			
			public float calculate(
					float inputs[], float filter[],
					int lowerBoundX, int upperBoundX, int lowerBoundY, int upperBoundY,
					int x, int y, int width, int height, int size,
					int filterWidth, int filterHeight, int filterSize, int filterDepth,
					int filterWidthPadding, int filterHeightPadding);
		}
		
		private interface ErrorSumCalculator{
			
			public float calculate(
					float inputs[], float errors[], int layerOffset,
					int lowerBoundX, int upperBoundX, int lowerBoundY, int upperBoundY,
					int offsetX, int offsetY, int offsetZ, int width, int height, int size);
		}
	}
	
	
	private class FilterLayer
		implements Serializable{
		
		private final float filters[][], outputs[], errors[];
		private final int width, height, size, filterDepth;
		private final PoolingMode poolingMode;
		
		
		private FilterLayer(float filters[][], PoolingMode poolingMode, int width, int height, int filterDepth){
			this.filters = filters;
			
			this.poolingMode = poolingMode;
			
			if(this.poolingMode == PoolingMode.NONE){
				this.width = width;
				this.height = height;
			}
			else{
				// TODO throw exception if width or height is not divisible by 2
				this.width = width / 2;
				this.height = height / 2;
			}
			this.size = this.width * this.height;
			this.filterDepth = filterDepth;
			
			this.outputs = new float[this.width * this.height * this.filters.length];
			this.errors = new float[this.width * this.height * this.filters.length];
		}
		
		
		private float[] calculate(float inputs[]){
			for(int j = 0; j < this.filters.length; j++){
				for(int y = 0; y < this.height; y++){
					for(int x = 0; x < this.width; x++){
						int lowerBoundX = x - SlowFold.this.filterWidthPadding < 0 ? -(x - SlowFold.this.filterWidthPadding) : 0;
						int upperBoundX = SlowFold.this.filterWidth + (
							x + SlowFold.this.filterWidthPadding >= this.width
							? (this.width - (x + SlowFold.this.filterWidthPadding + 1))
							: 0
						);
						int lowerBoundY = y - SlowFold.this.filterHeightPadding < 0 ? -(y - SlowFold.this.filterHeightPadding) : 0;
						int upperBoundY = SlowFold.this.filterHeight + (
							y + SlowFold.this.filterHeightPadding >= this.height
							? (this.height - (y + SlowFold.this.filterHeightPadding + 1))
							: 0
						);
						
						float sum = this.poolingMode.outputSum.calculate(
							inputs, this.filters[j],
							lowerBoundX, upperBoundX, lowerBoundY, upperBoundY,
							x, y, this.width, this.height, this.size,
							SlowFold.this.filterWidth, SlowFold.this.filterHeight, SlowFold.this.filterSize, this.filterDepth,
							SlowFold.this.filterWidthPadding, SlowFold.this.filterHeightPadding
						);
						
						// map sum between filter minimum and maximum to an output between 0 and 1
						float filterMin = this.filters[j][this.filters[j].length - 2];
						float filterMax = this.filters[j][this.filters[j].length - 1];
						
						this.outputs[x + (y * this.width) + (j * this.size)] =
							((sum - filterMin) / (filterMax - filterMin));
					}
				}
			}
			
			return this.outputs;
		}
		
		
		private void calculateErrors(float nextErrors[], float nextFilters[][], final int nextWidth, final int nextHeight){
			final int nextSize = nextWidth * nextHeight;
			final boolean nextScaled = this.size != nextWidth;
			
			for(int i = 0, index = 0; i < this.filters.length; i++){
				for(int y = 0; y < this.height; y++){
					for(int x = 0; x < this.width; x++, index++){
						float errorSum = 0;
						
						int lowerBoundX = x - SlowFold.this.filterWidthPadding < 0 ? -(x - SlowFold.this.filterWidthPadding) : 0;
						int upperBoundX = SlowFold.this.filterWidth + (
							x + SlowFold.this.filterWidthPadding >= this.width
							? (this.width - (x + SlowFold.this.filterWidthPadding + 1))
							: 0
						);
						int lowerBoundY = y - SlowFold.this.filterHeightPadding < 0 ? -(y - SlowFold.this.filterHeightPadding) : 0;
						int upperBoundY = SlowFold.this.filterHeight + (
							y + SlowFold.this.filterHeightPadding >= this.height
							? (this.height - (y + SlowFold.this.filterHeightPadding + 1))
							: 0
						);
						
						for(int fY = lowerBoundY, rY = y + (fY - SlowFold.this.filterHeightPadding); fY < upperBoundY; fY++, rY++){
							for(int fX = lowerBoundX, rX = x + (fX - SlowFold.this.filterWidthPadding); fX < upperBoundX; fX++, rX++){
								for(int j = 0; j < nextFilters.length; j++){
									errorSum +=
										(
											nextScaled == true
											? nextErrors[(rX / 2) + ((rY / 2) * nextWidth) + (j * nextSize)]
											: nextErrors[rX + (rY * nextWidth) + (j * nextSize)]
										)
										* nextFilters[j][fX + (fY * SlowFold.this.filterWidth) + (j * SlowFold.this.filterSize)];
								}
							}
						}
						
						this.errors[index] = (this.outputs[index] * (1 - this.outputs[index]) * errorSum);
					}
				}
			}
		}
		
		private void train(float inputs[], int inputDepth){
			for(int i = 0; i < this.filters.length; i++){
				float filterMin = 0f, filterMax = 0f;
				for(int fY = 0, rfY = -SlowFold.this.filterHeightPadding; fY < SlowFold.this.filterHeight; fY++, rfY++){
					for(int fX = 0, rfX = -SlowFold.this.filterWidthPadding; fX < SlowFold.this.filterWidth; fX++, rfX++){
						int lowerBoundX = Math.max(rfX, 0);
						int upperBoundX = this.width + Math.min(rfX, 0);
						int lowerBoundY = Math.max(rfY, 0);
						int upperBoundY = this.height + Math.min(rfY, 0);
						
						for(int fZ = 0; fZ < inputDepth; fZ++){
							float weight = this.filters[i][fX + (fY * SlowFold.this.filterWidth) + (fZ * SlowFold.this.filterSize)] =
								(float)(
									this.filters[i][fX + (fY * SlowFold.this.filterWidth) + (fZ * SlowFold.this.filterSize)]
									+ (
										SlowFold.this.fullyConnected.learningRate
										* this.poolingMode.errorSum.calculate(
											inputs, this.errors, i * this.size,
											lowerBoundX, upperBoundX, lowerBoundY, upperBoundY,
											rfX, rfY, fZ, this.width, this.height, this.size
										)
									)
								);
							
							if(weight < 0)
								filterMin += weight;
							else
								filterMax += weight;
						}
					}
				}
				
				// update filter minimum and maximum
				this.filters[i][this.filters[i].length - 2] = filterMin;
				this.filters[i][this.filters[i].length - 1] = filterMax;
			}
		}
	}
}
