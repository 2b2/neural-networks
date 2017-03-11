package de.ef.neuralnetworks.pipeline.image;

import java.util.function.Function;

import de.ef.neuralnetworks.pipeline.BufferFactory;

public abstract class ImageForegroundObjectExtractor
	implements Function<GrayscaleImage, GrayscaleImage>{
	
	private final int outputWidth, outputHeight, outputSize;
	protected final int backgroundThreshold;
	private final BufferFactory<byte[]> bufferFactory;
	
	
	public ImageForegroundObjectExtractor(
			int outputWidth, int outputHeight, int backgroundThreshold,
			BufferFactory<byte[]> bufferFactory){
		this.outputWidth = outputWidth;
		this.outputHeight = outputHeight;
		this.outputSize = outputWidth * outputHeight;
		
		this.backgroundThreshold = backgroundThreshold;
		
		this.bufferFactory = bufferFactory;
	}
	
	
	@Override
	public GrayscaleImage apply(GrayscaleImage input){
		// find object offsets
		int x = this.findSmallestX(input), y = this.findSmallestY(input);
		
		// test if any object was found
		if(x == -1 || y == -1)
			return new GrayscaleImage(
				this.outputWidth, this.outputHeight, this.bufferFactory.getBuffer(this.outputSize, true)
			);
		
		// find object dimensions
		int width = (this.findBiggestX(input) - x) + 1, height = (this.findBiggestY(input) - y) + 1;
		
		// adjust dimensions so that width and height are both evenly divisible by the output dimensions
		int virtualWidth = width + (this.outputWidth - (width % this.outputWidth));
		int virtualHeight = height + (this.outputHeight - (height % this.outputHeight));
		
		// calculate chunk dimensions
		int chunkWidth = virtualWidth / this.outputWidth, chunkHeight = virtualHeight / this.outputHeight;
		
		// extract object
		GrayscaleImage output =
			new GrayscaleImage(this.outputWidth, this.outputHeight, this.bufferFactory.getBuffer(this.outputSize, false));
		
		this.extractObject(x, y, width, height, virtualWidth, virtualHeight, chunkWidth, chunkHeight, input, output);
		
		return output;
	}
	

	protected int findSmallestX(GrayscaleImage input){
		for(int x = 0; x < input.width; x++)
			for(int y = 0; y < input.height; y++)
				if((input.data[x + y * input.width] & 0xFF) < this.backgroundThreshold)
					return x;
		return -1;
	}
	
	protected int findBiggestX(GrayscaleImage input){
		for(int x = input.width - 1; x >= 0; x--)
			for(int y = 0; y < input.height; y++)
				if((input.data[x + y * input.width] & 0xFF) < this.backgroundThreshold)
					return x;
		return -1;
	}
	
	protected int findSmallestY(GrayscaleImage input){
		for(int y = 0; y < input.height; y++)
			for(int x = 0; x < input.width; x++)
				if((input.data[x + y * input.width] & 0xFF) < this.backgroundThreshold)
					return y;
		return -1;
	}
	
	protected int findBiggestY(GrayscaleImage input){
		for(int y = input.height - 1; y >= 0; y--)
			for(int x = 0; x < input.width; x++)
				if((input.data[x + y * input.width] & 0xFF) < this.backgroundThreshold)
					return y;
		return -1;
	}
	
	
	protected abstract void extractObject(
			int x, int y, int width, int height,
			int virtualWidth, int virtualHeight,
			int chunkWidth, int chunkHeight,
			GrayscaleImage input, GrayscaleImage output);
}
