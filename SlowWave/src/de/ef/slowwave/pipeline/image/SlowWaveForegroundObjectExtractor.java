package de.ef.slowwave.pipeline.image;

import de.ef.neuralnetworks.pipeline.BufferFactory;
import de.ef.neuralnetworks.pipeline.image.ForegroundObjectExtractor;
import de.ef.neuralnetworks.pipeline.image.GrayscaleImage;

public class SlowWaveForegroundObjectExtractor
	extends ForegroundObjectExtractor{
	
	public SlowWaveForegroundObjectExtractor(
			int outputWidth, int outputHeight, int backgroundThreshold,
			BufferFactory<byte[]> bufferFactory){
		super(outputWidth, outputHeight, backgroundThreshold, bufferFactory);
	}
	
	
	@Override
	protected void extractObject(
			int x, int y, int width, int height, int virtualWidth, int virtualHeight,
			int chunkWidth, int chunkHeight, GrayscaleImage input, GrayscaleImage output){
		
		// max object size is 2^15 (~32k) in both dimensions, should be enough
		// multiply everything by 65536 to avoid floating point calculations while keeping some precision
		int mapRatioX = (width << 16) / virtualWidth + 1;
		int mapRatioY = (height << 16) / virtualHeight + 1;
		
		int chunkSize = chunkWidth * chunkHeight;
		int outputIndex = 0, offsetX = x, offsetY = y, virtualX, virtualY = 0;
		for(y = 0; y < output.height; y++, virtualY += chunkHeight){
			for(x = 0, virtualX = 0; x < output.width; x++, virtualX += chunkWidth, outputIndex++){
				int count = 0;
				for(int chunkY = 0; chunkY < chunkHeight; chunkY++){
					// map the current y coordinate (chunkX + virtualY) to its real coordinate and divide by 65536
					// then multiply to get y offset and add x offset (constant)
					int chunkOffset = ((((chunkY + virtualY) * mapRatioY) >>> 16) + offsetY) * input.width + offsetX;
					for(int chunkX = 0; chunkX < chunkWidth; chunkX++)
						// map current x coordinate to real x, divide by 65536 and add offset
						count += input.data[(((chunkX + virtualX) * mapRatioX) >>> 16) + chunkOffset] & 0xFF;
				}
				output.data[outputIndex] = (byte)(count / chunkSize);
			}
		}
	}
}
