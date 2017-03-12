package de.ef.neuralnetworks.pipeline.image;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.util.function.Function;

import de.ef.neuralnetworks.pipeline.BufferFactory;

public final class GrayscaleImageConverter{
	
	private GrayscaleImageConverter(){}
	
	
	public static Function<BufferedImage, GrayscaleImage> fromBufferedImage(BufferFactory<byte[]> bufferFactory, boolean clone){
		return input -> {
			if(input.getType() == BufferedImage.TYPE_BYTE_GRAY){
				if(clone == true){
					int width = input.getWidth(), height = input.getHeight();
					GrayscaleImage output = new GrayscaleImage(
						width, height, bufferFactory.getBuffer(width * height, false)
					);
					System.arraycopy(
						((DataBufferByte)input.getRaster().getDataBuffer()).getData(), 0, output.data, 0, width * height
					);
					
					return output;
				}
				else return new GrayscaleImage(
					input.getWidth(), input.getHeight(), ((DataBufferByte)input.getRaster().getDataBuffer()).getData()
				);
			}
			
			int width = input.getWidth(), height = input.getHeight();
			GrayscaleImage output = new GrayscaleImage(
				width, height, bufferFactory.getBuffer(width * height, false)
			);
			for(int y = 0, index = 0; y < height; y++){
				for(int x = 0; x < width; x++, index++){
					int rgb = input.getRGB(x, y);
					// grayscale by average value
					output.data[index] =
						(byte)((((rgb >> 16) & 0xFF) + ((rgb >> 8) & 0xFF) + (rgb & 0xFF)) / 3);
				}
			}
			
			return output;
		};
	}
	
	
	public static Function<GrayscaleImage, float[]> toFloatArray(BufferFactory<float[]> bufferFactory){
		return input -> {
			int length = input.width * input.height;
			float[] output = bufferFactory.getBuffer(length, false);
			for(int i = 0; i < length; i++)
				output[i] = (input.data[i] & 0xFF) / 255.0f;
			
			return output;
		};
	}
	
	public static Function<GrayscaleImage, double[]> toDoubleArray(BufferFactory<double[]> bufferFactory){
		return input -> {
			int length = input.width * input.height;
			double[] output = bufferFactory.getBuffer(length, false);
			for(int i = 0; i < length; i++)
				output[i] = (input.data[i] & 0xFF) / 255.0;
			
			return output;
		};
	}
}
