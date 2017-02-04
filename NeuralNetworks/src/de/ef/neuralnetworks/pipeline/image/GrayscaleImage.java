package de.ef.neuralnetworks.pipeline.image;

public class GrayscaleImage{
	
	public final int width, height;
	public final byte[] data;
	
	
	public GrayscaleImage(int width, int height, byte[] data){
		this.width = width;
		this.height = height;
		
		this.data = data;
	}
}
