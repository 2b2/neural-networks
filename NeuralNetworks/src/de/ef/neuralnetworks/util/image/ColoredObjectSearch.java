package de.ef.neuralnetworks.util.image;

import java.awt.image.BufferedImage;

// a utility class to find objects in images based on color
// this class is not thread-safe
// version: 1.0, date: 05.06.2016, author: Erik Fritzsche
public class ColoredObjectSearch{

	private ColorDescriptor descriptor;
	
	
	public ColoredObjectSearch(ColorDescriptor descriptor){
		this.descriptor = descriptor;
	}
	
	
	// tries to find and cut out an object based on the given
	// ColorDescriptor if null is returned then no object was found
	public BufferedImage findObject(BufferedImage image){
		int width = image.getWidth(), height = image.getHeight();
		int startX = 0, startY = 0, endX = 0, endY = 0;
		
		outer : for(int x = 0; x < width; x++){
			for(int y = 0; y < height; y++){
				if(this.descriptor.isObject(image.getRGB(x, y)) == true){
					startX = x;
					break outer;
				}
			}
		}
		outer : for(int x = width - 1; x >= 0; x--){
			for(int y = 0; y < height; y++){
				if(this.descriptor.isObject(image.getRGB(x, y)) == true){
					endX = x + 1;
					break outer;
				}
			}
		}
		outer : for(int y = 0; y < height; y++){
			for(int x = 0; x < width; x++){
				if(this.descriptor.isObject(image.getRGB(x, y)) == true){
					startY = y;
					break outer;
				}
			}
		}
		outer : for(int y = height - 1; y >= 0; y--){
			for(int x = 0; x < width; x++){
				if(this.descriptor.isObject(image.getRGB(x, y)) == true){
					endY = y + 1;
					break outer;
				}
			}
		}
		
		if(endX - startX == 0){
			return null;
		}
		return image.getSubimage(startX, startY, endX - startX, endY - startY);
	}
	
	
	
	// describes what colors are part of the object and which not
	public static interface ColorDescriptor{
		
		// this function should return if the color of
		// a given pixel is part of the object to find or not
		public boolean isObject(int color);
	}
	
	
	// two very simple implementations of ColorDescriptor to
	// find objects based on the object foreground/image background color
	public static class ForegroundDescriptor
		implements ColorDescriptor{
		
		private int objectColor;
		
		public ForegroundDescriptor(int objectColor){
			this.objectColor = objectColor;
		}
		
		@Override
		public boolean isObject(int color){
			return this.objectColor == color;
		}
	}
	
	public static class BackgroundDescriptor
		implements ColorDescriptor{
		
		private int backgroundColor;
		
		public BackgroundDescriptor(int backgroundColor){
			this.backgroundColor = backgroundColor;
		}
		
		@Override
		public boolean isObject(int color){
			return this.backgroundColor != color;
		}
	}
}
