package de.ef.neuralnetworks.util.image;

import java.awt.AlphaComposite;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.util.Arrays;

import de.ef.neuralnetworks.NeuralNetworkData;

/**
 * {@code MonochromeImageInput} generates the input for a
 * {@link de.ef.neuralnetworks.NeuralNetwork NeuralNetwork}
 * from an monochrome image, it implements the
 * {@link de.ef.neuralnetworks.NeuralNetworkData NeuralNetworkData}
 * interface.
 * <p>
 * The {@link de.ef.neuralnetworks.NeuralNetworkData#getData() getData()}
 * method is <b>thread-safe</b>.
 * </p>
 * 
 * @author Erik Fritzsche
 * @version 2.0
 * @since 1.0
 */
public class MonochromeImageData
	implements NeuralNetworkData{
	
	/**
	 * Make always same as @version in JavaDoc in format xxx.yyy.zzz
	 */
	private final static long serialVersionUID = 002_000_000L;
	
	
	
	private final int width, height;
	private final double[] inputs;
	
	
	// image.getWidth() has to be evenly divisible by width (same for height)
	// if that is not the case then the image is scaled to fit
	// width times height must be the input size of the neural network for
	// which the input is created
	public MonochromeImageData(BufferedImage image, int backgroundColor, int width, int height){
		this(image, backgroundColor, width, height, false);
	}
	
	// the same as the other constructor but if stretch is set to true
	// then the image is scaled and stretched in such a way that
	// (scaledImage.getWidth() / width) equals (scaledImage.getHeight() / height)
	public MonochromeImageData(BufferedImage image, int backgroundColor, int width, int height, boolean stretch){
		int imageWidth = image.getWidth(), imageHeight = image.getHeight();
		
		imageWidth += (width - (imageWidth % width));
		imageHeight += (height - (imageHeight % height));
		
		int chunkWidth = imageWidth / width, chunkHeight = imageHeight / height;
		
		if(stretch == true){
			if(chunkWidth < chunkHeight){
				chunkWidth = chunkHeight;
				imageWidth = width * chunkWidth;
			}
			else{
				chunkHeight = chunkWidth;
				imageHeight = height * chunkHeight;
			}
		}
		
		BufferedImage scaledImage = new BufferedImage(imageWidth, imageHeight, BufferedImage.TYPE_INT_ARGB);
		Graphics2D g = scaledImage.createGraphics();
		g.setComposite(AlphaComposite.Src);
		g.drawImage(image, 0, 0, imageWidth, imageHeight, null);
		g.dispose();
		
		double[] inputs = new double[width * height];
		double percent = (1.0 / (chunkWidth * chunkHeight));
		for(int x = 0; x < width; x++){
			for(int y = 0; y < height; y++){
				int count = 0;
				for(int chunkX = 0; chunkX < chunkWidth; chunkX++){
					for(int chunkY = 0; chunkY < chunkHeight; chunkY++){
						if(scaledImage.getRGB((x * chunkWidth) + chunkX, (y * chunkHeight) + chunkY) != backgroundColor){
							count++;
						}
					}
				}
				inputs[(y * width) + x] = percent * count;
			}
		}
		
		this.width = width;
		this.height = height;
		this.inputs = inputs;
	}
	
	
	/**
	 * Returns the image data as a double array. The array is cloned to
	 * prevent modification.
	 */
	@Override
	public double[] getData(){
		return Arrays.copyOf(this.inputs, this.inputs.length);
	}
	
	
	// returns a BufferedImage with the dimensions of the
	// neural network input size
	// the image is the completely filled with the given color
	// and is fully opaque where the neural network input
	// equals one and fully transparent where the input equals zero
	public BufferedImage generateInputView(int color){
		BufferedImage inputViewImage = new BufferedImage(this.width, this.height, BufferedImage.TYPE_INT_ARGB);
		
		color = color & 0x00ffffff;
		for(int i = 0; i < this.inputs.length; i++){
			inputViewImage.setRGB((i % this.width), (int)(i / this.width), color | (((int)(255 * inputs[i])) << 24));
		}
		return inputViewImage;
	}
}
