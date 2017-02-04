package de.ef.slowwave.pipeline.image.test;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JOptionPane;

import de.ef.neuralnetworks.pipeline.image.ForegroundObjectExtractor;
import de.ef.neuralnetworks.pipeline.image.GrayscaleImage;
import de.ef.slowwave.pipeline.image.SlowWaveForegroundObjectExtractor;

public class ForegroundObjectTest{
	
	public ForegroundObjectTest(){}
	
	
	public static void main(String ... args) throws IOException{
		int outputWidth = 256, outputHeight = 64;
		
		ForegroundObjectExtractor extractor =
			new SlowWaveForegroundObjectExtractor(outputWidth, outputHeight, 127, (s, c) -> new byte[s]);
		
		BufferedImage image =
			ImageIO.read(new File(args.length > 0 ? args[0] : "./test.png"));
		
		int width = image.getWidth(), height = image.getHeight();
		
		GrayscaleImage input = new GrayscaleImage(width, height, new byte[width * height]);
		for(int y = 0, index = 0; y < height; y++){
			for(int x = 0; x < width; x++, index++){
				int rgb = image.getRGB(x, y);
				input.data[index] =
					(byte)((((rgb >> 16) & 0xFF) + ((rgb >> 8) & 0xFF) + (rgb & 0xFF)) / 3);
			}
		}
		
		GrayscaleImage output = extractor.apply(input);
		
		BufferedImage outputImage =
			new BufferedImage(outputWidth, outputHeight, BufferedImage.TYPE_BYTE_GRAY);
		
		byte[] outputImageData = ((DataBufferByte)outputImage.getRaster().getDataBuffer()).getData();
		System.arraycopy(output.data, 0, outputImageData, 0, outputWidth * outputHeight);
		
		JOptionPane.showMessageDialog(null, new ImageIcon(outputImage), "Object Extractor Test", JOptionPane.PLAIN_MESSAGE);
	}
}
