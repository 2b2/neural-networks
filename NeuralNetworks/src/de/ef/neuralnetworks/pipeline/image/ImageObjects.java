package de.ef.neuralnetworks.pipeline.image;

import java.io.IOException;
import java.util.Map;

import de.ef.neuralnetworks.pipeline.BufferFactory;

public interface ImageObjects{
	
	public static ImageObjects load(String className)
			throws ClassNotFoundException, InstantiationException, IllegalAccessException{
		@SuppressWarnings("unchecked")
		Class<ImageObjects> classObject = (Class<ImageObjects>)Class.forName(className);
		
		return classObject.newInstance();
	}
	
	
	
	public void init(Map<String, Object> properties) throws IOException;
	
	
	public ImageForegroundObjectExtractor foregroundObjectExtractor(
			int outputWidth, int outputHeight, int backgroundThreshold,
			BufferFactory<byte[]> bufferFactory) throws IOException;
}
