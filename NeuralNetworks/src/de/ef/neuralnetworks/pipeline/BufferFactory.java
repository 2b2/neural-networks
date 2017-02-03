package de.ef.neuralnetworks.pipeline;

public interface BufferFactory<T>{
	
	public T getBuffer(int size, boolean clear);
}
