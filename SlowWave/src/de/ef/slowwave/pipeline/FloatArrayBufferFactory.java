package de.ef.slowwave.pipeline;

import java.util.HashMap;
import java.util.Map;

import de.ef.neuralnetworks.pipeline.BufferFactory;

public class FloatArrayBufferFactory
	implements BufferFactory<float[]>{
	
	private final int bufferRingSize;
	private Map<Integer, FixedFloatArrayBufferFactory> factories;
	
	
	public FloatArrayBufferFactory(int bufferRingSize){
		this.bufferRingSize = bufferRingSize;
		
		this.factories = new HashMap<>();
	}
	
	
	@Override
	public float[] getBuffer(int size, boolean clear){
		FixedFloatArrayBufferFactory factory = this.factories.get(size);
		if(factory == null)
			this.factories.put(size, factory = new FixedFloatArrayBufferFactory(size, this.bufferRingSize));
		
		return factory.getBufferUnchecked(clear);
	}
	
	
	
	public static class FixedFloatArrayBufferFactory
		implements BufferFactory<float[]>{
		
		private final int size;
		private final float[][] bufferRing;
		private int current = 0;
		
		
		public FixedFloatArrayBufferFactory(int size, int bufferRingSize){
			this.size = size;
			
			this.bufferRing = new float[bufferRingSize][];
		}
		
		
		@Override
		public float[] getBuffer(int size, boolean clear){
			if(size != this.size)
				throw new IllegalArgumentException("Size of fixed float array buffer can not be changed.");
			
			return this.getBufferUnchecked(clear);
		}
		
		private float[] getBufferUnchecked(boolean clear){
			if(this.bufferRing[this.current] == null)
				this.bufferRing[this.current] = new float[this.size];
			else if(clear == true)
				FloatArrayBufferFactory.fastFill(this.bufferRing[this.current], 0f, 0, this.size);
			
			float[] buffer = this.bufferRing[this.current];
			
			if(++this.current == this.bufferRing.length) this.current = 0;
			
			return buffer;
		}
	}
	
	
	
	public static void fastFill(float buffer[], float value, int offset, int length){
		// fast method to fill data with one value, faster then Arrays#fill
		buffer[offset] = value;
		// fill in 2^n float partitions
		int i;
		for(i = 1; i + i <= length; i += i)
			System.arraycopy(buffer, offset, buffer, offset + i, i);
		// fill remaining partition
		System.arraycopy(buffer, offset, buffer, offset + i, length - i);
	}
}
