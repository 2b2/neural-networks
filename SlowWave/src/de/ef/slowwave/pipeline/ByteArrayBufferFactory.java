package de.ef.slowwave.pipeline;

import java.util.HashMap;
import java.util.Map;

import de.ef.neuralnetworks.pipeline.BufferFactory;

public class ByteArrayBufferFactory
	implements BufferFactory<byte[]>{
	
	private final int bufferRingSize;
	private Map<Integer, FixedByteArrayBufferFactory> factories;
	
	
	public ByteArrayBufferFactory(int bufferRingSize){
		this.bufferRingSize = bufferRingSize;
		
		this.factories = new HashMap<>();
	}
	
	
	@Override
	public byte[] getBuffer(int size, boolean clear){
		FixedByteArrayBufferFactory factory = this.factories.get(size);
		if(factory == null)
			this.factories.put(size, factory = new FixedByteArrayBufferFactory(size, this.bufferRingSize));
		
		return factory.getBufferUnchecked(clear);
	}
	
	
	
	public static class FixedByteArrayBufferFactory
		implements BufferFactory<byte[]>{
		
		private final int size;
		private final byte[][] bufferRing;
		private int current = 0;
		
		
		public FixedByteArrayBufferFactory(int size, int bufferRingSize){
			this.size = size;
			
			this.bufferRing = new byte[bufferRingSize][];
		}
		
		
		@Override
		public byte[] getBuffer(int size, boolean clear){
			if(size != this.size)
				throw new IllegalArgumentException("Size of fixed byte array buffer can not be changed.");
			
			return this.getBufferUnchecked(clear);
		}
		
		private byte[] getBufferUnchecked(boolean clear){
			if(this.bufferRing[this.current] == null)
				this.bufferRing[this.current] = new byte[this.size];
			else if(clear == true)
				ByteArrayBufferFactory.fastFill(this.bufferRing[this.current], (byte)0, 0, this.size);
			
			byte[] buffer = this.bufferRing[this.current];
			
			if(++this.current == this.bufferRing.length) this.current = 0;
			
			return buffer;
		}
	}
	
	
	
	public static void fastFill(byte buffer[], byte value, int offset, int length){
		// fast method to fill data with one value, faster then Arrays#fill
		buffer[offset] = value;
		// fill in 2^n byte partitions
		int i;
		for(i = 1; i + i <= length; i += i)
			System.arraycopy(buffer, offset, buffer, offset + i, i);
		// fill remaining partition
		System.arraycopy(buffer, offset, buffer, offset + i, length - i);
	}
}
