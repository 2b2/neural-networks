package de.ef.fastflood.opencl;

import de.ef.fastflood.FastFlood;

// TODO comment
// native neural network implementation using OpenCL via the java wrapper jocl
// version: 1.1, date: 16.06.2016, author: Erik Fritzsche
public class FastFloodOpenCL
	extends FastFlood{
	
	private OpenCLNatives natives;
	
	
	FastFloodOpenCL(int inputSize, int hiddenSizes[], int outputSize, OpenCLNatives natives){
		super(inputSize, hiddenSizes, outputSize);
		
		this.natives = natives;
	}
	
	
	@Override
	protected void calculateLayer(int layer) throws RuntimeException{
		
	}
	
	
	@Override
	protected void loadResources() throws RuntimeException{
		
	}
	
	@Override
	protected void releaseResources() throws RuntimeException{
		
	}
	
	@Override
	protected void syncResources() throws RuntimeException{
		
	}
	
	@Override
	protected void loadStaticResources() throws RuntimeException{
		
	}
	
	@Override
	protected void releaseStaticResources() throws RuntimeException{
		
	}
}