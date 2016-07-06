package de.ef.fastflood.opencl;

import org.jocl.cl_device_id;

import de.ef.neuralnetworks.NeuralNetworkConstructor;

// lists all currently available OpenCL-devices
// and generate neural-network constructors for them
// version: 1.1, date: 16.06.2016, author: Erik Fritzsche
public class OpenCLDevices{
	
	// list of all OpenCL-devices for neural-networks
	public final static NeuralNetworkConstructor CONSTRUCTORS[] = getDeviceConstructors();
	
	
	// give each OpenCL-device a constructor
	private static NeuralNetworkConstructor[] getDeviceConstructors(){
		// TODO give each OpenCL-device a constructor
		return new NeuralNetworkConstructor[]{(i, h, o) -> OpenCLDevices.createNewNeuralNetwork(i, h, o, null)}; 
	}
	
	// make a new context for the given
	// device and create a neural-network
	private static FastFloodOpenCL createNewNeuralNetwork(int i, int h[], int o, cl_device_id device){
		// TODO create context
		return new FastFloodOpenCL(i, h, o, null);
	}
	
	
	
	private OpenCLDevices(){}
}