package de.ef.fastflood.opencl;

import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_kernel;
import org.jocl.cl_program;

// TODO many things
// version: 1.0, date: 15.06.2016, author: Erik Fritzsche
public class OpenCLNatives{
	
	private cl_context context;
	private cl_command_queue commandQueue;
	private cl_kernel kernel;
	private cl_program program;
	
	
	public OpenCLNatives(){
		
	}
	
	
//	public void loadKernel(){
//		
//	}
//	
//	public void unloadKernel(){
//		
//	}
}