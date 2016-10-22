package de.ef.fastflood.opencl;

import static org.jocl.CL.*;

import java.io.IOException;

import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_context;
import org.jocl.cl_device_id;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;
import org.jocl.cl_program;

import de.ef.fastflood.FastFlood;

// TODO comment
// native neural network implementation using OpenCL via the java wrapper JOCL
// version: 1.1, date: 16.06.2016, author: Erik Fritzsche
public class FastFloodOpenCL
	extends FastFlood{
	
	/**
	 * Make always same as @version in JavaDoc in format xxxx.yyyy.zzzz
	 */
	private static final long serialVersionUID = 0001_0000_0000L;
	
	
	
	private final cl_context context;
	private final cl_device_id device;
	private final cl_mem memory[];
	private final cl_program program;
	private final cl_kernel kernel;
	
	
	FastFloodOpenCL(int inputSize, int hiddenSizes[], int outputSize, cl_context context, cl_device_id device){
		super(inputSize, hiddenSizes, outputSize);
		
		this.context = context;
		this.device = device;
		
		this.memory = new cl_mem[5];
		this.memory[0] = clCreateBuffer(
			context,
			CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			Sizeof.cl_int * this.neuronCounts.length,
			Pointer.to(this.neuronCounts),
			null
		);
		this.memory[1] = clCreateBuffer(
			context,
			CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			Sizeof.cl_int * this.layerOffsets.length,
			Pointer.to(this.layerOffsets),
			null
		);
		this.memory[2] = clCreateBuffer(
			context,
			CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			Sizeof.cl_int * this.neuronOffsets.length,
			Pointer.to(this.neuronOffsets),
			null
		);
		this.memory[3] = clCreateBuffer(
			context,
			CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, // or CL_MEM_USE_HOST_PTR or nothing
			Sizeof.cl_float * this.neurons.length,
			Pointer.to(this.neurons),
			null
		);
		this.memory[4] = clCreateBuffer(
			context,
			CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, // or CL_MEM_USE_HOST_PTR or nothing
			Sizeof.cl_float * this.weights.length,
			Pointer.to(this.weights),
			null
		);
		
		this.program = clCreateProgramWithSource(context, 1, new String[]{/* TODO kernel source */}, null, null);
		clBuildProgram(this.program, 0, null, null, null, null);
		
		this.kernel = clCreateKernel(program, "TODO", null);
		for(int i = 0; i < this.memory.length; i++){
			clSetKernelArg(this.kernel, i, Sizeof.cl_mem, Pointer.to(this.memory[i]));
		}
	}
	
	
	@Override
	protected void calculateLayer(int layer) throws IOException{
		
	}
	
	
	@Override
	protected void sync() throws IOException{
		// TODO sync resources
	}
	
	@Override
	protected void read() throws IOException{
		clEnqueueReadBuffer(
			/* TODO queue */null,
			this.memory[3],
			CL_TRUE,
			0,
			this.neurons.length * Sizeof.cl_float,
			/* TODO destination pointer*/null,
			0,
			null,
			null
		);
		clEnqueueReadBuffer(
			/* TODO queue */null,
			this.memory[4],
			CL_TRUE,
			0,
			this.weights.length * Sizeof.cl_float,
			/* TODO destination pointer*/null,
			0,
			null,
			null
		);
	}
	
	@Override
	public void close() throws IOException{
		for(int i = 0; i < this.memory.length; i++){
			clReleaseMemObject(this.memory[i]);
		}
		clReleaseKernel(this.kernel);
		clReleaseProgram(this.program);
		clReleaseContext(this.context);
	}
}