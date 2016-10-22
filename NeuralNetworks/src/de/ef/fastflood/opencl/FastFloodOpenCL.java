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
	 * Make always same as @version in JavaDoc in format xxx.yyy.zzz
	 */
	private final static long serialVersionUID = 001_000_000L;
	
	private final static int
		NEURON_COUNTS_INDEX = 4, NEURON_OFFSETS_INDEX = 3,
		INPUTS_INDEX = 2, OUTPUTS_INDEX = 1, WEIGHTS_INDEX = 0;
	
	
	
	private final cl_context context;
	private final cl_device_id device;
	private final Pointer pointers[];
	private final cl_mem memory[];
	private final cl_program program;
	private final cl_kernel kernel;
	
	
	FastFloodOpenCL(int inputSize, int hiddenSizes[], int outputSize, cl_context context, cl_device_id device){
		super(inputSize, hiddenSizes, outputSize);
		
		this.context = context;
		this.device = device;
		
		this.pointers = new Pointer[3];
		this.pointers[INPUTS_INDEX] = Pointer.to(this.inputs);
		this.pointers[OUTPUTS_INDEX] = Pointer.to(this.outputs);
		this.pointers[WEIGHTS_INDEX] = Pointer.to(this.weights);
		
		this.memory = new cl_mem[5];
		this.memory[NEURON_COUNTS_INDEX] = clCreateBuffer(
			context,
			CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			Sizeof.cl_int * this.neuronCounts.length,
			Pointer.to(this.neuronCounts),
			null
		);
		this.memory[NEURON_OFFSETS_INDEX] = clCreateBuffer(
			context,
			CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			Sizeof.cl_int * this.neuronOffsets.length,
			Pointer.to(this.neuronOffsets),
			null
		);
		this.memory[INPUTS_INDEX] = clCreateBuffer(
			context,
			CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			Sizeof.cl_float * this.inputs.length,
			this.pointers[INPUTS_INDEX],
			null
		);
		this.memory[OUTPUTS_INDEX] = clCreateBuffer(
			context,
			CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, // or CL_MEM_USE_HOST_PTR or nothing
			Sizeof.cl_float * this.outputs.length,
			this.pointers[OUTPUTS_INDEX],
			null
		);
		this.memory[WEIGHTS_INDEX] = clCreateBuffer(
			context,
			CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, // or CL_MEM_USE_HOST_PTR or nothing
			Sizeof.cl_float * this.weights.length,
			this.pointers[WEIGHTS_INDEX],
			null
		);
		
		
		this.program = clCreateProgramWithSource(context, 1, new String[]{/* TODO kernel source */}, null, null);
		clBuildProgram(this.program, 0, null, null, null, null);
		
		this.kernel = clCreateKernel(program, "TODO", null);
		// set kernel arguments, reverse order
		for(int i = 0; i < this.memory.length; i++){
			clSetKernelArg(this.kernel, i, Sizeof.cl_mem, Pointer.to(this.memory[this.memory.length - (i + 1)]));
		}
	}
	
	
	@Override
	protected void calculateLayer(int layer) throws IOException{
		
	}
	
	
	@Override
	protected void writeInputs() throws IOException{
		clEnqueueWriteBuffer(
			/* TODO queue */null,
			this.memory[INPUTS_INDEX],
			CL_TRUE,
			0,
			this.outputs.length * Sizeof.cl_float,
			this.pointers[INPUTS_INDEX],
			0,
			null,
			null
		);
	}
	
	@Override
	protected void readOutputs() throws IOException{
		clEnqueueReadBuffer(
			/* TODO queue */null,
			this.memory[OUTPUTS_INDEX],
			CL_TRUE,
			0,
			this.outputs.length * Sizeof.cl_float,
			this.pointers[OUTPUTS_INDEX],
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