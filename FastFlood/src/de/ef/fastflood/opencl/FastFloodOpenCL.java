package de.ef.fastflood.opencl;

import java.io.IOException;

import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_context_properties;
import org.jocl.cl_device_id;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;
import org.jocl.cl_program;

import de.ef.fastflood.FastFlood;
import de.ef.fastflood.opencl.FastFloodOpenCLContext.OpenCLConfiguration;

import static org.jocl.CL.*;

// TODO comment
// native neural network implementation using OpenCL via the java wrapper JOCL
// version: 2, date: 16.06.2016, author: Erik Fritzsche
public class FastFloodOpenCL
	extends FastFlood{
	
	/**
	 * Make always same as @version in JavaDoc in format xxx.yyy.zzz
	 */
	private final static long serialVersionUID = 002_000_000L;
	
	private final static int
		INPUTS_INDEX = 0, OUTPUTS_INDEX = 1,
		NEURONS_INDEX = 0, WEIGHTS_INDEX = 1,
		NEURON_COUNTS_INDEX = 2, NEURON_OFFSETS_INDEX = 3,
		CURRENT_LAYER_INDEX = 4;
	
	
	
	private final cl_context context;
	private final cl_command_queue commandQueue;
	private final Pointer pointers[];
	private final int inputByteSize, outputByteSize, outputByteOffset;
	private final cl_mem memory[];
	private final cl_program program;
	private final cl_kernel calculatdeLayerKernel, trainLayerKernel, randomFloatArrayKernel;
	
	
	@SuppressWarnings("deprecation")
	FastFloodOpenCL(int inputSize, int hiddenSizes[], int outputSize, OpenCLConfiguration config){
		super(inputSize, hiddenSizes, outputSize);
		
		cl_context_properties contextProperties = new cl_context_properties();
		contextProperties.addProperty(CL_CONTEXT_PLATFORM, config.platform);
		context = clCreateContext(contextProperties, 1, new cl_device_id[]{config.device}, null, null, null);
		// using old command queue to support OpenCL 1.2
		commandQueue = clCreateCommandQueue(context, config.device, 0, null);
		//commandQueue = clCreateCommandQueueWithProperties(context, device, null, null);
		
		pointers = new Pointer[2];
		pointers[INPUTS_INDEX] = Pointer.to(inputs);
		pointers[OUTPUTS_INDEX] = Pointer.to(outputs);
		
		inputByteSize = Sizeof.cl_float * inputs.length;
		outputByteSize = Sizeof.cl_float * outputs.length;
		outputByteOffset = Sizeof.cl_float * (inputs.length + (neuronOffsets.length - outputs.length));
		
		memory = new cl_mem[4];
		memory[NEURONS_INDEX] = clCreateBuffer(
			context, CL_MEM_READ_WRITE, Sizeof.cl_float * (inputs.length + neuronOffsets.length), null, null
		);
		memory[WEIGHTS_INDEX] = clCreateBuffer(
			context, CL_MEM_READ_WRITE, Sizeof.cl_float * weightCount, null, null
		);
		memory[NEURON_COUNTS_INDEX] = clCreateBuffer(
			context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_int * neuronCounts.length, Pointer.to(neuronCounts), null
		);
		memory[NEURON_OFFSETS_INDEX] = clCreateBuffer(
			context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_int * neuronOffsets.length, Pointer.to(neuronOffsets), null
		);
		
		program = ProgramBuilder.loadAndBuildProgram(context, config.device, "/fast-flood.cl", true);
		
		calculatdeLayerKernel = clCreateKernel(program, "calculateLayer", null);
		trainLayerKernel = clCreateKernel(program, "trainLayer", null);
		// set kernel arguments
		for(int i = 0; i < memory.length; i++){
			clSetKernelArg(calculatdeLayerKernel, i, Sizeof.cl_mem, Pointer.to(memory[i]));
			clSetKernelArg(trainLayerKernel, i, Sizeof.cl_mem, Pointer.to(memory[i]));
		}
		
		randomFloatArrayKernel = clCreateKernel(program, "randomFloatArray", null);
		clSetKernelArg(randomFloatArrayKernel, 0, Sizeof.cl_mem, Pointer.to(memory[WEIGHTS_INDEX]));
		
		// fill weight random
		clEnqueueNDRangeKernel(
			commandQueue, randomFloatArrayKernel, 1, null, new long[]{weightCount}, new long[]{1}, 0, null, null
		);
		clFinish(commandQueue);
	}
	
	
	@Override
	protected void calculateLayer(int layer) throws IOException{
		clSetKernelArg(calculatdeLayerKernel, CURRENT_LAYER_INDEX, Sizeof.cl_int, Pointer.to(new int[]{layer}));
		
		clEnqueueNDRangeKernel(
			commandQueue, calculatdeLayerKernel, 1, null, new long[]{neuronCounts[layer]}, new long[]{1}, 0, null, null
		);
		clFinish(commandQueue);
	}
	
	
	@Override
	protected void trainLayer(int layer) throws IOException{
		clSetKernelArg(trainLayerKernel, CURRENT_LAYER_INDEX, Sizeof.cl_int, Pointer.to(new int[]{layer}));
		
		clEnqueueNDRangeKernel(
			commandQueue, trainLayerKernel, 1, null, new long[]{neuronCounts[layer]}, new long[]{1}, 0, null, null
		);
		clFinish(commandQueue);
	}
	
	
	@Override
	protected void writeInputs() throws IOException{
		clEnqueueWriteBuffer(
			commandQueue, memory[INPUTS_INDEX], CL_TRUE, 0, inputByteSize, pointers[INPUTS_INDEX], 0, null, null
		);
	}
	
	@Override
	protected void readOutputs() throws IOException{
		clEnqueueReadBuffer(
			commandQueue, memory[OUTPUTS_INDEX], CL_TRUE, outputByteOffset, outputByteSize, pointers[OUTPUTS_INDEX], 0, null, null
		);
	}
	
	
	@Override
	public void close() throws IOException{
		for(int i = 0; i < memory.length; i++){
			clReleaseMemObject(memory[i]);
		}
		
		clReleaseKernel(calculatdeLayerKernel);
		clReleaseKernel(trainLayerKernel);
		clReleaseKernel(randomFloatArrayKernel);
		
		clReleaseProgram(program);
		clReleaseCommandQueue(commandQueue);
		clReleaseContext(context);
	}
}
