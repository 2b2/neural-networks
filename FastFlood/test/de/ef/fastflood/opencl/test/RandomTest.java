package de.ef.fastflood.opencl.test;

import java.util.List;

import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_context_properties;
import org.jocl.cl_device_id;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;
import org.jocl.cl_program;

import org.junit.Assert;
import org.junit.Test;

import de.ef.fastflood.opencl.FastFloodOpenCLContext;
import de.ef.fastflood.opencl.ProgramBuilder;
import de.ef.fastflood.opencl.FastFloodOpenCLContext.OpenCLConfiguration;

import static org.jocl.CL.*;

public class RandomTest{
	
	public RandomTest(){}
	
	
	@Test
	@SuppressWarnings("deprecation")
	public void test(){
		CL.setExceptionsEnabled(true);
		
		List<OpenCLConfiguration> configs =
			FastFloodOpenCLContext.listConfigurations();
		
		if(configs == null || configs.size() == 0)
			Assert.fail("No device found.");
		
		OpenCLConfiguration config = configs.get(0);
		
		cl_context_properties contextProperties = new cl_context_properties();
		contextProperties.addProperty(CL_CONTEXT_PLATFORM, config.platform);
		cl_context context = clCreateContext(contextProperties, 1, new cl_device_id[]{config.device}, null, null, null);
		cl_command_queue commandQueue = clCreateCommandQueue(context, config.device, 0, null);
		
		cl_program program = ProgramBuilder.loadAndBuildProgram(context, config.device, "/fast-flood.cl", false);
		cl_kernel kernel = clCreateKernel(program, "randomFloatArray", null);
		
		float array[] = new float[100];
		cl_mem arrayObject = clCreateBuffer(context, CL_MEM_READ_WRITE, Sizeof.cl_float * array.length, null, null);
		
		clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(arrayObject));
		clSetKernelArg(kernel, 1, Sizeof.cl_int, Pointer.to(new int[]{}));
		clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, new long[]{array.length}, new long[]{1}, 0, null, null);
		clFinish(commandQueue);
		
		clEnqueueReadBuffer(commandQueue, arrayObject, CL_TRUE, 0, Sizeof.cl_float * array.length, Pointer.to(array), 0, null, null);
		
		clReleaseMemObject(arrayObject);
		clReleaseKernel(kernel);
		clReleaseProgram(program);
		clReleaseCommandQueue(commandQueue);
		clReleaseContext(context);
		
		for(int i = 0; i < array.length; i++)
			if(array[i] < -1 || array[i] > 1) Assert.fail("Random number out of range: " + array[i] + " (at index: " + i + ")");
	}
}
