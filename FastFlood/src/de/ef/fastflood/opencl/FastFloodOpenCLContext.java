package de.ef.fastflood.opencl;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_device_id;
import org.jocl.cl_platform_id;

import de.ef.neuralnetworks.NeuralNetwork;
import de.ef.neuralnetworks.NeuralNetworkContext;
import de.ef.neuralnetworks.NeuralNetworkContextFactory;
import de.ef.neuralnetworks.NeuralNetworkWrapper;

import static org.jocl.CL.*;

public class FastFloodOpenCLContext
	implements NeuralNetworkContext{
	
	static{
		try{
			Method register =
				NeuralNetworkContextFactory.class.getDeclaredMethod("register", String.class, Class.class);
			register.setAccessible(true);
			register.invoke(null, "FastFlood.OpenCL", FastFloodOpenCLContext.class);
		}catch(NoSuchMethodException | IllegalAccessException | IllegalArgumentException | InvocationTargetException e){
			throw new RuntimeException(e);
		}
	}
	
	
	
	public FastFloodOpenCLContext(){}
	
	
	@Override
	public <I, O> NeuralNetwork<I, O> createNeuralNetwork(Class<I> inputClass, Class<O> outputClass, Map<String, Object> properties){
		int inputLayerSize = (Integer)properties.get(INPUT_LAYER_SIZE);
		int outputLayerSize = (Integer)properties.get(OUTPUT_LAYER_SIZE);
		int hiddenLayerCount = (Integer)properties.get(HIDDEN_LAYER_COUNT);
		int hiddenLayerSizes[] = new int[hiddenLayerCount];
		for(int i = 0; i < hiddenLayerCount; i++)
			hiddenLayerSizes[i] = (Integer)properties.get(HIDDEN_LAYER_SIZE.replace("*", String.valueOf(i)));
		
		OpenCLConfiguration config = (OpenCLConfiguration)properties.get("opencl.config"); // TODO make key constant
		
		return NeuralNetworkWrapper.wrapPrimitiveArray(
			new FastFloodOpenCL(inputLayerSize, hiddenLayerSizes, outputLayerSize, config),
			float[].class, inputClass, outputClass
		);
	}
	
	
	
	public final static List<OpenCLConfiguration> listConfigurations(){
		return listConfigurations(CL_DEVICE_TYPE_ALL);
	}
	
	public final static List<OpenCLConfiguration> listConfigurations(long deviceType){
		Set<OpenCLConfiguration> configurations = new TreeSet<>(
			(a, b) -> a.type != b.type ? a.type.compareTo(b.type) : a.description.compareTo(b.description)
		);
		
		// get platform count
		int platformCount[] = new int[1];
		clGetPlatformIDs(0, null, platformCount);
		
		// get all platforms
		cl_platform_id platforms[] = new cl_platform_id[platformCount[0]];
		clGetPlatformIDs(platforms.length, platforms, null);
		
		// loop thru platforms and list devices
		for(cl_platform_id platform : platforms){
			String platformName = getPlatformName(platform);
			
			// get device count
			int deviceCount[] = new int[1];
			clGetDeviceIDs(platform, deviceType, 0, null, deviceCount);
			
			// get all devices
			cl_device_id devices[] = new cl_device_id[deviceCount[0]];
			clGetDeviceIDs(platform, deviceType, deviceCount[0], devices, null);
			
			// loop true all available devices of current platform
			for(cl_device_id device : devices){
				configurations.add(
					new OpenCLConfiguration(platform, device, getDeviceType(device), getDeviceName(device) + " [" + platformName + "]")
				);
			}
		}
		
		return new ArrayList<>(configurations);
	}
	
	final static String getPlatformName(cl_platform_id platform){
		// get name length
		long length[] = new long[1];
		clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, null, length);
		
		// get name
		byte buffer[] = new byte[(int)length[0]];
		clGetPlatformInfo(platform, CL_PLATFORM_NAME, buffer.length, Pointer.to(buffer), null);

		// create string and trim white-spaces
		return new String(buffer, 0, buffer.length).trim();
	}
	
	final static String getDeviceName(cl_device_id device){
		// get name length
		long length[] = new long[1];
		clGetDeviceInfo(device, CL_DEVICE_NAME, 0, null, length);
		
		// get name
		byte buffer[] = new byte[(int)length[0]];
		clGetDeviceInfo(device, CL_DEVICE_NAME, buffer.length, Pointer.to(buffer), null);
		
		// create string and trim white-spaces
		return new String(buffer, 0, buffer.length).trim();
	}
	
	final static OpenCLConfiguration.Type getDeviceType(cl_device_id device){
		long type[] = new long[1];
		clGetDeviceInfo(device, CL_DEVICE_TYPE, Sizeof.cl_long, Pointer.to(type), null);
		
		if((type[0] & CL_DEVICE_TYPE_GPU) != 0)
			return OpenCLConfiguration.Type.GPU;
		else if((type[0] & CL_DEVICE_TYPE_ACCELERATOR) != 0)
			return OpenCLConfiguration.Type.ACCELERATOR;
		else if((type[0] & CL_DEVICE_TYPE_CPU) != 0)
			return OpenCLConfiguration.Type.CPU;
		else if((type[0] & CL_DEVICE_TYPE_DEFAULT) != 0)
			return OpenCLConfiguration.Type.DEFAULT;
		else
			return OpenCLConfiguration.Type.OTHER;
	}
	
	
	
	public final static class OpenCLConfiguration{
		
		public final cl_platform_id platform;
		public final cl_device_id device;
		
		public final Type type;
		public final String description;
		
		private OpenCLConfiguration(cl_platform_id platform, cl_device_id device, Type type, String description){
			this.platform = platform;
			this.device = device;
			
			this.type = type;
			this.description = description;
		}
		
		
		@Override
		public String toString(){
			return this.description;
		}
		
		
		
		public static enum Type{
			GPU, ACCELERATOR, CPU, DEFAULT, OTHER
		}
	}
}
