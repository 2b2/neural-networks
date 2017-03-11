package de.ef.neuralnetworks;

import java.io.IOException;
import java.util.function.Function;

public final class NeuralNetworkWrapper<I, O, IW, OW>
	implements NeuralNetwork<IW, OW>{
	
	@SuppressWarnings("unchecked")
	public static <I, O, P> NeuralNetwork<I, O> wrapPrimitiveArray(
			NeuralNetwork<P, P> network, Class<P> primitiveArrayClass, Class<I> inputClass, Class<O> outputClass){
		if(primitiveArrayClass == float[].class)
			return wrapFloat((NeuralNetwork<float[], float[]>)network, inputClass, outputClass);
		if(primitiveArrayClass == double[].class)
			return wrapDouble((NeuralNetwork<double[], double[]>)network, inputClass, outputClass);
		throw new NoWrapperFoundException("No primitive array wrapper for type: " + primitiveArrayClass);
	}
	
	@SuppressWarnings("unchecked")
	private static <I, O> NeuralNetwork<I, O> wrapFloat(
			NeuralNetwork<float[], float[]> network, Class<I> inputClass, Class<O> outputClass){
		if(inputClass == float[].class){
			if(outputClass == float[].class)
				return (NeuralNetwork<I, O>)network;
			
			OutputConverter<float[], O> outputWrap = wrapOutputFloat(outputClass);
			return (NeuralNetwork<I, O>)
				new NeuralNetworkOutputWrapper<>(network, outputWrap.forward, outputWrap.reverse);
		}
		else if(outputClass == float[].class)
			return (NeuralNetwork<I, O>)new NeuralNetworkInputWrapper<>(network, wrapInputFloat(inputClass));
		
		OutputConverter<float[], O> outputWrap = wrapOutputFloat(outputClass);
		return (NeuralNetwork<I, O>)
			new NeuralNetworkWrapper<>(network, wrapInputFloat(inputClass), outputWrap.forward, outputWrap.reverse);
	}
	
	private static <I> Function<I, float[]> wrapInputFloat(Class<I> inputClass){
		if(inputClass == Float.class) return i -> new float[]{(float)i};
		if(inputClass.getSuperclass() == Number.class) return i -> new float[]{((Number)i).floatValue()};
		if(inputClass == double[].class)
			return d -> {
				double c[] = (double[])d; float f[] = new float[c.length];
				for(int i = 0; i < c.length; i++) f[i] = (float)c[i]; return f;
			};
		throw new NoWrapperFoundException("No input wrapper found.");
	}
	
	@SuppressWarnings("unchecked")
	private static <O> OutputConverter<float[], O> wrapOutputFloat(Class<O> outputClass){
		OutputConverter<float[], ?> converter;
		if(outputClass == Float.class)
			converter = new OutputConverter<float[], Float>(o -> o[0], o -> new float[]{(Float)o});
		else if(outputClass == Double.class)
			converter = new OutputConverter<float[], Double>(o -> (double)o[0], o -> new float[]{o.floatValue()});
		else if(outputClass == double[].class)
			converter = new OutputConverter<float[], double[]>(
				o -> {double d[] = new double[o.length]; for(int i = 0; i < o.length; i++) d[i] = (double)o[i]; return d;},
				o -> {float f[] = new float[o.length]; for(int i = 0; i < o.length; i++) f[i] = (float)o[i]; return f;}
			);
		else throw new NoWrapperFoundException("No output wrapper found.");
		
		return (OutputConverter<float[], O>)converter;
	}
	
	
	@SuppressWarnings("unchecked")
	private static <I, O> NeuralNetwork<I, O> wrapDouble(
			NeuralNetwork<double[], double[]> network, Class<I> inputClass, Class<O> outputClass){
		if(inputClass == double[].class){
			if(outputClass == double[].class)
				return (NeuralNetwork<I, O>)network;
			
			OutputConverter<double[], O> outputWrap = wrapOutputDouble(outputClass);
			return (NeuralNetwork<I, O>)
				new NeuralNetworkOutputWrapper<>(network, outputWrap.forward, outputWrap.reverse);
		}
		else if(outputClass == double[].class)
			return (NeuralNetwork<I, O>)new NeuralNetworkInputWrapper<>(network, wrapInputDouble(inputClass));
		
		OutputConverter<double[], O> outputWrap = wrapOutputDouble(outputClass);
		return (NeuralNetwork<I, O>)
			new NeuralNetworkWrapper<>(network, wrapInputDouble(inputClass), outputWrap.forward, outputWrap.reverse);
	}
	
	private static <I> Function<I, double[]> wrapInputDouble(Class<I> inputClass){
		if(inputClass == Float.class) return i -> new double[]{(double)i};
		if(inputClass.getSuperclass() == Number.class) return i -> new double[]{((Number)i).floatValue()};
		throw new NoWrapperFoundException("No input wrapper found.");
	}
	
	@SuppressWarnings("unchecked")
	private static <O> OutputConverter<double[], O> wrapOutputDouble(Class<O> outputClass){
		OutputConverter<double[], ?> converter;
		if(outputClass == Float.class)
			converter = new OutputConverter<double[], Float>(o -> (float)o[0], o -> new double[]{o.doubleValue()});
		else if(outputClass == Double.class)
			converter = new OutputConverter<double[], Double>(o -> o[0], o -> new double[]{o});
		else throw new NoWrapperFoundException("No output wrapper found.");
		
		return (OutputConverter<double[], O>)converter;
	}
	
	
	private static class OutputConverter<O, OW>{
		
		private final Function<O, OW> forward;
		private final Function<OW, O> reverse;
		
		private OutputConverter(Function<O, OW> forward, Function<OW, O> reverse){
			this.forward= forward;
			this.reverse = reverse;
		}
	}
	
	public static class NoWrapperFoundException
		extends RuntimeException{
		
		private final static long serialVersionUID = 1L;
		
		private NoWrapperFoundException(String message){
			super(message);
		}
	}
	
	
	
	private final static long serialVersionUID = 1L;
	// TODO serial conversion passthru
	
	
	
	private final NeuralNetwork<I, O> network;
	private final Function<IW, I> inputConverter;
	private final Function<O, OW> outputConverter;
	private final Function<OW, O> reverseOutputConverter;
	
	
	private NeuralNetworkWrapper(NeuralNetwork<I, O> network,
			Function<IW, I> inputConverter, Function<O, OW> outputConverter, Function<OW, O> reverseOutputConverter){
		this.network = network;
		
		this.inputConverter = inputConverter;
		this.outputConverter = outputConverter;
		this.reverseOutputConverter = reverseOutputConverter;
	}
	
	
	@Override
	public OW calculate(IW input) throws IOException{
		return this.outputConverter.apply(this.network.calculate(this.inputConverter.apply(input)));
	}
	
	@Override
	public double train(IW input, OW output) throws IOException{
		return this.network.train(this.inputConverter.apply(input), this.reverseOutputConverter.apply(output));
	}
	
	@Override
	public double train(IW input, OW output, double learningRate) throws IOException{
		return this.network.train(this.inputConverter.apply(input), this.reverseOutputConverter.apply(output), learningRate);
	}
}
