package de.ef.neuralnetworks;

import java.io.IOException;
import java.util.function.Function;

public final class NeuralNetworkWrapper<I, O, IW, OW>
	implements NeuralNetwork<IW, OW>{
	
	@SuppressWarnings("unchecked")
	public final static <I, O> NeuralNetwork<I, O> wrap(
			NeuralNetwork<float[], float[]> network, Class<I> inputClass, Class<O> outputClass){
		if(inputClass == float[].class){
			if(outputClass == float[].class)
				return (NeuralNetwork<I, O>)network;
			
			OutputConverter<float[], O> outputWrap = wrapOutput(outputClass);
			return (NeuralNetwork<I, O>)
				new NeuralNetworkOutputWrapper<>(network, outputWrap.forward, outputWrap.reverse);
		}
		else if(outputClass == float[].class)
			return (NeuralNetwork<I, O>)new NeuralNetworkInputWrapper<>(network, wrapInput(inputClass));
		
		OutputConverter<float[], O> outputWrap = wrapOutput(outputClass);
		return (NeuralNetwork<I, O>)
			new NeuralNetworkWrapper<>(network, wrapInput(inputClass), outputWrap.forward, outputWrap.reverse);
	}
	
	private final static <I> Function<I, float[]> wrapInput(Class<I> inputClass){
		if(inputClass == Float.class) return i -> new float[]{(Float)i};
		if(inputClass.getSuperclass() == Number.class) return i -> new float[]{((Number)i).floatValue()};
		throw new NoWrapperFoundException("No input wrapper found.");
	}
	
	@SuppressWarnings("unchecked")
	private final static <O> OutputConverter<float[], O> wrapOutput(Class<O> outputClass){
		OutputConverter<?, ?> converter;
		if(outputClass == Float.class)
			converter = new OutputConverter<float[], Float>(o -> o[0], o -> new float[]{(Float)o});
		else if(outputClass == Double.class)
			converter = new OutputConverter<float[], Double>(o -> Double.valueOf(o[0]), o -> new float[]{o.floatValue()});
		else throw new NoWrapperFoundException("No output wrapper found.");
		
		return (OutputConverter<float[], O>)converter;
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
