package de.ef.neuralnetworks;

import java.io.IOException;
import java.util.function.Function;

public final class NeuralNetworkWrapper<I, O, N>
	implements NeuralNetwork<I, N>{
	
	private static final long serialVersionUID = 1L;
	// TODO serial conversion passthru
	
	
	
	private final NeuralNetwork<I, O> network;
	private final Function<O, N> converter;
	private final Function<N, O> reverseConverter;
	
	
	public NeuralNetworkWrapper(NeuralNetwork<I, O> network, Function<O, N> converter, Function<N, O> reverseConverter){
		this.network = network;
		
		this.converter = converter;
		this.reverseConverter = reverseConverter;
	}
	
	
	@Override
	public N calculate(I input) throws IOException{
		return this.converter.apply(this.network.calculate(input));
	}
	
	@Override
	public double train(I input, N output) throws IOException{
		return this.network.train(input, this.reverseConverter.apply(output));
	}
	
	@Override
	public double train(I input, N output, double learningRate) throws IOException{
		return this.network.train(input, this.reverseConverter.apply(output), learningRate);
	}
}
