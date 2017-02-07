package de.ef.neuralnetworks;

import java.io.IOException;
import java.util.function.Function;

final class NeuralNetworkOutputWrapper<I, O, OW>
	implements NeuralNetwork<I, OW>{
	
	private final static long serialVersionUID = 1L;
	// TODO serial conversion passthru
	
	
	
	private final NeuralNetwork<I, O> network;
	private final Function<O, OW> converter;
	private final Function<OW, O> reverseConverter;
	
	
	public NeuralNetworkOutputWrapper(NeuralNetwork<I, O> network, Function<O, OW> converter, Function<OW, O> reverseConverter){
		this.network = network;
		
		this.converter = converter;
		this.reverseConverter = reverseConverter;
	}
	
	
	@Override
	public OW calculate(I input) throws IOException{
		return this.converter.apply(this.network.calculate(input));
	}
	
	@Override
	public double train(I input, OW output) throws IOException{
		return this.network.train(input, this.reverseConverter.apply(output));
	}
	
	@Override
	public double train(I input, OW output, double learningRate) throws IOException{
		return this.network.train(input, this.reverseConverter.apply(output), learningRate);
	}
}
