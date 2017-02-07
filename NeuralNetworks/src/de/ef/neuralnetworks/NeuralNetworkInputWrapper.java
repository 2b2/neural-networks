package de.ef.neuralnetworks;

import java.io.IOException;
import java.util.function.Function;

public final class NeuralNetworkInputWrapper<I, O, IW>
	implements NeuralNetwork<IW, O>{
	
	private static final long serialVersionUID = 1L;
	// TODO serial conversion passthru
	
	
	
	private final NeuralNetwork<I, O> network;
	private final Function<IW, I> converter;
	
	
	public NeuralNetworkInputWrapper(NeuralNetwork<I, O> network, Function<IW, I> converter){
		this.network = network;
		
		this.converter = converter;
	}
	
	
	@Override
	public O calculate(IW input) throws IOException{
		return this.network.calculate(this.converter.apply(input));
	}
	
	@Override
	public double train(IW input, O output) throws IOException{
		return this.network.train(this.converter.apply(input), output);
	}
	
	@Override
	public double train(IW input, O output, double learningRate) throws IOException{
		return this.network.train(this.converter.apply(input), output, learningRate);
	}
}
