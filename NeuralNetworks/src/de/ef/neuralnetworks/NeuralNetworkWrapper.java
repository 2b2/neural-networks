package de.ef.neuralnetworks;

import java.io.IOException;
import java.util.function.Function;

public final class NeuralNetworkWrapper<I, O, IW, OW>
	implements NeuralNetwork<IW, OW>{
	
	private static final long serialVersionUID = 1L;
	// TODO serial conversion passthru
	
	
	
	private final NeuralNetwork<I, O> network;
	private final Function<IW, I> inputConverter;
	private final Function<O, OW> outputConverter;
	private final Function<OW, O> reverseOutputConverter;
	
	
	public NeuralNetworkWrapper(NeuralNetwork<I, O> network,
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
