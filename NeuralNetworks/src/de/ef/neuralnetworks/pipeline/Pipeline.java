package de.ef.neuralnetworks.pipeline;

import java.util.List;
import java.util.function.Function;

public class Pipeline<I, O>{
	
	private final List<Function<Object, Object>> steps;
	
	
	Pipeline(List<Function<Object, Object>> steps){
		this.steps = steps;
	}
	
	
	@SuppressWarnings("unchecked")
	public O process(I input){
		Object lastOutput = input;
		for(Function<Object, Object> step : steps){
			lastOutput = step.apply(lastOutput);
		}
		return (O)lastOutput;
	}
}
