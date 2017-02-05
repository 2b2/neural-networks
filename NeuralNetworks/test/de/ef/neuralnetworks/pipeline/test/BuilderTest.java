package de.ef.neuralnetworks.pipeline.test;

import org.junit.Assert;
import org.junit.Test;

import de.ef.neuralnetworks.pipeline.Pipeline;
import de.ef.neuralnetworks.pipeline.PipelineBuilder;

public class BuilderTest{
	
	public BuilderTest(){}
	
	
	@Test
	public void test(){
		Pipeline<Integer, Boolean> pipeline =
			new PipelineBuilder<Integer, Boolean>().singleFunction(i -> i > 0).build();
		
		if(pipeline.process(1) != true) Assert.fail("Single pipeline not working.");
		
		pipeline =
			new PipelineBuilder<Integer, Boolean>()
			.root(i -> (double)i)
			.branch(d -> d.floatValue(), f -> f > 1, new PipelineBuilder<Float, Boolean>().singleFunction(f -> false))
			.pipe(f -> f < 0.5 ? true : false)
			.exit(b -> b == false)
			.build();
		
		if(pipeline.process(1) != true) Assert.fail("Basic pipeline not working.");
		
		if(pipeline.process(2) != false) Assert.fail("Branching pipeline not working.");
	}
}
