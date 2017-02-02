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
			new PipelineBuilder<Integer, Boolean>()
			.root(i -> (float)i)
				.pipe(f -> f < 0.5 ? true : false)
				.exit(b -> b == false)
			.build();
		
		if(pipeline.process(1) != true) Assert.fail();
		
		pipeline = new PipelineBuilder<Integer, Boolean>().singleFunction(i -> i > 0).build();
		
		if(pipeline.process(1) != true) Assert.fail();
	}
}
