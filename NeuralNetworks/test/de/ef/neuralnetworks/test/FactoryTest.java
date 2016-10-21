package de.ef.neuralnetworks.test;

import org.junit.Assert;
import org.junit.Test;

import de.ef.neuralnetworks.NeuralNetwork;
import de.ef.neuralnetworks.NeuralNetworkFactory;

public class FactoryTest{

	public FactoryTest(){}
	
	
	@Test
	public void test(){
		String config = "{\"implementation\": \"SlowWave\", \"layers\": [3, 2, 1]}";
		
		NeuralNetwork network = NeuralNetworkFactory.create(config);
		
		if(network == null){
			// LANGUAGE is or was?
			Assert.fail("The returned network is null.");
		}
	}
}