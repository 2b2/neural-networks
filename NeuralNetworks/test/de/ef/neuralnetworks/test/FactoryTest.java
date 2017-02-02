package de.ef.neuralnetworks.test;

import java.util.HashMap;
import java.util.Map;

import org.junit.Assert;
import org.junit.Test;

import de.ef.neuralnetworks.NeuralNetwork;
import de.ef.neuralnetworks.NeuralNetworkContext;
import de.ef.neuralnetworks.NeuralNetworkContextFactory;

public class FactoryTest{

	public FactoryTest(){}
	
	
	@Test
	public void test() throws ClassNotFoundException{
		Map<String, Object> properties = new HashMap<>();
		
		Class.forName("de.ef.slowwave.SlowWaveContext");
		NeuralNetworkContext context = NeuralNetworkContextFactory.create("SlowWave");
		
		NeuralNetwork<double[], double[]> network = context.createNeuralNetwork(double[].class, double[].class, properties);
		
		if(network == null)
			Assert.fail("The returned network is null.");
	}
}
