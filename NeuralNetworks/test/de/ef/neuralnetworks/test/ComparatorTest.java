package de.ef.neuralnetworks.test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.junit.Assert;
import org.junit.Test;

import de.ef.neuralnetworks.NeuralNetwork;
import de.ef.neuralnetworks.NeuralNetworkComparator;
import de.ef.neuralnetworks.NeuralNetworkFactory;

public class ComparatorTest{
	
	public ComparatorTest(){}
	
	
	@Test
	public void test() throws IOException{
		String config = "{\"implementation\": \"SlowWave\", \"layers\": [2, 3, 1]}";

		NeuralNetwork<double[], Float> same = NeuralNetworkFactory.create(config);
		NeuralNetwork<double[], Float> compare = NeuralNetworkFactory.create(config);
		
		if(same == null || compare == null)
			Assert.fail("The returned network is null.");
		
		NeuralNetworkComparator<Double> comparator =
			new NeuralNetworkComparator<>(same, compare, (a, b) -> {
				double[] combined = new double[]{a, b};
				return combined;
			});
		
		Comparator<Double> reverse = comparator.reversed();
		
		Map<Long, List<Double>> testSets = new HashMap<>();
		testSets.put(0L, Arrays.asList(0.0, 0.01));
		testSets.put(1L, Arrays.asList(0.19, 0.2, 0.21));
		testSets.put(2L, Arrays.asList(0.39, 0.4, 0.41));
		testSets.put(3L, Arrays.asList(0.59, 0.6, 0.61));
		testSets.put(4L, Arrays.asList(0.79, 0.8, 0.81));
		testSets.put(5L, Arrays.asList(0.99, 1.0));
		
		List<Double> merged = new ArrayList<>(Arrays.asList(0.0, 0.2, 0.4, 0.6, 0.8, 1.0));
		
		comparator.train(testSets, error -> error < 0.001);
		
		Collections.shuffle(merged);
		System.out.println("Random: " + merged.toString());
		
		merged.sort(comparator);
		System.out.println("Sorted: " + merged.toString());
		
		List<Double> reversed = new ArrayList<>(merged);
		reversed.sort(reverse);
		Collections.reverse(reversed);
		
		if(merged.equals(reversed) == false)
			Assert.fail("The reversed sorted list is not equal to the sorted list.");
	}
}
