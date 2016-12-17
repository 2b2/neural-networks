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
import de.ef.neuralnetworks.NeuralNetworkData;
import de.ef.neuralnetworks.NeuralNetworkFactory;

public class ComparatorTest{
	
	public ComparatorTest(){}
	
	
	@Test
	public void text() throws IOException{
		String config = "{\"implementation\": \"SlowWave\", \"layers\": [2, 3, 1]}";

		NeuralNetwork same = NeuralNetworkFactory.create(config);
		NeuralNetwork compare = NeuralNetworkFactory.create(config);
		
		if(same == null || compare == null)
			Assert.fail("The returned network is null.");
		
		NeuralNetworkComparator<NeuralNetworkData> comparator =
			new NeuralNetworkComparator<>(same, compare);
		
		Comparator<NeuralNetworkData> reverse = comparator.reversed();
		
		Map<Long, List<NeuralNetworkData>> testSets = new HashMap<>();
		testSets.put(0L, Arrays.asList(() -> new double[]{0.0}));
		testSets.put(0L, Arrays.asList(() -> new double[]{0.01}));

		testSets.put(1L, Arrays.asList(() -> new double[]{0.19}));
		testSets.put(1L, Arrays.asList(() -> new double[]{0.2}));
		testSets.put(1L, Arrays.asList(() -> new double[]{0.21}));
		
		testSets.put(2L, Arrays.asList(() -> new double[]{0.39}));
		testSets.put(2L, Arrays.asList(() -> new double[]{0.4}));
		testSets.put(2L, Arrays.asList(() -> new double[]{0.41}));
		
		testSets.put(3L, Arrays.asList(() -> new double[]{0.59}));
		testSets.put(3L, Arrays.asList(() -> new double[]{0.6}));
		testSets.put(3L, Arrays.asList(() -> new double[]{0.61}));
		
		testSets.put(4L, Arrays.asList(() -> new double[]{0.79}));
		testSets.put(4L, Arrays.asList(() -> new double[]{0.8}));
		testSets.put(4L, Arrays.asList(() -> new double[]{0.81}));
		
		testSets.put(5L, Arrays.asList(() -> new double[]{0.99}));
		testSets.put(5L, Arrays.asList(() -> new double[]{1.0}));
		
		List<NeuralNetworkData> merged = new ArrayList<>();
		merged.add(() -> new double[]{0.0});
		merged.add(() -> new double[]{0.2});
		merged.add(() -> new double[]{0.4});
		merged.add(() -> new double[]{0.6});
		merged.add(() -> new double[]{0.8});
		merged.add(() -> new double[]{1.0});
		
		comparator.train(testSets, error -> error < 0.001);
		
		Collections.shuffle(merged);
		System.out.println("Random: " + this.toString(merged));
		merged.sort(comparator);
		
		List<NeuralNetworkData> reversed = new ArrayList<>(merged);
		reversed.sort(reverse);
		
		System.out.println("Sorted: " + this.toString(merged));
		
		Collections.reverse(reversed);
		if(merged.equals(reversed) == false)
			Assert.fail("The reversed sorted list is not equal to the sorted list.");
	}
	
	private String toString(List<NeuralNetworkData> elements){
		StringBuilder builder = new StringBuilder();
		for(NeuralNetworkData element : elements){
			builder.append(", ");
			builder.append(element.getData()[0]);
		}
		return builder.replace(0, 2, "[").append("]").toString();
	}
}
