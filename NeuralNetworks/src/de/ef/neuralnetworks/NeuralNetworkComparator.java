package de.ef.neuralnetworks;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.function.DoubleFunction;

/**
 * {@code NeuralNetworkComparator} is a {@link java.util.Comparator Comparator}
 * which uses a {@link de.ef.neuralnetworks.NeuralNetwork NeuralNetwork} to
 * figure out if two values are the same or which one is greater.
 * <p>
 * The functionality of this class could be useful to for example
 * sort data into a tree structure to have faster access rates.
 * </p>
 * <p>
 * The calculations are made by a neural-network that was provided
 * on instantiation. This class itself just gives some functionality
 * for the {@link java.util.Comparator#compare compare} method and provides
 * training routines.
 * </p>
 * 
 * @param <T> the {@link de.ef.neuralnetworks.NeuralNetworkData NeuralNetworkData}
 *            type that has to be compared
 * 
 * @author Erik Fritzsche
 * @version 1.0
 * @since 1.0
 */
public class NeuralNetworkComparator<T extends NeuralNetworkData>
	implements NeuralNetwork, Comparator<T>{
	
	/**
	 * Make always same as @version in JavaDoc in format xxx.yyy.zzz
	 */
	private final static long serialVersionUID = 001_000_000L;
	
	
	/**
	 * Constant to indicate the result of the comparison, the closer the
	 * output of the neural network the more likely the result.
	 */
	public final static double LESS_THAN = 0.0, EQUALS = 0.5, GREATER_THAN = 1.0;
	
	
	/**
	 * Constant to indicate the cutoff point of the result.
	 */
	private final static double LT_THRESHOLD = 0.33, GT_THRESHOLD = 0.66;
	
	/**
	 * Internal cache so that the array is not created every time it is needed.
	 */
	private final static double
		LT_OUTPUT[] = {LESS_THAN}, E_OUTPUT[] = {EQUALS}, GT_OUTPUT[] = {GREATER_THAN};
	
	
	
	private final NeuralNetwork implementation;
	
	
	public NeuralNetworkComparator(NeuralNetwork implementation){
		this.implementation = implementation;
	}
	
	
	@Override
	public int compare(T a, T b){
		try{
			double result = this.compareSmooth(a, b);
			if(result < LT_THRESHOLD)
				return -1;
			else if(result > GT_THRESHOLD)
				return 1;
			return 0;
		}
		catch(IOException e){
			throw new RuntimeException(e);
		}
	}
	
	private double compareSmooth(T a, T b) throws IOException{
		double aData[] = a.getData(), bData[] = b.getData();
		double inputs[] = new double[aData.length + bData.length];
		
		System.arraycopy(aData, 0, inputs, 0, aData.length);
		System.arraycopy(bData, 0, inputs, aData.length, bData.length);
		
		return this.implementation.calculate(inputs)[0];
	}
	
	
	@Override
	public double[] calculate(double[] inputs) throws IOException{
		return this.implementation.calculate(inputs);
	}
	
	
	public void train(Map<Long, Collection<T>> trainingSets, DoubleFunction<Boolean> completed) throws IOException{
		List<TrainingData> allData = new LinkedList<>();
		
		int totalCount = 0;
		for(Entry<Long, Collection<T>> container : trainingSets.entrySet())
			for(Iterator<T> i = container.getValue().iterator(); i.hasNext(); totalCount++)
				allData.add(new TrainingData(i.next().getData(), container.getKey()));
		Collections.shuffle(allData);
		
		Collection<OrderedPair> pairs = new ArrayList<>(totalCount * totalCount);
		for(TrainingData a : allData)
			for(TrainingData b : allData)
				pairs.add(new OrderedPair(a, b));
		
		Map<String, Double> expected = new HashMap<>(pairs.size());
		
		double error;
		do{
			error = this.train(pairs, expected);
		}while(completed.apply(error) == false);
	}
	
	protected double train(Collection<OrderedPair> pairs, Map<String, Double> expected) throws IOException{
		double error = 0;
		for(OrderedPair pair : pairs){
			if(pair.same == true)
				error += this.implementation.train(pair.data, E_OUTPUT);
			else{
				Double expectedResult = expected.get(pair.id);
				if(expectedResult == null){
					Double reversedResult = expected.get(pair.reverseId);
					if(reversedResult == null){
						expectedResult = this.implementation.calculate(pair.data)[0];
						if(expectedResult < 0.5 /* TODO maybe make constant */)
							error += this.implementation.train(pair.data, LT_OUTPUT);
						else
							error += this.implementation.train(pair.data, GT_OUTPUT);
						expected.put(pair.id, expectedResult);
						expected.put(pair.reverseId, 1 - expectedResult);
					}
					else{
						if(reversedResult < 0.5)
							error += this.implementation.train(pair.data, GT_OUTPUT);
						else
							error += this.implementation.train(pair.data, LT_OUTPUT);
						reversedResult =
							(reversedResult + (1 - this.implementation.calculate(pair.data)[0])) / 2;
						expected.put(pair.id, 1 - reversedResult);
						expected.put(pair.reverseId, reversedResult);
					}
				}
				else{
					expectedResult =
						(expectedResult + this.implementation.calculate(pair.data)[0]) / 2;
					
					Double reversedResult = expected.get(pair.reverseId);
					if(reversedResult == null){
						if(expectedResult < 0.5 /* TODO maybe make constant */)
							error += this.implementation.train(pair.data, LT_OUTPUT);
						else
							error += this.implementation.train(pair.data, GT_OUTPUT);
						expected.put(pair.id, expectedResult);
						expected.put(pair.reverseId, 1 - expectedResult);
					}
					else{
						if(expectedResult < 0.5 && reversedResult >= 0.5)
							error += this.implementation.train(pair.data, LT_OUTPUT);
						else if(expectedResult >= 0.5 && reversedResult < 0.5)
							error += this.implementation.train(pair.data, GT_OUTPUT);
						else if(expectedResult < 0.5 && reversedResult < 0.5)
							error += this.implementation.train(
								pair.data,
								expectedResult < reversedResult ? LT_OUTPUT : GT_OUTPUT
							);
						else
							error += this.implementation.train(
								pair.data,
								expectedResult > reversedResult ? GT_OUTPUT : LT_OUTPUT
							);
						expected.put(pair.id, expectedResult);
						expected.put(pair.reverseId, (reversedResult + (1 - expectedResult)) / 2);
					}
				}
			}
		}
		return error / pairs.size();
	}
	
	
	@Override
	public double train(double[] inputs, double[] outputs) throws IOException{
		return this.implementation.train(inputs, outputs);
	}
	
	@Override
	public double train(double[] inputs, double[] outputs, double learningRate) throws IOException{
		return this.implementation.train(inputs, outputs, learningRate);
	}
	
	
	// serialization
	private void writeObject(ObjectOutputStream output) throws IOException{
		output.writeObject(this.implementation);
	}
	
	private void readObject(ObjectInputStream input) throws IOException{
		try{
			Field implementationField =
				NeuralNetworkComparator.class.getDeclaredField("implementation");
			implementationField.setAccessible(true);
			implementationField.set(this, input.readObject());
		}
		catch(IOException e){
			throw e;
		}
		catch(Throwable t){
			throw new IOException(t);
		}
	}
	
	
	
	private static class TrainingData{
		
		private final double data[];
		private final long containerId;
		
		
		public TrainingData(double data[], long containerId){
			this.data = data;
			this.containerId = containerId;
		}
	}
	
	
	private static class OrderedPair{
		
		private final String id, reverseId;
		private final boolean same;
		private final double data[];
		
		
		public OrderedPair(TrainingData a, TrainingData b){
			this.data = new double[a.data.length + b.data.length];
			
			System.arraycopy(a.data, 0, this.data, 0, a.data.length);
			System.arraycopy(b.data, 0, this.data, a.data.length, b.data.length);
			
			this.id =
				Long.toHexString(a.containerId) + "#" + Long.toHexString(b.containerId);
			this.reverseId =
				Long.toHexString(b.containerId) + "#" + Long.toHexString(a.containerId);
			
			this.same = (a.containerId == b.containerId);
		}
		
		
		@Override
		public int hashCode(){
			// TODO cache hash if it has any performance benefit
			return this.id.hashCode();
		}
		
		@Override
		public boolean equals(Object other){
			if(other instanceof OrderedPair == false)
				return false;
			
			return this.id == ((OrderedPair)other).id;
		}
	}
}
