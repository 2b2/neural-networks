package de.ef.neuralnetworks.utils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import de.ef.neuralnetworks.NeuralNetwork;

// TODO comment
// version: 1.1, date: 14.06.2016, author: Erik Fritzsche
public final class NeuralNetworkTraining{
	
	public final static int DEFAULT_VALIDATION_SET_SIZE_PERCENT = 20, DEFAULT_AUTO_SHUFFLE = 10;
	
	
	
	private final NeuralNetwork network;
	private final List<DataSet> dataSets;
	private final int validationSetSize, autoShuffle;
	private final double learningRate = 1, targetError = 0.001;
	private final ExecutorService executor;
	private double lastError = 0;
	private boolean isRunning, isInterrupted;
	
	
	public NeuralNetworkTraining(NeuralNetwork network, List<DataSet> dataSets){
		this(network, dataSets, DEFAULT_VALIDATION_SET_SIZE_PERCENT, DEFAULT_AUTO_SHUFFLE);
	}
	
	public NeuralNetworkTraining(NeuralNetwork network, List<DataSet> dataSets, int validationSetSizePercent){
		this(network, dataSets, validationSetSizePercent, DEFAULT_AUTO_SHUFFLE);
	}
	
	// TODO make sure 0% <= validationSetSizePercent <= 50%
	// autoShuffle with zero means no autoSuffle
	public NeuralNetworkTraining(NeuralNetwork network, List<DataSet> dataSets, int validationSetSizePercent, int autoShuffle){
		this.network = network;
		this.dataSets = dataSets;
		this.validationSetSize = (int)(dataSets.size() * (validationSetSizePercent / 100.0));
		this.autoShuffle = autoShuffle <= 0 ? 0 : autoShuffle;
		this.executor = Executors.newSingleThreadExecutor(
			r -> {Thread t = new Thread(r, "neural-network-training-executor"); t.setDaemon(true); return t;}
		);
	}
	
	
	public synchronized void startTraining(){
		if(this.isRunning == false){
			this.isInterrupted = false;
			this.executor.execute(() -> this.asyncTraining());
		}
	}
	
	public synchronized void stopTraining(){
		if(this.isRunning == true){
			this.isInterrupted = true;
		}
	}
	
	private void asyncTraining(){
		try{
			int autoShuffleIndex = 0;
			TrainingSet sets = this.generateTrainingSet();
		
			double error = this.calculateError(sets.getValidationSets());
			while(this.isInterrupted == false && error > this.targetError){
				for(DataSet trainingSet : sets.getTrainingSets()){
					if(this.isInterrupted == true){
						throw new InterruptedException();
					}
					this.network.train(trainingSet.getInputs(), trainingSet.getOutputs(), this.learningRate);
				}
				error = this.calculateError(sets.getValidationSets());
				this.lastError = error;
				
				autoShuffleIndex++;
				if(autoShuffleIndex == this.autoShuffle){
					sets = this.generateTrainingSet();
					autoShuffleIndex = 0;
				}
			}
		}catch(InterruptedException e){
			return;
		}finally{
			this.isRunning = false;
		}
	}
	
	
	public double getLastError(){
		return this.lastError;
	}
	
	
	private TrainingSet generateTrainingSet(){
		List<DataSet> trainingSets = new ArrayList<DataSet>(this.dataSets.size() - this.validationSetSize);
		List<DataSet> validationSets = new ArrayList<DataSet>(this.validationSetSize);
		Collections.copy(trainingSets, this.dataSets);
		Collections.shuffle(trainingSets);
		for(int i = 0; i < this.validationSetSize; i++){
			validationSets.add(trainingSets.remove(i));
		}
		return new TrainingSet(trainingSets, validationSets);
	}
	
	
	private double calculateError(List<DataSet> dataSets) throws InterruptedException{
		double error = 0.0;
		for(DataSet dataSet : dataSets){
			if(this.isInterrupted == true){
				throw new InterruptedException();
			}
			double[] outputs = this.network.calculate(dataSet.getInputs());
			for(int i = 0; i < outputs.length; i++){
				error += Math.abs(dataSet.getOutputs()[i] - outputs[i]);
			}
		}
		return error;
	}
	
	
	
	public static class DataSet{
		
		private final double inputs[], outputs[];
		
		
		public DataSet(double inputs[], double outputs[]){
			this.inputs = inputs;
			this.outputs = outputs;
		}
		
		
		double[] getInputs(){
			return this.inputs;
		}
		
		double[] getOutputs(){
			return this.outputs;
		}
	}
	
	
	private class TrainingSet{
		
		private final List<DataSet> trainingSets, validationSets;
		
		
		TrainingSet(List<DataSet> trainingSets, List<DataSet> validationSets){
			this.trainingSets = trainingSets;
			this.validationSets = validationSets;
		}
		
		
		List<DataSet> getTrainingSets(){
			return this.trainingSets;
		}
		
		List<DataSet> getValidationSets(){
			return this.validationSets;
		}
	}
}