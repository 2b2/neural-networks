package de.ef.neuralnetworks;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.stream.DoubleStream;

// TODO comment
// version: 1.1, date: 08.06.2016, author: Erik Fritzsche
public class AsyncNeuralNetwork{
	
	private NeuralNetwork network;
	private BlockingQueue<AsyncDataContainer> queue;
	private ExecutorService executor;
	
	
	public AsyncNeuralNetwork(NeuralNetwork network){
		this.network = network;
		this.queue = new LinkedBlockingQueue<AsyncDataContainer>();
		this.executor = Executors.newSingleThreadExecutor(
			r -> {Thread t = new Thread(r, "async-neural-network-executor"); t.setDaemon(true); return t;}
		);
		this.executor.execute(() -> this.asyncCalculate());
	}
	
	
	public Future<Double[]> calculate(double inputs[]){
		CompletableFuture<Double[]> outputsFuture = new CompletableFuture<Double[]>();
		this.queue.add(new AsyncDataContainer(inputs, outputsFuture));
		// TODO may hide that Future<Double[]> is CompletableFuture<Double[]>
		return (Future<Double[]>)outputsFuture;
	}
	
	// TODO may overwork
	private void asyncCalculate(){
		while(this.executor.isShutdown() == false){
			AsyncDataContainer container;
			try{
				container = this.queue.take();
			}catch(Throwable t){
				this.executor.shutdown();
				throw new RuntimeException(t);
			}
			double outputs[] = this.network.calculate(container.getInputs());
			container.getOutputsFuture().complete(
				DoubleStream.of(outputs).boxed().toArray(Double[]::new)
			);
		}
	}
	
	
	
	private class AsyncDataContainer{
		
		private double inputs[];
		private CompletableFuture<Double[]> outputsFuture;
		
		
		public AsyncDataContainer(double inputs[], CompletableFuture<Double[]> outputsFuture){
			this.inputs = inputs;
			this.outputsFuture = outputsFuture;
		}
		
		
		public double[] getInputs(){
			return this.inputs;
		}
		
		public CompletableFuture<Double[]> getOutputsFuture(){
			return this.outputsFuture;
		}
	}
}