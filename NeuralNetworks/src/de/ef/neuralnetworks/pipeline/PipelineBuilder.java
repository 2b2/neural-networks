package de.ef.neuralnetworks.pipeline;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

public class PipelineBuilder<I, O>{
	
	private Pipe<I, ?> root = null;
	private Pipe<?, ?> exit = null;
	private int count = 0;
	
	
	public PipelineBuilder(){}
	
	
	public <R> Pipe<I, R> root(Function<I, R> function){
		if(this.root != null)
			throw new IllegalStateException("Root pipe already set.");
		
		Pipe<I, R> root = new Pipe<>(function);
		this.root = root;
		return root;
	}
	
	public PipelineBuilder<I, O> singleFunction(Function<I, O> function){
		if(this.root != null)
			throw new IllegalStateException("Root pipe already set.");
		
		this.root = new ExitPipe<>(function);
		return this;
	}
	
	
	public Pipe<?, ?> last(){
		return this.exit;
	}
	
	@SuppressWarnings("unchecked")
	public Pipeline<I, O> build(){
		if(this.root == null)
			throw new IllegalStateException("Root pipe not set.");
		
		if(this.exit instanceof ExitPipe == false)
			throw new IllegalStateException("Bad exit pipe.");
		
		List<Function<Object, Object>> steps = new ArrayList<>(this.count);
		Pipe<?, ?> current = this.root;
		do{
			steps.add((Function<Object, Object>)current.function);
			
			current = current.next;
		}while(current != null);
		
		return new Pipeline<>(steps);
	}
	
	
	
	public class Pipe<I2, O2>{
		
		private final Function<I2, O2> function;
		
		private Pipe<O2, ?> next;
		
		
		private Pipe(Function<I2, O2> function){
			this.function = function;
			
			PipelineBuilder.this.exit = this;
			PipelineBuilder.this.count++;
		}
		
		
		public <R> Pipe<O2, R> pipe(Function<O2, R> function){
			Pipe<O2, R> next = new Pipe<>(function);
			this.next = next;
			return next;
		}
		
		public PipelineBuilder<I, O> exit(Function<O2, O> function){
			this.next = new ExitPipe<>(function);
			return PipelineBuilder.this;
		}
	}
	
	private class ExitPipe<I2>
		extends Pipe<I2, O>{
		
		private ExitPipe(Function<I2, O> function){
			super(function);
		}
		
		
		@Override
		public <R> Pipe<O, R> pipe(Function<O, R> function){
			throw new IllegalStateException("Can not create pipe after exit pipe.");
		}
		
		@Override
		public PipelineBuilder<I, O> exit(Function<O, O> function){
			throw new IllegalStateException("Can not create exit pipe after exit pipe.");
		}
	}
}
