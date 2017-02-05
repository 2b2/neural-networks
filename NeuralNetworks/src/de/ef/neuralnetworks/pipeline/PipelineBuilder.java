package de.ef.neuralnetworks.pipeline;

import java.util.function.Function;
import java.util.function.Predicate;

import de.ef.neuralnetworks.pipeline.Pipeline.Branch;
import de.ef.neuralnetworks.pipeline.Pipeline.Part;

public class PipelineBuilder<I, O>{
	
	private boolean hasRoot = false;
	private Pipe<?, ?> exit = null;
	
	
	public PipelineBuilder(){}
	
	
	public <R> Pipe<I, R> root(Function<I, R> function){
		if(this.hasRoot == true)
			throw new IllegalStateException("Root pipe already set.");
		
		this.hasRoot = true;
		return new Pipe<>(function, null);
	}
	
	public PipelineBuilder<I, O> singleFunction(Function<I, O> function){
		if(this.hasRoot == true)
			throw new IllegalStateException("Root pipe already set.");
		
		this.hasRoot = true;
		new ExitPipe<>(function, null);
		return this;
	}
	
	public Pipe<?, ?> last(){
		if(this.exit instanceof ExitPipe)
			throw new IllegalStateException("Can not create anything after exit.");
		
		return this.exit;
	}
	
	
	public Pipeline<I, O> build(){
		if(this.hasRoot == false)
			throw new IllegalStateException("Root pipe not set.");
		
		if(this.exit instanceof ExitPipe == false)
			throw new IllegalStateException("Bad exit pipe.");
		
		return new Pipeline<>(this.buildPart());
	}
	
	@SuppressWarnings("unchecked")
	private Part buildPart(){
		// going backwards from exit to start
		Pipe<Object, Object> pipe = (Pipe<Object, Object>)this.exit;
		Part part = null;
		while(pipe != null){
			if(pipe instanceof BranchPipe){
				BranchPipe<Object, Object> branchPipe = (BranchPipe<Object, Object>)pipe;
				part = new Branch(pipe.function, branchPipe.branchOff, part, branchPipe.branch.buildPart());
			}
			else part = new Part(pipe.function, part);
			
			pipe = (Pipe<Object, Object>)pipe.last;
		}
		
		return part;
	}
	
	
	
	public class Pipe<I2, O2>{
		
		private final Function<I2, O2> function;
		private final Pipe<?, ?> last;
		
		
		private Pipe(Function<I2, O2> function, Pipe<?, ?> last){
			this.function = function;
			this.last = last;
			
			PipelineBuilder.this.exit = this;
		}
		
		
		public <R> Pipe<O2, R> pipe(Function<O2, R> function){
			return new Pipe<>(function, this);
		}
		
		public <R> Pipe<O2, R> branch(Function<O2, R> function, Predicate<R> branchOff, PipelineBuilder<R, O> branch){
			return new BranchPipe<>(function, this, branchOff, branch);
		}
		
		public PipelineBuilder<I, O> exit(Function<O2, O> function){
			new ExitPipe<>(function, this);
			return PipelineBuilder.this;
		}
	}
	
	private class BranchPipe<I2, O2>
		extends Pipe<I2, O2>{
		
		private final Predicate<O2> branchOff;
		private final PipelineBuilder<O2, O> branch;
		
		
		private BranchPipe(Function<I2, O2> function, Pipe<?, ?> last, Predicate<O2> branchOff, PipelineBuilder<O2, O> branch){
			super(function, last);
			
			this.branchOff = branchOff;
			this.branch = branch;
		}
	}
	
	private class ExitPipe<I2>
		extends Pipe<I2, O>{
		
		private ExitPipe(Function<I2, O> function, Pipe<?, ?> last){
			super(function, last);
		}
	}
}
