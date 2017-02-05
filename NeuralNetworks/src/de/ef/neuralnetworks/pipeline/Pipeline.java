package de.ef.neuralnetworks.pipeline;

import java.util.function.Function;
import java.util.function.Predicate;

public class Pipeline<I, O>{
	
	private final Part root;
	
	
	Pipeline(Part root){
		this.root = root;
	}
	
	
	@SuppressWarnings("unchecked")
	public O process(I input){
		Part part = this.root;
		Object output = input;
		while(part != null){
			output = part.function.apply(output);
			
			if(part instanceof Branch && ((Branch)part).branchOff.test(output) == true)
				part = ((Branch)part).branch;
			else
				part = part.next;
		}
		return (O)output;
	}
	
	
	
	static class Part{
		
		private final Function<Object, Object> function;
		private final Part next;
		
		
		public Part(Function<Object, Object> function, Part next){
			this.function = function;
			this.next = next;
		}
	}
	
	static class Branch
		extends Part{
		
		private final Predicate<Object> branchOff;
		private final Part branch;
		
		public Branch(Function<Object, Object> function, Predicate<Object> branchOff, Part next, Part branch){
			super(function, next);
			
			this.branchOff = branchOff;
			this.branch = branch;
		}
	}
}
