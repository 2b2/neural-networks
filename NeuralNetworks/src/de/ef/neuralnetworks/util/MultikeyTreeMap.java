package de.ef.neuralnetworks.util;

import java.util.AbstractMap.SimpleEntry;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map.Entry;
import java.util.Queue;

public class MultikeyTreeMap<K, V>{
	
	private final Comparator<K> comparator;
	private Node root;
	
	
	public MultikeyTreeMap(Comparator<K> comparator){
		this.comparator = comparator;
	}
	
	
	private int compare(K key, List<K> list){
		int equal = 0, greater = 0, less = 0;
		int result;
		for(K other : list)
			if((result = this.comparator.compare(key, other)) == 0)
				equal++;
			else if(result > 0)
				greater++;
			else
				less++;
		if(equal > greater && equal > less) // TODO or >=
			return 0;
		else if(greater > less)
			return 1;
		return -1;
	}
	
	
	public boolean containsKey(K key){
		Entry<List<K>, V> entry = this.getEntry(key);
		if(entry == null)
			return false;
		return true;
	}
	
	public V get(K key){
		Entry<List<K>, V> entry = this.getEntry(key);
		if(entry == null)
			return null;
		return entry.getValue();
	}
	
	protected Entry<List<K>, V> getEntry(K key){
		Node parent = root;
		int result;
		while(parent != null)
			if((result = this.compare(key, parent.entry.getKey())) == 0)
				return parent.entry;
			else if(result > 0)
				parent = parent.right;
			else
				parent = parent.left;
		return null;
	}
	
	
	public void put(K key, V value){
		Node parent = null, node = this.root;
		int result = 0;
		while(node != null){
			parent = node;
			result = this.compare(key, node.entry.getKey());
			if(result == 0){
				if(value != node.entry.getValue())
					throw new IllegalArgumentException(""); // TODO good error text
				node.entry.getKey().add(key);
				return;
			}
			else if(result > 0)
				node = node.right;
			else
				node = node.left;
		}
		
		if(result == 0)
			node = this.root = new Node(value, null);
		else if(result > 0)
			node = parent.right = new Node(value, parent);
		else
			node = parent.left = new Node(value, parent);
		
		node.entry.getKey().add(key);
		
		this.balance(node);
	}
	
	// implements the four possible AVL-tree rotations
	// starting from the given node traversing up the tree
	private void balance(Node node){
		while(node != null){
			int leftHeight = this.height(node.left),
				rightHeight = this.height(node.right);
			
			node.height = Math.max(leftHeight, rightHeight) + 1;
			
			int balance = rightHeight - leftHeight;
			// right unbalanced
			if(balance > 1){
				int childBalance =
					this.height(node.right.right) - this.height(node.right.left);
				// rotate right-left
				if(childBalance < 0){
					// swap nodes around
					Node tmp = node;
					if(node.parent == null)
						node = this.root = tmp.right.left;
					else if(node.parent.left == node)
						node = node.parent.left = tmp.right.left;
					else
						node = node.parent.right = tmp.right.left;
					tmp.right.left = node.right;
					node.right = tmp.right;
					tmp.right = node.left;
					node.left = tmp;
					
					// update parent references
					node.parent = tmp.parent;
					node.left.parent = node;
					if(node.left.right != null)
						node.left.right.parent = node.left;
					node.right.parent = node;
					if(node.right.left != null)
						node.right.left.parent = node.right;
					
					// update heights
					node.left.height = Math.max(
						this.height(node.left.left),
						this.height(node.left.right)
					) + 1;
					node.right.height = Math.max(
						this.height(node.right.left),
						this.height(node.right.right)
					) + 1;
					node.height = Math.max(node.left.height, node.right.height) + 1;
				}
				// rotate left
				else{
					// swap nodes around
					Node tmp = node;
					if(node.parent == null)
						node = this.root = tmp.right;
					else if(node.parent.left == node)
						node = node.parent.left = tmp.right;
					else
						node = node.parent.right = tmp.right;
					tmp.right = node.left;
					node.left = tmp;
					
					// update parent references
					node.parent = tmp.parent;
					node.left.parent = node;
					if(node.left.right != null)
						node.left.right.parent = node.left;
					
					// update heights
					node.left.height = Math.max(
						this.height(node.left.left),
						this.height(node.left.right)
					) + 1;
					node.height = Math.max(node.left.height, node.right.height) + 1;
				}
			}
			// left unbalanced
			else if(balance < -1){
				int childBalance =
					this.height(node.left.right) - this.height(node.left.left);
				// rotate left-right
				if(childBalance > 0){
					// swap nodes around
					Node tmp = node;
					if(node.parent == null)
						node = this.root = tmp.left.right;
					else if(node.parent.left == node)
						node = node.parent.left = tmp.left.right;
					else
						node = node.parent.right = tmp.left.right;
					tmp.left.right = node.left;
					node.left = tmp.left;
					tmp.left = node.right;
					node.right = tmp;
					
					// update parent references
					node.parent = tmp.parent;
					node.left.parent = node;
					if(node.left.right != null)
						node.left.right.parent = node.left;
					node.right.parent = node;
					if(node.right.left != null)
						node.right.left.parent = node.right;
					
					// update heights
					node.left.height = Math.max(
						this.height(node.left.left),
						this.height(node.left.right)
					) + 1;
					node.right.height = Math.max(
						this.height(node.right.left),
						this.height(node.right.right)
					) + 1;
					node.height = Math.max(node.left.height, node.right.height) + 1;
				}
				// rotate right
				else{
					// swap nodes around
					Node tmp = node;
					if(node.parent == null)
						node = this.root = tmp.left;
					else if(node.parent.left == node)
						node = node.parent.left = tmp.left;
					else
						node = node.parent.right = tmp.left;
					tmp.left = node.right;
					node.right = tmp;
					
					// update parent references
					node.parent = tmp.parent;
					node.right.parent = node;
					if(node.right.left != null)
						node.right.left.parent = node.right;
					
					// update heights
					node.right.height = Math.max(
						this.height(node.right.left),
						this.height(node.right.right)
					) + 1;
					node.height = Math.max(node.left.height, node.right.height) + 1;
				}
			}
			
			node = node.parent;
		}
	}
	
	private int height(Node node){
		return node != null ? node.height : 0;
	}
	
	
	@Override
	public String toString(){
		StringBuilder builder = new StringBuilder();
		
		int height = Math.min(this.root.height, 10 /* TODO make max-height constant */);
		int level = 0, index = 0;
		
		Queue<Node> next = new LinkedList<>();
		next.add(this.root);
		
		builder.append("{(");
		while(next.isEmpty() == false && level < height){
			Node current = next.poll();
			if(current != null){
				next.add(current.left);
				next.add(current.right);
			
				builder.append(current.entry.getKey());
			}
			else{
				next.add(null);
				next.add(null);
			
				builder.append("[]");
			}
			
			if(++index == Math.pow(2, level)){
				index = 0;
				level++;
				
				builder.append("), (");
			}
		}
		builder.replace(builder.length() - 3, builder.length(), "}");
		
		return builder.toString();
	}
	
	
	
	private class Node{
		
		private final Entry<List<K>, V> entry;
		
		private Node parent, left = null, right = null;
		private int height = 1;
		
		
		public Node(V value, Node parent){
			this.entry = new SimpleEntry<>(new LinkedList<>(), value);
			
			this.parent = parent;
		}
	}
}
