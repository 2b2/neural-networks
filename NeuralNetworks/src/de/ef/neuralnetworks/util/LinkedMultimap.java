package de.ef.neuralnetworks.util;

import java.util.AbstractMap.SimpleImmutableEntry;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

/**
 * ...
 * 
 * @param <K> ...
 * @param <V> ...
 * 
 * @author Erik Fritzsche
 * @version 1.0
 * @since 1.0
 */
public class LinkedMultimap<K, V>{
	
	private final Map<K, List<V>> keyMap;
	private final MultikeyTreeMap<V, K> valueTree;
	
	
	public LinkedMultimap(Comparator<V> comparator){
		// TODO thread-safe, concurrent
		this.keyMap = new HashMap<>();
		this.valueTree = new MultikeyTreeMap<>(comparator);
	}
	
	
	public boolean containsKey(K key){
		return this.keyMap.containsKey(key);
	}
	
	public List<V> get(K key){
		List<V> values = this.keyMap.get(key);
		if(values == null)
			return Collections.emptyList();
		return values;
	}
	
	public Entry<K, List<V>> getByValue(V value){
		K key = this.valueTree.get(value);
		return new SimpleImmutableEntry<>(key, this.keyMap.get(key));
	}
	
	
	public void put(K key, V value){
		List<V> values = this.keyMap.get(key);
		if(values == null)
			this.keyMap.put(key, values = new LinkedList<>()); // TODO thread-safe
		values.add(value);
		
		this.valueTree.put(value, key);
	}
}
