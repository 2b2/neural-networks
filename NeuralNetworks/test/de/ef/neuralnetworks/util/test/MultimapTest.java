package de.ef.neuralnetworks.util.test;

import java.lang.reflect.Field;

import org.junit.Assert;
import org.junit.Test;

import de.ef.neuralnetworks.util.LinkedMultimap;
import de.ef.neuralnetworks.util.MultikeyTreeMap;

public class MultimapTest{
	
	public MultimapTest(){}
	
	
	@Test
	public void test() throws Exception{
		LinkedMultimap<Long, String> map =
			new LinkedMultimap<>((a, b) -> Character.compare(a.charAt(0), b.charAt(0)));
		
		map.put(5L, "5a");
		map.put(5L, "5b");
		map.put(6L, "6a");
		map.put(7L, "7a");
		map.put(9L, "9a");
		map.put(8L, "8a");
		map.put(4L, "4a");
		map.put(3L, "3a");
		map.put(2L, "2a");
		map.put(0L, "0a");
		map.put(1L, "1a");
		map.put(5L, "HOLY SH*T it works");
		
		Field treeField = LinkedMultimap.class.getDeclaredField("valueTree");
		treeField.setAccessible(true);
		MultikeyTreeMap<?, ?> tree = (MultikeyTreeMap<?, ?>)treeField.get(map);
		
		Field rootField = MultikeyTreeMap.class.getDeclaredField("root");
		rootField.setAccessible(true);
		Object root = rootField.get(tree);
		Field heightField = root.getClass().getDeclaredField("height");
		heightField.setAccessible(true);
		int height = (int)heightField.get(root);
		
		if(height != 4)
			Assert.fail("The height of the AVL-tree sould be 4.");
		Assert.assertEquals(map.get(5L), map.getByValue("5b").getValue());
	}
}
