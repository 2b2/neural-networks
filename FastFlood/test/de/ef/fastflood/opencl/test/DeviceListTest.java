package de.ef.fastflood.opencl.test;

import java.util.List;

import org.junit.Assert;
import org.junit.Test;

import de.ef.fastflood.opencl.FastFloodOpenCLContext;
import de.ef.fastflood.opencl.FastFloodOpenCLContext.OpenCLConfiguration;

public class DeviceListTest{
	
	public DeviceListTest(){}
	
	
	@Test
	public void test(){
		List<OpenCLConfiguration> devices = 
			FastFloodOpenCLContext.listConfigurations();
		
		Assert.assertNotNull(devices);
		
		System.out.println(devices);
	}
}
