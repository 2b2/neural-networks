package de.ef.fastflood.opencl;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.Locale;

import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_context;
import org.jocl.cl_device_id;
import org.jocl.cl_program;

import static org.jocl.CL.*;

class ProgramBuilder{
	
	private ProgramBuilder(){}
	
	
	
	static cl_program loadAndBuildProgram(cl_context context, cl_device_id device, String file, boolean cache){
		try{
			String deviceName =
				FastFloodOpenCLContext
				.getDeviceName(device)
				.toLowerCase(Locale.ENGLISH)
				.replaceAll("[^a-zA-Z0-9()]", "_");
			
			File cacheDir = new File("./cache/");
			File binaryFile = new File(cacheDir, file + "." + deviceName + ".bin");
			
			if(cache == true && binaryFile.exists() && binaryFile.isFile()){
				InputStream binary = new FileInputStream(binaryFile);
				
				byte programBytes[] = readAllBytes(binary);
				binary.close();
				
				long lengths[] = new long[]{programBytes.length};
				byte binaries[][] = new byte[][]{programBytes};
				
				cl_program program =
					clCreateProgramWithBinary(context, 1, new cl_device_id[]{device}, lengths, binaries, null, null);
				clBuildProgram(program, 1, new cl_device_id[]{device}, null, null, null);
				
				return program;
			}
			else{
				// find caller
				StackTraceElement stack[] = Thread.currentThread().getStackTrace();
				Class<?> relative = Class.forName(stack[2].getClassName());
				
				InputStream source = relative.getResourceAsStream(file);
				
				String programSource = new String(readAllBytes(source), StandardCharsets.UTF_8);
				source.close();
				
				cl_program program = clCreateProgramWithSource(context, 1, new String[]{programSource}, null, null);
				clBuildProgram(program, 1, new cl_device_id[]{device}, null, null, null);
				
				if(cache == true){
					cacheDir.mkdirs();
					if(binaryFile.createNewFile() && binaryFile.canWrite()){
						long sizes[] = new long[1];
						clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, Sizeof.size_t, Pointer.to(sizes), null);
						
						byte binaryBuffer[] = new byte[(int)sizes[0]];
						Pointer bufferPointer[] = new Pointer[]{Pointer.to(binaryBuffer)};
						clGetProgramInfo(program, CL_PROGRAM_BINARIES, Sizeof.POINTER , Pointer.to(bufferPointer), null);
						
						OutputStream binary = new FileOutputStream(binaryFile);
						binary.write(binaryBuffer);
						binary.close();
					}
				}
				
				return program;
			}
		}
		catch(Throwable t){
			throw new RuntimeException(t);
		}
	}
	
	private static byte[] readAllBytes(InputStream stream) throws IOException{
		int read;
		byte buffer[] = new byte[8192], bytes[] = null;
		while((read = stream.read(buffer)) != -1){
			if(bytes == null){
				// first block -> create buffer and copy read bytes
				bytes = new byte[read];
				System.arraycopy(buffer, 0, bytes, 0, read);
			}
			else{
				// new block -> create buffer with new length, copy old buffer and copy newly read bytes
				byte tmp[] = bytes;
				bytes = new byte[tmp.length + read];
				System.arraycopy(tmp, 0, bytes, 0, tmp.length);
				System.arraycopy(buffer, 0, bytes, tmp.length, read);
			}
		}
		return bytes == null ? new byte[0] : bytes;
	}
}
