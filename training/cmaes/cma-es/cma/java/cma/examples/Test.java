package cma.examples;
import java.io.*;
class Test{
	static PrintWriter pri;
	public static void main(String args[]){
		try
		{
			pri = new PrintWriter(new FileWriter("/home/suyog/cmaes/temp/file1.txt", true));
			double vals[]={1,2,3.0,4.5};
			for (int i = 0; i < vals.length; i++)
			{
				pri.println(vals[i]);
				
			}
		}
 
		catch (Exception e)
		{
			e.printStackTrace();
			System.out.println("No such file exists.");
		}
 
		finally
		{
			pri.close();
		}
	
	}
}