package cma.examples;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.io.Reader;
import java.io.StreamTokenizer;

import cma.*;

//Command Line Iterations : IterNo
public class CMARoboKick {
	static PrintWriter pri;
	static String exppath = "";
	static Reader r;
	
	static String paramNames[] = {
			  "agent2beamX",
			  "agent2beamY", 
			  "agent2beamAngle",     
			  //"forwardKickScale",  
			  "returnKickScale",     
			  "forwardKickScaleLL1", 
			  "forwardKickScaleLL2",  
			  "forwardKickScaleLL3",  
			  "forwardKickScaleLL4",  
			  "forwardKickScaleLL5",  
			  "forwardKickScaleLL6",  
			  "targetHeight",         
			  "angleAtKick"    
			};     
	static double initParams[] = {-0.718,-0.125,5.2,1,1,1,1,1,1,1,0.05,0};
	//static double initParams[] = {-0.718,-0.125,5.2,1,0.05,0};
	static double initSigma[]={0.01,0.01,1,1,1,1,1,1,1,1,0.005,5.0};
	//static double initSigma[]={0.01,0.01,1,1,0.005,5.0};
	//static double initSigma[]={1,1,1,1,1,1,1,1,1,1,1,1};

	static int noParams = 12;
	static int popSize = 0;
	static int totIter = 0;
	public static void writeParams(double pop[][],int curIter)
	{
		try
		{
			for(int i =0;i<pop.length;i++)
			{
				pri = new PrintWriter(new FileWriter(exppath+"params_"+curIter+"_i_"+i+".txt", true));				
				for (int j = 0; j < noParams; j++)
				{					
					pri.println(paramNames[j]+"\t"+pop[i][j]);
			
				}
				pri.close();
			}				
			pri = new PrintWriter(new FileWriter(exppath+"paramswritten_"+curIter+".txt", true));				
			//pri.println("\n");
			pri.close();
		}
		catch (Exception e)
		{
			e.printStackTrace();
			System.out.println("No such file exists.");
		}							

	
	}
	
	public static double[] readFitness(int curIter)
	{
		double fitness[] = new double[popSize];
		try
		{			
			for(int i =0;i<popSize;i++)
				{
				r = new BufferedReader(new FileReader(exppath+"value_"+curIter+"_i_"+i+".txt"));
			    StreamTokenizer stok = new StreamTokenizer(r);
			    stok.parseNumbers();
			    stok.nextToken();				    
			    while (stok.ttype != StreamTokenizer.TT_EOF) {
			        if (stok.ttype == StreamTokenizer.TT_NUMBER){
			           fitness[i]=stok.nval;
			        }			          
			        stok.nextToken();
			      }
			    r.close();
				}
		}
		
		catch (Exception e)
		{
			e.printStackTrace();
			System.out.println("No such file exists.");
		}
		return fitness;
	}
	
	public static void main(String[] args) throws InterruptedException {
		
		
		int curIter = 1;
		exppath = args[0];
		totIter = Integer.parseInt(args[1]);
		popSize = Integer.parseInt(args[2]);
		System.out.println(exppath);
		System.out.println(""+totIter);
		CMAEvolutionStrategy cma = new CMAEvolutionStrategy();
		cma.readProperties(); // read options`, see file CMAEvolutionStrategy.properties
		cma.setDimension(noParams); // overwrite some loaded properties
		cma.parameters.setPopulationSize(popSize);
	    cma.setInitialX(initParams);
	    cma.setInitialStandardDeviations(initSigma); // also a mandatory setting
	    double[] fitness = cma.init();
	    while(curIter<=totIter)
	    {
	    	double[][] pop = cma.samplePopulation(); // get a new population of solutions	    	
	 		writeParams(pop,curIter);
	 		File f = new File(""+exppath+"valuationdone_"+curIter+".txt");
	 		while(true)
	 		{
	 			//wait
	 			if(f.exists()){
	 				System.out.println("Exists");
	 				break;
	 			}
	 			else
	 			{
	 				Thread.sleep(2500);
	 				System.out.println("Not Exists");
	 				//break;
	 			}
	 			
	 			
	 		}
	 		//Read fitness function
	 		fitness = readFitness(curIter);
			cma.updateDistribution(fitness);        // pass fitness array to update search distribution
	    	curIter = curIter+1;
	    }
		
	} // main  
} // class
