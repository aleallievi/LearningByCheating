package cma.examples;
import java.io.FileWriter;
import java.io.PrintWriter;

import cma.*;
import cma.fitness.IObjectiveFunction;

//Command Line Iterations : IterNo
public class CMAMyNewExample {
	static PrintWriter pri;
	static String exppath = "/home/suyog/cmaes/temp/";
	public static void main(String[] args) {
		
		int curIter = Integer.parseInt(args[0]);
		if(curIter ==1)
		{			
			CMAEvolutionStrategy cma = new CMAEvolutionStrategy();
			cma.readProperties(); // read options`, see file CMAEvolutionStrategy.properties
			cma.setDimension(11); // overwrite some loaded properties
			double initX[]={0.1,-0.05,-4,1.2,7,8,-99,-0.6,0.8,3.4,1.5};
			cma.setInitialStandardDeviation(0.2); // also a mandatory setting
			double[][] pop = cma.samplePopulation(); // get a new population of solutions
			
		
			
		}
		
		else
		{
			double initX[]={0.1,-0.05,-4,1.2,7,8,-99,-0.6,0.8,3.4,1.5};
			double initSigma = 0.2;
			try
			{
				pri = new PrintWriter(new FileWriter(exppath+"file1.txt", true));
			
				for (int i = 0; i < initX.length; i++)
				{
					pri.println(initX[i]);
				
				}
			}
			catch (Exception e)
			{
				e.printStackTrace();
				System.out.println("No such file exists.");
			}			
			
		}
		
		
		/*
		EuclideanNew fitfun = new EuclideanNew();

		// new a CMA-ES and set some initial values
		CMAEvolutionStrategy cma = new CMAEvolutionStrategy();
		cma.readProperties(); // read options`, see file CMAEvolutionStrategy.properties
		cma.setDimension(11); // overwrite some loaded properties
		double initX[]={0.1,-0.05,-4,1.2,7,8,-99,-0.6,0.8,3.4,1.5};
		cma.setInitialStandardDeviation(0.2); // also a mandatory setting 
		double[] fitness = cma.init();  // new double[cma.parameters.getPopulationSize()];

		while(cma.stopConditions.getNumber() == 0) {

            // --- core iteration step ---
			System.out.println("Hello------>"+cma.sigma);
			double[][] pop = cma.samplePopulation(); // get a new population of solutions
			System.out.println(pop[0].length);
			for(int i = 0; i < pop.length; ++i) {    // for each candidate solution i
            	fitness[i] = fitfun.valueOf(pop[i]); // fitfun.valueOf() is to be minimized
			}
			cma.updateDistribution(fitness);         // pass fitness array to update search distribution
			
    		}*/
	} // main  
} // class
